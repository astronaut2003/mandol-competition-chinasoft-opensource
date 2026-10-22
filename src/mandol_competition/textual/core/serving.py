"""Pinned vLLM launcher and read-only runtime identity endpoint.

The endpoint reads the running engine configuration, not environment assertions.
It exposes no environment variables, credentials, or administrative operations.
Compatibility boundary: vLLM 0.17.1. Bind these model servers to loopback.
"""

import argparse
import hashlib
import json
import os
import sys

from starlette.responses import JSONResponse

# Official vLLM v0.17.1 examples/pooling/score/template/qwen3_reranker.jinja
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
SCORE_TEMPLATE = """<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: {{ messages | selectattr("role", "eq", "system") | map(attribute="content") | first | default("Given a web search query, retrieve relevant passages that answer the query") }}
<Query>: {{ messages | selectattr("role", "eq", "query") | map(attribute="content") | first }}
<Document>: {{ messages | selectattr("role", "eq", "document") | map(attribute="content") | first }}<|im_end|>
<|im_start|>assistant
<think>

</think>{{ '\\n\\n' }}
"""
SCORE_TEMPLATE_HASH = hashlib.sha256(SCORE_TEMPLATE.encode()).hexdigest()
RERANK_OVERRIDES = {
    "architectures": ["Qwen3ForSequenceClassification"],
    "classifier_from_token": ["no", "yes"],
    "is_original_qwen3_reranker": True,
}


class QualificationWorkerExtension:
    """Optional fixed, read-only worker metrics; never accepts executable input."""

    def mandol_precision_profile(self):
        import torch
        import triton

        return {
            "triton_version": triton.__version__,
            "triton_dot_precision": triton.knobs.language.fp32_default,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        }

    def mandol_gpu_memory_profile(self):
        import torch

        torch.cuda.synchronize()
        runner = self.model_runner
        free, total = torch.cuda.mem_get_info()
        tensors = getattr(runner, "kv_caches", ())
        return {
            "pid": os.getpid(),
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "max_allocated": torch.cuda.max_memory_allocated(),
            "max_reserved": torch.cuda.max_memory_reserved(),
            "device_free": free,
            "device_total": total,
            "parameter_bytes": sum(
                p.numel() * p.element_size() for p in runner.model.parameters()
            ),
            "kv_tensor_bytes": sum(
                t.numel() * t.element_size()
                for t in tensors
                if isinstance(t, torch.Tensor)
            ),
        }


class RuntimeProfileMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] not in {
            "/mandol/runtime-profile",
            "/mandol/gpu-profile",
        }:
            return await self.app(scope, receive, send)
        if scope["method"] != "GET":
            return await JSONResponse({}, status_code=405)(scope, receive, send)
        try:
            state = scope["app"].state
            if scope["path"] == "/mandol/gpu-profile":
                extension = state.vllm_config.parallel_config.worker_extension_cls
                if (
                    extension
                    != "mandol_competition.textual.core.serving.QualificationWorkerExtension"
                ):
                    return await JSONResponse(
                        {"error": "qualification metrics disabled"}, status_code=404
                    )(scope, receive, send)
                metrics = await state.engine_client.collective_rpc(
                    "mandol_gpu_memory_profile", timeout=15
                )
                return await JSONResponse({"workers": metrics})(scope, receive, send)
            config = state.vllm_config.model_config
            scorer = state.openai_serving_scores
            template = getattr(scorer, "score_template", None) or ""
            payload = {
                "contract": "vllm-runtime-v1",
                "dtype": str(config.dtype).removeprefix("torch."),
                "runner": config.runner_type,
                "max_tokens": config.max_model_len,
                "score_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
                "hf_overrides": {
                    key: getattr(config.hf_config, key, None)
                    for key in RERANK_OVERRIDES
                },
                "enforce_eager": config.enforce_eager,
                "max_num_seqs": state.vllm_config.scheduler_config.max_num_seqs,
                "max_num_batched_tokens": state.vllm_config.scheduler_config.max_num_batched_tokens,
                "enable_chunked_prefill": state.vllm_config.scheduler_config.enable_chunked_prefill,
                "gpu_memory_utilization": state.vllm_config.cache_config.gpu_memory_utilization,
            }
            if template:
                payload["worker_precision"] = await state.engine_client.collective_rpc(
                    "mandol_precision_profile", timeout=15
                )
            response = JSONResponse(payload)
        except Exception:  # noqa: BLE001 -- failed runtime proof must return 503
            response = JSONResponse(
                {"error": "runtime profile unavailable"}, status_code=503
            )
        await response(scope, receive, send)


def select_launch_profile(task, profile_path=None):
    """Resolve an explicit finalist declaration without changing formal defaults."""
    from dataclasses import asdict
    from pathlib import Path

    from .models import ServingProfile
    from .profile import formal_profile
    from .qwen_reference import OFFICIAL_CONTEXT_TOKENS, REFERENCE_MODEL_REVISIONS

    if task not in {"embedding", "reranker"}:
        raise ValueError("unknown serving task")
    if profile_path is None:
        return formal_profile().options[task + "_serving"]
    if task != "reranker":
        raise ValueError("explicit finalist profile is only supported for reranking")
    declared = json.loads(Path(profile_path).read_text())
    if not isinstance(declared, dict):
        raise TypeError("serving profile must be a JSON object")
    profile = ServingProfile(**declared)
    if (
        profile.task != task
        or REFERENCE_MODEL_REVISIONS.get(profile.model) != profile.revision
        or not 1 <= profile.max_pair_tokens <= profile.max_tokens <= OFFICIAL_CONTEXT_TOKENS
    ):
        raise ValueError("unsupported finalist identity/task/context declaration")
    return asdict(profile)


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["embedding", "reranker"])
    parser.add_argument("--model-cache", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.22)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--qualification-metrics", action="store_true")
    parser.add_argument(
        "--serving-profile",
        type=Path,
        help="Explicit pinned Qwen finalist JSON; omitted keeps current formal defaults",
    )
    args = parser.parse_args()
    profile = select_launch_profile(args.task, args.serving_profile)
    if args.serving_profile and args.max_model_len not in (None, profile["max_tokens"]):
        parser.error("max-model-len cannot override an explicit frozen serving profile")
    effective_max_tokens = args.max_model_len or profile["max_tokens"]
    if args.max_num_batched_tokens < effective_max_tokens:
        parser.error("pooling token budget must fit one complete maximum-length input")
    snapshot = (
        Path(args.model_cache)
        / ("models--" + profile["model"].replace("/", "--"))
        / "snapshots"
        / profile["revision"]
    )
    if not snapshot.is_dir():
        raise RuntimeError("frozen model snapshot must be downloaded before launch")
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(snapshot.resolve()),
        "--served-model-name",
        profile["model"] + "@" + profile["revision"],
        "--runner",
        "pooling",
        "--dtype",
        profile["dtype"],
        "--max-model-len",
        str(effective_max_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--enforce-eager",
        # vLLM0.17.1's running-request scheduler reserves one sampled-token
        # slot, which can strand split maximum-length pooling prompts forever.
        # Keep each pooling prompt in one prefill step; no precision override.
        # Upstream mechanism/fix: vllm-project/vllm#48039.
        "--no-enable-chunked-prefill",
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--middleware",
        "mandol_competition.textual.core.serving.RuntimeProfileMiddleware",
    ]
    if args.qualification_metrics or args.task == "reranker":
        command += [
            "--worker-extension-cls",
            "mandol_competition.textual.core.serving.QualificationWorkerExtension",
        ]
    if args.task == "reranker":
        command += [
            "--hf-overrides",
            json.dumps(RERANK_OVERRIDES),
            "--chat-template",
            SCORE_TEMPLATE,
        ]
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
