"""Versioned evaluation semantics, independent of Search and model transport.

Only hash-verified, pure functions from the pinned public source are executed.
The public API config, HTTP helper and main are never imported or executed.
Private AML evaluator/model parity is unknown. Legacy scoring is opt-in.
"""

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from importlib.resources import files

DEFAULT_EVALUATION_PROTOCOL = "aml_official_public_v1"
LEGACY_EVALUATION_PROTOCOL = "legacy_local_v1"
EVALUATION_PROTOCOLS = (DEFAULT_EVALUATION_PROTOCOL, LEGACY_EVALUATION_PROTOCOL)
OFFICIAL_REVISION = "1b8142bfe0f20f1c5218d6b554aa0012de34e504"
SOURCE_HASHES = {
    "locomo": "31a49fc09e381bc117b59063c7a4cea543a672b23e1264aedb03290239671c1a",
    "longmemeval": "c7f17c363bd0803cfeedd2b793a0cb0a74ab119a70eeced2c7a835b2b2aecb46",
}
FUNCTIONS = (
    "memory_text",
    "render_answer_prompt",
    "gold_answer",
    "render_accuracy_prompt",
    "parse_judge_label",
)


def sha256(value):
    return hashlib.sha256(
        value.encode() if isinstance(value, str) else value
    ).hexdigest()


def validate_protocol(name):
    if name not in EVALUATION_PROTOCOLS:
        raise ValueError(
            "Unknown evaluation protocol; no legacy fallback: " + str(name)
        )
    return name


@dataclass(frozen=True)
class EvaluationProtocol:
    dataset: str
    name: str = DEFAULT_EVALUATION_PROTOCOL
    official_revision: str = OFFICIAL_REVISION

    def __post_init__(self):
        validate_protocol(self.name)
        if self.dataset not in SOURCE_HASHES:
            raise ValueError("Unsupported public QA dataset")
        if self.official_revision != OFFICIAL_REVISION:
            raise ValueError(
                "Unknown official revision; explicit protocol version required"
            )
        # Eager verification: fail before a model call, never silently fall back.
        if self.name == DEFAULT_EVALUATION_PROTOCOL:
            self._load()

    def _load(self):
        raw = (
            files("mandol_competition.textual.core")
            .joinpath("protocol_sources", self.dataset + ".txt")
            .read_bytes()
        )
        if sha256(raw) != SOURCE_HASHES[self.dataset]:
            raise ValueError("Pinned official source hash mismatch")
        source = raw.decode("utf-8")
        tree = ast.parse(source)
        env = {"json": json, "re": re}
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id
                in {"OPEN_ENDED_ANSWER_TEMPLATE", "ACCURACY_PROMPT"}
            ):
                env[node.targets[0].id] = ast.literal_eval(node.value)
        funcs = [
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS
        ]
        if {n.name for n in funcs} != set(FUNCTIONS):
            raise ValueError("Incomplete official pure-function contract")
        hashes = {n.name: sha256(ast.get_source_segment(source, n)) for n in funcs}
        exec(  # noqa: S102 -- SHA-pinned pure functions only
            compile(
                ast.Module(body=funcs, type_ignores=[]), "<pinned-aml-public>", "exec"
            ),
            env,
        )
        return env, hashes

    @property
    def manifest(self):
        if self.name == LEGACY_EVALUATION_PROTOCOL:
            return {
                "evaluation_protocol": self.name,
                "status": "DIAGNOSTIC_ONLY",
                "official_revision": None,
            }
        env, hashes = self._load()
        renderer_hash = sha256(
            json.dumps(
                {k: hashes[k] for k in FUNCTIONS if k != "parse_judge_label"},
                sort_keys=True,
            )
        )
        return {
            "evaluation_protocol": self.name,
            "official_revision": self.official_revision,
            "dataset": self.dataset,
            "source_sha256": SOURCE_HASHES[self.dataset],
            "answer_prompt_sha256": sha256(env["OPEN_ENDED_ANSWER_TEMPLATE"]),
            "judge_prompt_sha256": sha256(env["ACCURACY_PROMPT"]),
            "renderer_version": "aml-public-pure-functions-v1",
            "renderer_sha256": renderer_hash,
            "judge_parser_version": "aml-public-parse-judge-label-v1",
            "judge_parser_sha256": hashes["parse_judge_label"],
            "function_sha256": hashes,
            "PROMPT_PROTOCOL_PARITY": "EXACT_PUBLIC",
            "PRIVATE_MODEL_PARITY": "UNKNOWN",
            "PRIVATE_AML_EVALUATOR": "UNKNOWN",
            "PUBLIC_HELPER_TOKEN_BUDGET_TRANSPORT_PARITY": "PARTIAL",
        }

    def _function(self, name):
        if self.name != DEFAULT_EVALUATION_PROTOCOL:
            raise ValueError("Legacy protocol requires the explicit legacy adapter")
        return self._load()[0][name]

    def render_answer_prompt(self, item):
        return self._function("render_answer_prompt")(item)

    def render_accuracy_prompt(self, item, generated_answer):
        return self._function("render_accuracy_prompt")(item, generated_answer)

    def parse_judge_label(self, response):
        return self._function("parse_judge_label")(response)
