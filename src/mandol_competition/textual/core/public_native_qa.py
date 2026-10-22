"""Public QA adapter: pinned AML official-public by default, legacy explicit."""

import json
from dataclasses import dataclass, field

from . import paired_quality
from . import reranker_quality as q
from .evaluation_protocol import DEFAULT_EVALUATION_PROTOCOL, EvaluationProtocol
from .paired_quality import file_sha256


@dataclass(frozen=True)
class NativePublicQaContract:
    dataset: str
    evaluator_path: str | None = None
    evaluation_protocol: str = DEFAULT_EVALUATION_PROTOCOL
    _official: object = field(init=False, repr=False, compare=False)
    _native_prompt: object = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        protocol = EvaluationProtocol(self.dataset, self.evaluation_protocol)
        object.__setattr__(
            self,
            "_official",
            protocol
            if self.evaluation_protocol == DEFAULT_EVALUATION_PROTOCOL
            else None,
        )
        object.__setattr__(self, "_native_prompt", None)
        if self._official:
            return
        if self.dataset not in {"locomo", "longmemeval"}:
            raise ValueError("Unsupported native public benchmark")
        if (self.dataset == "longmemeval") != bool(self.evaluator_path):
            raise ValueError("Only LongMemEval requires its pinned native evaluator")
        object.__setattr__(
            self,
            "_native_prompt",
            q.load_longmemeval_judge_prompt(self.evaluator_path)
            if self.evaluator_path
            else None,
        )

    @property
    def manifest(self):
        if self._official:
            return self._official.manifest
        return {
            "evaluation_protocol": self.evaluation_protocol,
            "status": "DIAGNOSTIC_ONLY",
            "contract": "existing-native-public-qa-on-api-content-v1",
            "dataset": self.dataset,
            "answer": "unchanged Q-1M _answer_messages",
            "judge": "pinned official LongMemEval get_anscheck_prompt"
            if self.dataset == "longmemeval"
            else "unchanged project LoCoMo semantic judge + native benchmark metric",
            "quality_source_sha256": file_sha256(q.__file__),
            "protocol_source_sha256": file_sha256(paired_quality.__file__),
            "contract_source_sha256": file_sha256(__file__),
            "official_evaluator_sha256": q.LONGMEMEVAL_JUDGE_SHA256
            if self.evaluator_path
            else None,
            "context": "actual API content only; no evaluator-side augmentation",
            "second_cycle_official_evaluator": "NOT_CLAIMED",
        }

    def _validate(self, qa):
        if qa.get("dataset") != self.dataset:
            raise ValueError("QA dataset differs from its frozen native contract")

    def answer_messages(self, qa, context):
        self._validate(qa)
        if self._official:
            # Preserve the input context string and official fallback buckets.
            # Dataset category/gold never enter the Answer request.
            item = {"question": qa["question"], "retrieved_context": context}
            return [
                {"role": "user", "content": self._official.render_answer_prompt(item)}
            ]
        return q._answer_messages(qa, context)

    def judge_messages(self, qa, answer):
        self._validate(qa)
        if self._official:
            gold = (
                "No information available"
                if self.dataset == "locomo" and qa.get("category") == 5
                else qa["answer"]
            )
            prompt = self._official.render_accuracy_prompt(
                {"question": qa["question"], "gold_answer": gold}, answer.strip()
            )
            return [{"role": "user", "content": prompt}], False
        if self._native_prompt:
            return [
                {
                    "role": "user",
                    "content": self._native_prompt(
                        qa["question_type"],
                        qa["question"],
                        qa["answer"],
                        answer,
                        abstention=qa["unanswerable"],
                    ),
                }
            ], False
        return q._locomo_judge_messages(qa, answer), True

    def answer_metrics(self, qa, answer):
        self._validate(qa)
        if self.dataset == "longmemeval":
            return {}
        gold = "No information available" if qa["category"] == 5 else qa["answer"]
        return {
            "locomo_native_score": q.locomo_native_score(answer, gold, qa["category"])
        }

    def parse_score(self, content):
        if self._official:
            return float(self._official.parse_judge_label(content.strip()) == "CORRECT")
        # Exact schema/normalization from paired_quality.generate_arm. Do not
        # accept a new public AML label schema as an implicit evaluator swap.
        if self.dataset == "locomo":
            parsed = json.loads(content)
            if (
                not isinstance(parsed, dict)
                or set(parsed) != {"correct", "reason"}
                or type(parsed["correct"]) is not bool
                or not isinstance(parsed["reason"], str)
            ):
                raise ValueError("Malformed project judge")
            return float(parsed["correct"])
        parsed = content.strip().lower().removesuffix(".")
        if parsed not in {"yes", "no"}:
            raise ValueError("Malformed official judge")
        return float(parsed == "yes")
