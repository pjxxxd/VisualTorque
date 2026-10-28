"""Configuration records for reproducible behavior experiments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from pathlib import Path
from typing import Any


class PromptCondition(str, Enum):
    IMAGE_RULE = "image_rule"
    SYMBOLIC_RULE = "symbolic_rule"
    IMAGE_NO_RULE = "image_no_rule"

    @property
    def uses_image(self) -> bool:
        return self is not PromptCondition.SYMBOLIC_RULE

    @property
    def supplies_rule(self) -> bool:
        return self is not PromptCondition.IMAGE_NO_RULE


@dataclass(frozen=True, slots=True)
class BehaviorRunConfig:
    dataset_dir: Path
    output_dir: Path
    model_id: str
    model_revision: str = "main"
    conditions: tuple[PromptCondition, ...] = (PromptCondition.IMAGE_RULE,)
    question_kinds: tuple[str, ...] = ("outcome",)
    splits: tuple[str, ...] = ("probe_train", "probe_dev", "probe_test")
    batch_size: int = 8
    max_scenes: int | None = None
    scenes_per_stratum: int | None = None
    max_examples: int | None = None
    generation_every: int = 0
    seed: int = 20260817
    resume: bool = True

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if not self.conditions or len(set(self.conditions)) != len(self.conditions):
            raise ValueError("conditions must be non-empty and unique")
        if not self.question_kinds or not self.splits:
            raise ValueError("question_kinds and splits must not be empty")
        for name in ("batch_size",):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("max_scenes", "scenes_per_stratum", "max_examples"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when provided")
        if self.generation_every < 0:
            raise ValueError("generation_every must be non-negative")

    def to_record(self) -> dict[str, Any]:
        return {
            "dataset_dir": str(self.dataset_dir.expanduser().resolve()),
            "output_dir": str(self.output_dir.expanduser().resolve()),
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "conditions": [condition.value for condition in self.conditions],
            "question_kinds": list(self.question_kinds),
            "splits": list(self.splits),
            "batch_size": self.batch_size,
            "max_scenes": self.max_scenes,
            "scenes_per_stratum": self.scenes_per_stratum,
            "max_examples": self.max_examples,
            "generation_every": self.generation_every,
            "seed": self.seed,
            "resume": self.resume,
        }

    def identity_record(self) -> dict[str, Any]:
        """Fields that must remain fixed when resuming an existing run."""

        record = self.to_record()
        record.pop("resume")
        record.pop("output_dir")
        record.pop("dataset_dir")
        record["dataset_fingerprint"] = self.dataset_fingerprint()
        return record

    def dataset_fingerprint(self) -> str:
        """Return a path-independent identity for the experiment dataset.

        Images are deterministic renders of ``scenes.jsonl`` under the stored
        dataset configuration. Hashing the canonical records and manifests is
        therefore enough to distinguish Dataset versions without tying a run
        to Colab's changing extraction directory.
        """

        root = self.dataset_dir.expanduser().resolve()
        names = (
            "dataset_config.json",
            "scenes.jsonl",
            "counterfactual_pairs.jsonl",
            "questions.jsonl",
            "splits.json",
            "statistics.json",
        )
        digest = hashlib.sha256()
        for name in names:
            path = root / name
            if not path.is_file():
                raise FileNotFoundError(
                    f"cannot fingerprint dataset; missing file: {path}"
                )
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            digest.update(b"\0")
        return f"sha256:{digest.hexdigest()}"
