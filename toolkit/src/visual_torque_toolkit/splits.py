"""Leakage-resistant probe and intervention splits."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from .generate import GenerationConfig
from .prompts import QuestionRecord
from .world import CounterfactualPair, ProblemType, Scene


PROBE_SPLITS = ("probe_train", "probe_dev", "probe_test")
INTERVENTION_SPLITS = ("intervention_dev", "intervention_test")


def _bucket(value: object, modulus: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value!r}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulus


@dataclass(frozen=True, slots=True)
class SplitConfig:
    heldout_fraction: float = 0.20
    dev_modulus: int = 10
    intervention_dev_modulus: int = 4
    seed: int = 20260817

    def __post_init__(self) -> None:
        if not 0.0 < self.heldout_fraction < 1.0:
            raise ValueError("heldout_fraction must lie strictly between 0 and 1")
        if self.dev_modulus < 2 or self.intervention_dev_modulus < 2:
            raise ValueError("split moduli must be at least 2")


@dataclass(frozen=True, slots=True)
class SplitManifest:
    scene_split: dict[str, str]
    pair_split: dict[str, str]
    question_split: dict[str, str]
    heldout_unit_distance: tuple[tuple[int, int], ...]

    def to_record(self) -> dict[str, Any]:
        grouped_scenes = {
            split: sorted(scene_id for scene_id, value in self.scene_split.items() if value == split)
            for split in PROBE_SPLITS
        }
        grouped_pairs = {
            split: sorted(pair_id for pair_id, value in self.pair_split.items() if value == split)
            for split in INTERVENTION_SPLITS
        }
        grouped_questions = {
            split: sorted(question_id for question_id, value in self.question_split.items() if value == split)
            for split in PROBE_SPLITS
        }
        return {
            "probe_scenes": grouped_scenes,
            "intervention_pairs": grouped_pairs,
            "questions": grouped_questions,
            "heldout_unit_distance_combinations": [list(item) for item in self.heldout_unit_distance],
        }


def select_heldout_combinations(
    generation: GenerationConfig,
    split: SplitConfig,
) -> tuple[tuple[int, int], ...]:
    combinations = [
        (units, distance)
        for units in range(1, generation.max_units + 1)
        for distance in range(1, generation.max_distance + 1)
    ]
    combinations.sort(key=lambda item: _bucket(item, 2**31 - 1, split.seed))
    count = max(1, round(len(combinations) * split.heldout_fraction))
    count = min(count, len(combinations) - 1)
    return tuple(sorted(combinations[:count]))


def _is_multi_structure_holdout(scene: Scene, config: SplitConfig) -> bool:
    if scene.problem_type is not ProblemType.MULTI_STACK:
        return False
    structure = tuple(
        sorted(
            (
                stack.side.value,
                len([candidate for candidate in scene.stacks if candidate.side is stack.side]),
                stack.distance,
            )
            for stack in scene.stacks
        )
    )
    return _bucket(structure, 10, config.seed + 17) == 0


def build_splits(
    scenes: tuple[Scene, ...],
    pairs: tuple[CounterfactualPair, ...],
    questions: tuple[QuestionRecord, ...],
    generation: GenerationConfig,
    config: SplitConfig = SplitConfig(),
) -> SplitManifest:
    heldout = select_heldout_combinations(generation, config)
    heldout_set = set(heldout)
    scene_split: dict[str, str] = {}
    for scene in scenes:
        contains_heldout = any(
            (stack.units, stack.distance) in heldout_set for stack in scene.stacks
        )
        if contains_heldout or _is_multi_structure_holdout(scene, config):
            value = "probe_test"
        elif _bucket(scene.scene_id, config.dev_modulus, config.seed) == 0:
            value = "probe_dev"
        else:
            value = "probe_train"
        scene_split[scene.scene_id] = value

    family_split: dict[str, str] = {}
    pair_split: dict[str, str] = {}
    for pair in pairs:
        family = pair.equivalence_key
        if family not in family_split:
            is_dev = _bucket(family, config.intervention_dev_modulus, config.seed) == 0
            family_split[family] = "intervention_dev" if is_dev else "intervention_test"
        pair_split[pair.pair_id] = family_split[family]

    question_split = {
        question.question_id: scene_split[question.scene_id] for question in questions
    }
    manifest = SplitManifest(scene_split, pair_split, question_split, heldout)
    validate_splits(manifest, scenes, pairs, questions)
    return manifest


def validate_splits(
    manifest: SplitManifest,
    scenes: tuple[Scene, ...],
    pairs: tuple[CounterfactualPair, ...],
    questions: tuple[QuestionRecord, ...],
) -> None:
    scene_ids = {scene.scene_id for scene in scenes}
    pair_ids = {pair.pair_id for pair in pairs}
    question_ids = {question.question_id for question in questions}
    if set(manifest.scene_split) != scene_ids:
        raise ValueError("scene split does not cover exactly the generated scenes")
    if set(manifest.pair_split) != pair_ids:
        raise ValueError("pair split does not cover exactly the generated pairs")
    if set(manifest.question_split) != question_ids:
        raise ValueError("question split does not cover exactly the generated questions")
    if not set(manifest.scene_split.values()) <= set(PROBE_SPLITS):
        raise ValueError("unknown probe split name")
    if not set(manifest.pair_split.values()) <= set(INTERVENTION_SPLITS):
        raise ValueError("unknown intervention split name")

    heldout = set(manifest.heldout_unit_distance)
    for scene in scenes:
        if manifest.scene_split[scene.scene_id] == "probe_test":
            continue
        if any((stack.units, stack.distance) in heldout for stack in scene.stacks):
            raise ValueError("held-out unit-distance combination leaked outside probe_test")

    family_assignments: dict[str, str] = {}
    for pair in pairs:
        value = manifest.pair_split[pair.pair_id]
        previous = family_assignments.setdefault(pair.equivalence_key, value)
        if previous != value:
            raise ValueError("an intervention equivalence family crosses splits")
        if pair.target.scene_id not in scene_ids or pair.source.scene_id not in scene_ids:
            raise ValueError("counterfactual pair references a missing scene")

    for question in questions:
        expected = manifest.scene_split[question.scene_id]
        if manifest.question_split[question.question_id] != expected:
            raise ValueError("question split disagrees with its source scene")
