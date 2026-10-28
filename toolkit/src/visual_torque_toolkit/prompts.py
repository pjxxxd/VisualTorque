"""Question templates with explicit option semantics and answer keys."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import itertools
from typing import Any

from .world import Outcome, SCHEMA_VERSION, Scene


class QuestionKind(str, Enum):
    OUTCOME = "outcome"
    LEFT_TORQUE = "left_torque"
    RIGHT_TORQUE = "right_torque"
    NET_TORQUE = "net_torque"
    TOTAL_UNITS = "total_units"
    LEFT_UNITS = "left_units"
    RIGHT_UNITS = "right_units"
    LEFT_DISTANCE = "left_distance"
    RIGHT_DISTANCE = "right_distance"


@dataclass(frozen=True, slots=True)
class QuestionRecord:
    question_id: str
    scene_id: str
    kind: QuestionKind
    prompt: str
    options: tuple[str, ...]
    option_semantics: tuple[str | int, ...]
    correct_index: int
    variant: str = "default"

    def __post_init__(self) -> None:
        if len(self.options) < 2 or len(self.options) != len(self.option_semantics):
            raise ValueError("options and option_semantics must have equal length >= 2")
        if len(set(self.options)) != len(self.options):
            raise ValueError("question options must be unique")
        if not 0 <= self.correct_index < len(self.options):
            raise ValueError("correct_index is out of range")

    @property
    def correct_label(self) -> str:
        return chr(ord("A") + self.correct_index)

    @property
    def semantic_answer(self) -> str | int:
        return self.option_semantics[self.correct_index]

    def to_record(self, split: str | None = None) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "record_type": "question",
            "question_id": self.question_id,
            "scene_id": self.scene_id,
            "question_kind": self.kind.value,
            "variant": self.variant,
            "prompt": self.prompt,
            "options": [
                {"label": chr(ord("A") + index), "text": text, "semantic": semantic}
                for index, (text, semantic) in enumerate(zip(self.options, self.option_semantics))
            ],
            "correct_label": self.correct_label,
            "semantic_answer": self.semantic_answer,
        }
        if split is not None:
            record["split"] = split
        return record


_OUTCOME_TEXT = {
    Outcome.LEFT.value: "The left side moves downward",
    Outcome.RIGHT.value: "The right side moves downward",
    Outcome.BALANCED.value: "The beam remains level",
}


def _numeric_question(
    scene: Scene,
    kind: QuestionKind,
    prompt: str,
    answer: int,
) -> QuestionRecord:
    candidates = [answer, answer - 1, answer + 1]
    if answer == 0:
        candidates = [0, -1, 1]
    digest = hashlib.sha256(f"{scene.scene_id}:{kind.value}".encode("utf-8")).digest()
    shift = digest[0] % len(candidates)
    semantics = tuple(candidates[shift:] + candidates[:shift])
    options = tuple(str(value) for value in semantics)
    return QuestionRecord(
        question_id=f"q_{scene.scene_id}_{kind.value}",
        scene_id=scene.scene_id,
        kind=kind,
        prompt=prompt,
        options=options,
        option_semantics=semantics,
        correct_index=semantics.index(answer),
    )


def generate_questions(scene: Scene) -> tuple[QuestionRecord, ...]:
    """Generate outcome variants plus exact intermediate-variable targets."""

    outcome_questions = []
    semantic_order = (Outcome.LEFT.value, Outcome.RIGHT.value, Outcome.BALANCED.value)
    for variant_index, permutation in enumerate(itertools.permutations(semantic_order)):
        options = tuple(_OUTCOME_TEXT[value] for value in permutation)
        outcome_questions.append(
            QuestionRecord(
                question_id=f"q_{scene.scene_id}_outcome_p{variant_index}",
                scene_id=scene.scene_id,
                kind=QuestionKind.OUTCOME,
                prompt="Based only on the pictured blocks and their distances from the pivot, what happens?",
                options=options,
                option_semantics=permutation,
                correct_index=permutation.index(scene.outcome.value),
                variant=f"permutation_{variant_index}",
            )
        )

    numeric_questions = [
        _numeric_question(scene, QuestionKind.LEFT_TORQUE, "What is the total left-side torque (block-distance units)?", scene.left_torque),
        _numeric_question(scene, QuestionKind.RIGHT_TORQUE, "What is the total right-side torque (block-distance units)?", scene.right_torque),
        _numeric_question(scene, QuestionKind.NET_TORQUE, "What is left torque minus right torque?", scene.net_torque),
        _numeric_question(scene, QuestionKind.TOTAL_UNITS, "How many blocks are shown in total?", scene.left_units + scene.right_units),
        _numeric_question(scene, QuestionKind.LEFT_UNITS, "How many blocks are on the left side?", scene.left_units),
        _numeric_question(scene, QuestionKind.RIGHT_UNITS, "How many blocks are on the right side?", scene.right_units),
    ]
    left_stacks = [stack for stack in scene.stacks if stack.side.value == "left"]
    right_stacks = [stack for stack in scene.stacks if stack.side.value == "right"]
    if len(left_stacks) == 1 and len(right_stacks) == 1:
        numeric_questions.extend(
            (
                _numeric_question(
                    scene,
                    QuestionKind.LEFT_DISTANCE,
                    "How many tick intervals from the pivot is the left stack?",
                    left_stacks[0].distance,
                ),
                _numeric_question(
                    scene,
                    QuestionKind.RIGHT_DISTANCE,
                    "How many tick intervals from the pivot is the right stack?",
                    right_stacks[0].distance,
                ),
            )
        )
    return tuple(outcome_questions) + tuple(numeric_questions)


def generate_all_questions(scenes: tuple[Scene, ...]) -> tuple[QuestionRecord, ...]:
    return tuple(question for scene in scenes for question in generate_questions(scene))
