"""Load Dataset v1 records and construct controlled model prompts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .schema import BehaviorRunConfig, PromptCondition


_RULE = (
    "Each square is one unit of weight. Distance is measured in tick intervals "
    "from the central pivot. A stack's turning effect is its number of squares "
    "multiplied by its distance. Add the turning effects on each side and compare "
    "the two side totals."
)


@dataclass(frozen=True, slots=True)
class BehaviorExample:
    example_id: str
    question_id: str
    scene_id: str
    condition: PromptCondition
    question_kind: str
    variant: str
    split: str
    problem_type: str
    outcome: str
    net_torque: int
    prompt: str
    image_path: Path | None
    labels: tuple[str, ...]
    option_text: tuple[str, ...]
    option_semantics: tuple[str | int, ...]
    gold_label: str
    gold_semantic: str | int

    def __post_init__(self) -> None:
        if not self.labels or len(self.labels) != len(self.option_semantics):
            raise ValueError("labels and option semantics must align")
        if self.gold_label not in self.labels:
            raise ValueError("gold label is not one of the legal labels")
        if self.condition.uses_image != (self.image_path is not None):
            raise ValueError("image presence disagrees with prompt condition")

    @property
    def label_to_semantic(self) -> dict[str, str | int]:
        return dict(zip(self.labels, self.option_semantics))

    def messages(self) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        if self.image_path is not None:
            content.append({"type": "image", "path": str(self.image_path)})
        content.append({"type": "text", "text": self.prompt})
        return [{"role": "user", "content": content}]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _symbolic_scene(scene: dict[str, Any]) -> str:
    descriptions = []
    for side in ("left", "right"):
        stacks = sorted(
            (stack for stack in scene["stacks"] if stack["side"] == side),
            key=lambda stack: stack["distance"],
        )
        if not stacks:
            descriptions.append(f"{side.capitalize()} side: no blocks")
            continue
        text = "; ".join(
            f"{stack['units']} block(s) at distance {stack['distance']}"
            for stack in stacks
        )
        descriptions.append(f"{side.capitalize()} side: {text}")
    return ". ".join(descriptions) + "."


def _make_prompt(
    scene: dict[str, Any],
    question: dict[str, Any],
    condition: PromptCondition,
) -> str:
    sections = [
        "Solve this balance-scale question using only the supplied scene.",
    ]
    if condition.supplies_rule:
        sections.append(_RULE)
    if condition is PromptCondition.SYMBOLIC_RULE:
        sections.append("Exact symbolic scene description: " + _symbolic_scene(scene))
    question_text = question["prompt"]
    if question["question_kind"] == "outcome":
        question_text = "What happens to the balance scale?"
    sections.append("Question: " + question_text)
    option_lines = [
        f"{option['label']}. {option['text']}" for option in question["options"]
    ]
    sections.append("Options:\n" + "\n".join(option_lines))
    sections.append(
        "Do not explain your reasoning. Reply with exactly one option letter and "
        "nothing else. Valid replies are: "
        + ", ".join(option["label"] for option in question["options"])
        + "."
    )
    return "\n\n".join(sections)


def _stable_order(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _select_scene_ids(
    scenes: list[dict[str, Any]], config: BehaviorRunConfig
) -> set[str]:
    eligible = [scene for scene in scenes if scene["split"] in config.splits]
    if config.scenes_per_stratum is not None:
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for scene in eligible:
            key = (scene["problem_type"], scene["derived"]["outcome"], scene["split"])
            groups.setdefault(key, []).append(scene)
        selected = []
        for group in groups.values():
            group.sort(key=lambda scene: _stable_order(scene["scene_id"], config.seed))
            selected.extend(group[: config.scenes_per_stratum])
        eligible = selected
    if config.max_scenes is not None:
        eligible.sort(key=lambda scene: _stable_order(scene["scene_id"], config.seed))
        eligible = eligible[: config.max_scenes]
    return {scene["scene_id"] for scene in eligible}


def load_behavior_examples(config: BehaviorRunConfig) -> tuple[BehaviorExample, ...]:
    root = config.dataset_dir.expanduser().resolve()
    scenes = _read_jsonl(root / "scenes.jsonl")
    questions = _read_jsonl(root / "questions.jsonl")
    selected_scene_ids = _select_scene_ids(scenes, config)
    scene_by_id = {scene["scene_id"]: scene for scene in scenes}

    examples = []
    for condition in config.conditions:
        for question in questions:
            if question["scene_id"] not in selected_scene_ids:
                continue
            if question["question_kind"] not in config.question_kinds:
                continue
            scene = scene_by_id[question["scene_id"]]
            image_path = None
            if condition.uses_image:
                relative = scene.get("image_path")
                if not relative:
                    raise ValueError(f"scene {scene['scene_id']} has no image_path")
                image_path = (root / relative).resolve()
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
            options = question["options"]
            example_id = f"{condition.value}:{question['question_id']}"
            examples.append(
                BehaviorExample(
                    example_id=example_id,
                    question_id=question["question_id"],
                    scene_id=scene["scene_id"],
                    condition=condition,
                    question_kind=question["question_kind"],
                    variant=question["variant"],
                    split=scene["split"],
                    problem_type=scene["problem_type"],
                    outcome=scene["derived"]["outcome"],
                    net_torque=scene["derived"]["net_torque"],
                    prompt=_make_prompt(scene, question, condition),
                    image_path=image_path,
                    labels=tuple(option["label"] for option in options),
                    option_text=tuple(option["text"] for option in options),
                    option_semantics=tuple(option["semantic"] for option in options),
                    gold_label=question["correct_label"],
                    gold_semantic=question["semantic_answer"],
                )
            )

    examples.sort(key=lambda example: (example.condition.value, example.question_id))
    if config.max_examples is not None:
        examples = examples[: config.max_examples]
    if not examples:
        raise ValueError("the run configuration selected no behavior examples")
    return tuple(examples)
