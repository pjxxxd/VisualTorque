"""Build, summarize, and independently validate a VisualTorque dataset."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from .edits import validate_pair
from .generate import GenerationConfig, generate_counterfactual_pairs, generate_scenes
from .prompts import QuestionKind, QuestionRecord, generate_all_questions
from .render import RenderStyle, render_scene
from .splits import SplitConfig, SplitManifest, build_splits, validate_splits
from .world import CounterfactualPair, ProblemType, SCHEMA_VERSION, Scene, Side, Stack


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    output_dir: Path
    scene_count: int
    pair_count: int
    question_count: int
    image_count: int
    statistics: dict[str, Any]


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON in {path.name}:{line_number}") from error
    return records


def _with_progress(
    values: Iterable[Any],
    *,
    enabled: bool,
    total: int,
    description: str,
    unit: str,
) -> Iterable[Any]:
    """Add an optional notebook/terminal progress bar without a hard dependency."""

    if not enabled:
        return values
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return values
    return tqdm(
        values,
        total=total,
        desc=description,
        unit=unit,
        dynamic_ncols=True,
    )


def compute_statistics(
    scenes: tuple[Scene, ...],
    pairs: tuple[CounterfactualPair, ...],
    questions: tuple[QuestionRecord, ...],
    splits: SplitManifest,
) -> dict[str, Any]:
    def counts(values: Iterable[str]) -> dict[str, int]:
        return dict(sorted(Counter(values).items()))

    return {
        "schema_version": SCHEMA_VERSION,
        "counts": {
            "scenes": len(scenes),
            "counterfactual_pairs": len(pairs),
            "questions": len(questions),
        },
        "scene_outcomes": counts(scene.outcome.value for scene in scenes),
        "problem_types": counts(scene.problem_type.value for scene in scenes),
        "edit_types": counts(pair.edit_type for pair in pairs),
        "question_kinds": counts(question.kind.value for question in questions),
        "probe_splits": counts(splits.scene_split.values()),
        "intervention_splits": counts(splits.pair_split.values()),
        "question_splits": counts(splits.question_split.values()),
        "net_torque": {
            "minimum": min(scene.net_torque for scene in scenes),
            "maximum": max(scene.net_torque for scene in scenes),
            "balanced_count": sum(scene.net_torque == 0 for scene in scenes),
        },
    }


def build_dataset(
    output_dir: str | Path,
    generation: GenerationConfig = GenerationConfig(),
    split_config: SplitConfig = SplitConfig(),
    *,
    render_images: bool = True,
    show_progress: bool = False,
) -> DatasetSummary:
    """Build Dataset v1 in a new or empty directory, then validate it."""

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    scenes = generate_scenes(generation)
    pairs = generate_counterfactual_pairs(scenes)
    questions = generate_all_questions(scenes)
    splits = build_splits(scenes, pairs, questions, generation, split_config)
    statistics = compute_statistics(scenes, pairs, questions, splits)

    style = RenderStyle(ticks_per_side=generation.max_distance)
    image_dir = destination / "images"
    if render_images:
        image_dir.mkdir(parents=True, exist_ok=True)

    scene_records = []
    scene_iterator = _with_progress(
        scenes,
        enabled=show_progress,
        total=len(scenes),
        description="Rendering dataset images",
        unit="image",
    )
    for scene in scene_iterator:
        record = scene.to_record()
        record["split"] = splits.scene_split[scene.scene_id]
        if render_images:
            relative_image = Path("images") / f"{scene.scene_id}.png"
            render_scene(scene, destination / relative_image, style)
            record["image_path"] = relative_image.as_posix()
        scene_records.append(record)

    pair_records = []
    for pair in pairs:
        record = pair.to_record()
        record["split"] = splits.pair_split[pair.pair_id]
        pair_records.append(record)

    question_records = [
        question.to_record(splits.question_split[question.question_id])
        for question in questions
    ]

    _write_json(
        destination / "dataset_config.json",
        {
            "schema_version": SCHEMA_VERSION,
            "generation": generation.to_record(),
            "splits": asdict(split_config),
            "render": asdict(style) if render_images else None,
        },
    )
    _write_jsonl(destination / "scenes.jsonl", scene_records)
    _write_jsonl(destination / "counterfactual_pairs.jsonl", pair_records)
    _write_jsonl(destination / "questions.jsonl", question_records)
    _write_json(destination / "splits.json", splits.to_record())
    _write_json(destination / "statistics.json", statistics)

    validation = validate_dataset(
        destination,
        verify_images=render_images,
        show_progress=show_progress,
    )
    return DatasetSummary(
        destination,
        len(scenes),
        len(pairs),
        len(questions),
        validation["image_count"],
        statistics,
    )


def _load_scene(record: dict[str, Any]) -> Scene:
    stacks = tuple(
        Stack(
            stack_id=item["stack_id"],
            side=Side(item["side"]),
            distance=item["distance"],
            units=item["units"],
        )
        for item in record["stacks"]
    )
    metadata = tuple(sorted(record.get("metadata", {}).items()))
    scene = Scene(record["scene_id"], stacks, ProblemType(record["problem_type"]), metadata)
    if record.get("derived") != scene.to_record()["derived"]:
        raise ValueError(f"stored derived values are wrong for scene {scene.scene_id}")
    return scene


def _load_question(record: dict[str, Any]) -> QuestionRecord:
    options = tuple(option["text"] for option in record["options"])
    semantics = tuple(option["semantic"] for option in record["options"])
    label = record["correct_label"]
    correct_index = ord(label) - ord("A")
    question = QuestionRecord(
        question_id=record["question_id"],
        scene_id=record["scene_id"],
        kind=QuestionKind(record["question_kind"]),
        prompt=record["prompt"],
        options=options,
        option_semantics=semantics,
        correct_index=correct_index,
        variant=record.get("variant", "default"),
    )
    if question.semantic_answer != record["semantic_answer"]:
        raise ValueError(f"wrong semantic answer for question {question.question_id}")
    return question


def _manifest_from_records(
    split_record: dict[str, Any],
) -> SplitManifest:
    scene_split = {
        scene_id: split
        for split, scene_ids in split_record["probe_scenes"].items()
        for scene_id in scene_ids
    }
    pair_split = {
        pair_id: split
        for split, pair_ids in split_record["intervention_pairs"].items()
        for pair_id in pair_ids
    }
    question_split = {
        question_id: split
        for split, question_ids in split_record["questions"].items()
        for question_id in question_ids
    }
    heldout = tuple(tuple(item) for item in split_record["heldout_unit_distance_combinations"])
    return SplitManifest(scene_split, pair_split, question_split, heldout)


def validate_dataset(
    dataset_dir: str | Path,
    *,
    verify_images: bool = True,
    show_progress: bool = False,
) -> dict[str, int]:
    """Reload files from disk and verify physics, edits, answers, splits, and images."""

    root = Path(dataset_dir).expanduser().resolve()
    required = (
        "dataset_config.json",
        "scenes.jsonl",
        "counterfactual_pairs.jsonl",
        "questions.jsonl",
        "splits.json",
        "statistics.json",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"dataset is missing required files: {missing}")

    scene_records = _read_jsonl(root / "scenes.jsonl")
    scenes = tuple(_load_scene(record) for record in scene_records)
    if len({scene.scene_id for scene in scenes}) != len(scenes):
        raise ValueError("duplicate scene_id in scenes.jsonl")
    by_scene = {scene.scene_id: scene for scene in scenes}

    pair_records = _read_jsonl(root / "counterfactual_pairs.jsonl")
    pairs = []
    for record in pair_records:
        pair = CounterfactualPair(
            pair_id=record["pair_id"],
            target=by_scene[record["target_scene_id"]],
            source=by_scene[record["source_scene_id"]],
            edit_type=record["edit_type"],
            changed_stack_ids=tuple(record["changed_stack_ids"]),
            metadata=tuple(sorted(record.get("metadata", {}).items())),
        )
        validate_pair(pair)
        if pair.delta_net_torque != record["delta_net_torque"]:
            raise ValueError(f"wrong torque delta for pair {pair.pair_id}")
        pairs.append(pair)
    pair_tuple = tuple(pairs)
    if len({pair.pair_id for pair in pair_tuple}) != len(pair_tuple):
        raise ValueError("duplicate pair_id in counterfactual_pairs.jsonl")

    question_records = _read_jsonl(root / "questions.jsonl")
    questions = tuple(_load_question(record) for record in question_records)
    if len({question.question_id for question in questions}) != len(questions):
        raise ValueError("duplicate question_id in questions.jsonl")
    if any(question.scene_id not in by_scene for question in questions):
        raise ValueError("question references a missing scene")

    manifest = _manifest_from_records(_read_json(root / "splits.json"))
    validate_splits(manifest, scenes, pair_tuple, questions)

    # Split annotations in JSONL are redundant on purpose: disagreements expose
    # accidental file mixing immediately.
    for record in scene_records:
        if record.get("split") != manifest.scene_split[record["scene_id"]]:
            raise ValueError(f"scene split annotation mismatch: {record['scene_id']}")
    for record in pair_records:
        if record.get("split") != manifest.pair_split[record["pair_id"]]:
            raise ValueError(f"pair split annotation mismatch: {record['pair_id']}")
    for record in question_records:
        if record.get("split") != manifest.question_split[record["question_id"]]:
            raise ValueError(f"question split annotation mismatch: {record['question_id']}")

    image_count = 0
    if verify_images:
        image_records = _with_progress(
            scene_records,
            enabled=show_progress,
            total=len(scene_records),
            description="Verifying dataset images",
            unit="image",
        )
        for record in image_records:
            relative = record.get("image_path")
            if not relative:
                raise ValueError(f"scene has no image_path: {record['scene_id']}")
            path = (root / relative).resolve()
            if root not in path.parents:
                raise ValueError("image_path escapes the dataset directory")
            with Image.open(path) as image:
                image.verify()
            image_count += 1

    stored_statistics = _read_json(root / "statistics.json")
    computed_statistics = compute_statistics(scenes, pair_tuple, questions, manifest)
    if stored_statistics != computed_statistics:
        raise ValueError("statistics.json does not match dataset contents")

    return {
        "scene_count": len(scenes),
        "pair_count": len(pair_tuple),
        "question_count": len(questions),
        "image_count": image_count,
    }
