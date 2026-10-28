"""Behavioral reports for competence, robustness, shortcuts, and edits."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Iterable, Sequence


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return fmean(materialized) if materialized else None


def _summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    generated = [record for record in records if record["generated_response"] is not None]
    parsed = [record for record in generated if record["generation_parse_success"]]
    return {
        "count": len(records),
        "likelihood_accuracy": _mean(
            float(record["likelihood_correct"]) for record in records
        ),
        "mean_correct_margin": _mean(
            float(record["correct_vs_best_alternative_margin"]) for record in records
        ),
        "generation_audit_count": len(generated),
        "generation_parse_rate": (
            len(parsed) / len(generated) if generated else None
        ),
        "generation_accuracy_among_parsed": _mean(
            float(record["generation_correct"]) for record in parsed
        ),
        "mean_latency_seconds": _mean(
            float(record["latency_seconds"]) for record in records
        ),
    }


def _group_summary(
    records: Sequence[dict[str, Any]], key: Callable[[dict[str, Any]], str]
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(key(record), []).append(record)
    return {name: _summary(group) for name, group in sorted(groups.items())}


def _ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        average_rank = (start + end - 1) / 2.0
        for position in range(start, end):
            ranks[indexed[position][0]] = average_rank
        start = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean, right_mean = fmean(left), fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_ss = sum((x - left_mean) ** 2 for x in left)
    right_ss = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_ss * right_ss)
    return numerator / denominator if denominator else None


def _torque_margin_report(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    outcome_records = [
        record
        for record in records
        if record["question_kind"] == "outcome"
        and record["signed_left_right_margin"] is not None
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in outcome_records:
        grouped.setdefault((record["condition"], record["scene_id"]), []).append(record)

    by_condition: dict[str, list[tuple[float, float]]] = {}
    for (condition, _), group in grouped.items():
        torque = float(group[0]["net_torque"])
        margin = fmean(float(record["signed_left_right_margin"]) for record in group)
        by_condition.setdefault(condition, []).append((torque, margin))

    report = {}
    for condition, pairs in sorted(by_condition.items()):
        torques = [pair[0] for pair in pairs]
        margins = [pair[1] for pair in pairs]
        report[condition] = {
            "scene_count": len(pairs),
            "pearson": _pearson(torques, margins),
            "spearman": _pearson(_ranks(torques), _ranks(margins)),
        }
    return report


def _option_robustness(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    outcome_records = [record for record in records if record["question_kind"] == "outcome"]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in outcome_records:
        groups.setdefault((record["condition"], record["scene_id"]), []).append(record)

    report: dict[str, Any] = {}
    for condition in sorted({key[0] for key in groups}):
        condition_groups = [group for (name, _), group in groups.items() if name == condition]
        complete = [group for group in condition_groups if len(group) == 6]
        label_counts = Counter(
            record["predicted_label"] for group in condition_groups for record in group
        )
        gold_label_groups: dict[str, list[dict[str, Any]]] = {}
        for group in condition_groups:
            for record in group:
                gold_label_groups.setdefault(record["gold_label"], []).append(record)
        report[condition] = {
            "scene_groups": len(condition_groups),
            "complete_six_permutation_groups": len(complete),
            "semantic_consistency_rate": _mean(
                float(len({record["predicted_semantic"] for record in group}) == 1)
                for group in complete
            ),
            "all_six_correct_rate": _mean(
                float(all(record["likelihood_correct"] for record in group))
                for group in complete
            ),
            "mean_within_scene_accuracy": _mean(
                fmean(float(record["likelihood_correct"]) for record in group)
                for group in complete
            ),
            "predicted_label_counts": dict(sorted(label_counts.items())),
            "accuracy_by_gold_label": {
                label: _summary(group)
                for label, group in sorted(gold_label_groups.items())
            },
        }
    return report


def _image_vs_symbolic(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_question: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        by_question.setdefault(record["question_id"], {})[record["condition"]] = record
    pairs = [
        (conditions["image_rule"], conditions["symbolic_rule"])
        for conditions in by_question.values()
        if "image_rule" in conditions and "symbolic_rule" in conditions
    ]

    def report(group: Sequence[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
        quadrants = Counter()
        for image, symbolic in group:
            key = (
                "image_correct" if image["likelihood_correct"] else "image_wrong",
                "symbolic_correct" if symbolic["likelihood_correct"] else "symbolic_wrong",
            )
            quadrants["__".join(key)] += 1
        return {
            "paired_count": len(group),
            "image_accuracy": _mean(float(image["likelihood_correct"]) for image, _ in group),
            "symbolic_accuracy": _mean(float(symbolic["likelihood_correct"]) for _, symbolic in group),
            "semantic_agreement_rate": _mean(
                float(image["predicted_semantic"] == symbolic["predicted_semantic"])
                for image, symbolic in group
            ),
            "quadrants": dict(sorted(quadrants.items())),
        }

    by_problem_type = {}
    for problem_type in sorted({image["problem_type"] for image, _ in pairs}):
        by_problem_type[problem_type] = report(
            [pair for pair in pairs if pair[0]["problem_type"] == problem_type]
        )
    return {"overall": report(pairs), "by_problem_type": by_problem_type}


def _compare(left: int, right: int) -> str:
    if left > right:
        return "left"
    if left < right:
        return "right"
    return "balanced"


def _heuristics(scene: dict[str, Any]) -> dict[str, str]:
    left = [stack for stack in scene["stacks"] if stack["side"] == "left"]
    right = [stack for stack in scene["stacks"] if stack["side"] == "right"]
    return {
        "total_weight": _compare(
            sum(stack["units"] for stack in left),
            sum(stack["units"] for stack in right),
        ),
        "farthest_stack": _compare(
            max((stack["distance"] for stack in left), default=0),
            max((stack["distance"] for stack in right), default=0),
        ),
        "stack_count": _compare(len(left), len(right)),
        "maximum_single_torque": _compare(
            max((stack["torque"] for stack in left), default=0),
            max((stack["torque"] for stack in right), default=0),
        ),
    }


def _heuristic_report(
    records: Sequence[dict[str, Any]], scenes: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    scene_by_id = {scene["scene_id"]: scene for scene in scenes}
    outcome_records = [record for record in records if record["question_kind"] == "outcome"]
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for record in outcome_records:
        by_condition.setdefault(record["condition"], []).append(record)

    report = {}
    for condition, group in sorted(by_condition.items()):
        # Use one option permutation per scene for model-agreement estimates.
        one_per_scene: dict[str, dict[str, Any]] = {}
        for record in sorted(group, key=lambda item: item["variant"]):
            one_per_scene.setdefault(record["scene_id"], record)
        condition_result = {}
        for heuristic_name in (
            "total_weight",
            "farthest_stack",
            "stack_count",
            "maximum_single_torque",
        ):
            pairs = [
                (
                    record,
                    _heuristics(scene_by_id[record["scene_id"]])[heuristic_name],
                )
                for record in one_per_scene.values()
            ]
            condition_result[heuristic_name] = {
                "scene_count": len(pairs),
                "heuristic_accuracy": _mean(
                    float(heuristic == record["gold_semantic"])
                    for record, heuristic in pairs
                ),
                "model_agreement": _mean(
                    float(heuristic == record["predicted_semantic"])
                    for record, heuristic in pairs
                ),
            }
        report[condition] = condition_result
    return report


def _counterfactual_report(
    records: Sequence[dict[str, Any]], pairs: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    outcome_records = [record for record in records if record["question_kind"] == "outcome"]
    index = {
        (record["condition"], record["scene_id"], record["variant"]): record
        for record in outcome_records
    }
    observations = []
    conditions = sorted({record["condition"] for record in outcome_records})
    variants = sorted({record["variant"] for record in outcome_records})
    order = {"right": -1, "balanced": 0, "left": 1}
    for pair in pairs:
        for condition in conditions:
            for variant in variants:
                target = index.get((condition, pair["target_scene_id"], variant))
                source = index.get((condition, pair["source_scene_id"], variant))
                if target is None or source is None:
                    continue
                true_direction = order[source["gold_semantic"]] - order[target["gold_semantic"]]
                predicted_direction = order[source["predicted_semantic"]] - order[target["predicted_semantic"]]
                delta = int(pair["delta_net_torque"])
                margin_delta = (
                    float(source["signed_left_right_margin"])
                    - float(target["signed_left_right_margin"])
                )
                observations.append(
                    {
                        "edit_type": pair["edit_type"],
                        "condition": condition,
                        "strict_both_correct": bool(
                            target["likelihood_correct"] and source["likelihood_correct"]
                        ),
                        "outcome_change_direction_correct": (
                            (predicted_direction > 0) - (predicted_direction < 0)
                            == (true_direction > 0) - (true_direction < 0)
                        ),
                        "margin_direction_correct": (
                            None
                            if delta == 0
                            else ((margin_delta > 0) - (margin_delta < 0))
                            == ((delta > 0) - (delta < 0))
                        ),
                        "absolute_margin_change": abs(margin_delta),
                        "delta_net_torque": delta,
                    }
                )

    def report(group: Sequence[dict[str, Any]]) -> dict[str, Any]:
        nonzero = [item for item in group if item["margin_direction_correct"] is not None]
        zero = [item for item in group if item["delta_net_torque"] == 0]
        return {
            "observation_count": len(group),
            "strict_both_correct_rate": _mean(
                float(item["strict_both_correct"]) for item in group
            ),
            "outcome_change_direction_accuracy": _mean(
                float(item["outcome_change_direction_correct"]) for item in group
            ),
            "signed_margin_direction_accuracy_nonzero_delta": _mean(
                float(item["margin_direction_correct"]) for item in nonzero
            ),
            "zero_delta_observation_count": len(zero),
            "mean_absolute_margin_change_zero_delta": _mean(
                item["absolute_margin_change"] for item in zero
            ),
        }

    by_edit = {
        edit_type: report([item for item in observations if item["edit_type"] == edit_type])
        for edit_type in sorted({item["edit_type"] for item in observations})
    }
    by_condition = {
        condition: report([item for item in observations if item["condition"] == condition])
        for condition in sorted({item["condition"] for item in observations})
    }
    return {"overall": report(observations), "by_edit_type": by_edit, "by_condition": by_condition}


def analyze_behavior_run(
    dataset_dir: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    """Create all paper-facing JSON reports from append-only predictions."""

    dataset_root = Path(dataset_dir).expanduser().resolve()
    prediction_file = Path(predictions_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    records = _read_jsonl(prediction_file)
    if not records:
        raise ValueError("cannot analyze an empty predictions file")
    scenes = _read_jsonl(dataset_root / "scenes.jsonl")
    pairs = _read_jsonl(dataset_root / "counterfactual_pairs.jsonl")

    metrics = {
        "overall": _summary(records),
        "by_condition": _group_summary(records, lambda record: record["condition"]),
        "by_question_kind": _group_summary(records, lambda record: record["question_kind"]),
        "by_split": _group_summary(records, lambda record: record["split"]),
        "by_condition_and_question": _group_summary(
            records,
            lambda record: f"{record['condition']}::{record['question_kind']}",
        ),
        "torque_margin_correlation": _torque_margin_report(records),
    }
    by_problem_type = _group_summary(records, lambda record: record["problem_type"])
    option_robustness = _option_robustness(records)
    image_symbolic = _image_vs_symbolic(records)
    counterfactual = _counterfactual_report(records, pairs)
    heuristics = _heuristic_report(records, scenes)

    reports = {
        "metrics": destination / "metrics.json",
        "metrics_by_problem_type": destination / "metrics_by_problem_type.json",
        "option_robustness": destination / "option_robustness.json",
        "image_vs_symbolic": destination / "image_vs_symbolic.json",
        "counterfactual_consistency": destination / "counterfactual_consistency.json",
        "heuristic_agreement": destination / "heuristic_agreement.json",
    }
    values = {
        "metrics": metrics,
        "metrics_by_problem_type": by_problem_type,
        "option_robustness": option_robustness,
        "image_vs_symbolic": image_symbolic,
        "counterfactual_consistency": counterfactual,
        "heuristic_agreement": heuristics,
    }
    for name, path in reports.items():
        _write_json(path, values[name])
    return {name: str(path) for name, path in reports.items()}
