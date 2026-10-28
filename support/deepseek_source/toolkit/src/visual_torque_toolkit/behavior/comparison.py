"""Cross-model, scene-level analysis for VisualTorque behavior runs.

The primary behavior run deliberately scores all six mappings between the
semantic outcomes (left, right, balanced) and the neutral labels (A, B, C).
This module uses that design to cancel stable answer-token priors: scores are
mapped back to semantics and averaged across all six permutations before a
single scene-level prediction is made.
"""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import random
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence


SEMANTIC_OUTCOMES = ("balanced", "left", "right")


@dataclass(frozen=True, slots=True)
class BehaviorModelArtifacts:
    """Prediction files produced for one evaluated model."""

    name: str
    primary_predictions: Path
    control_predictions: Path | None = None
    full_symbolic_predictions: Path | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("model artifact name must not be empty")


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return fmean(materialized) if materialized else None


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a percentile of no values")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def _bootstrap_mean_interval(
    values: Sequence[float],
    *,
    replicates: int,
    rng: random.Random,
) -> list[float] | None:
    if not values:
        return None
    if replicates <= 0:
        return None
    count = len(values)
    draws = sorted(
        fmean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(replicates)
    )
    return [_percentile(draws, 0.025), _percentile(draws, 0.975)]


def _bootstrap_macro_interval(
    groups: Mapping[str, Sequence[float]],
    *,
    replicates: int,
    rng: random.Random,
) -> list[float] | None:
    nonempty = [tuple(values) for values in groups.values() if values]
    if not nonempty or replicates <= 0:
        return None
    draws = []
    for _ in range(replicates):
        type_means = []
        for values in nonempty:
            count = len(values)
            type_means.append(
                fmean(values[rng.randrange(count)] for _ in range(count))
            )
        draws.append(fmean(type_means))
    draws.sort()
    return [_percentile(draws, 0.025), _percentile(draws, 0.975)]


def _ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        rank = (start + end - 1) / 2.0
        for position in range(start, end):
            result[indexed[position][0]] = rank
        start = end
    return result


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean, right_mean = fmean(left), fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_ss = sum((x - left_mean) ** 2 for x in left)
    right_ss = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_ss * right_ss)
    return numerator / denominator if denominator else None


def _one_value(records: Sequence[Mapping[str, Any]], key: str) -> Any:
    values = {json.dumps(record.get(key), sort_keys=True) for record in records}
    if len(values) != 1:
        raise ValueError(f"permutation group disagrees on {key}: {sorted(values)}")
    return records[0].get(key)


def aggregate_permuted_outcomes(
    predictions: str | Path | Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate label likelihoods into one token-prior-balanced scene score."""

    rows = (
        _read_jsonl(predictions)
        if isinstance(predictions, (str, Path))
        else [dict(record) for record in predictions]
    )
    rows = [record for record in rows if record.get("question_kind") == "outcome"]
    if not rows:
        raise ValueError("prediction input contains no outcome records")

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in rows:
        key = (str(record["condition"]), str(record["scene_id"]))
        groups.setdefault(key, []).append(record)

    aggregated = []
    for (condition, scene_id), group in sorted(groups.items()):
        variants = [str(record["variant"]) for record in group]
        if len(variants) != len(set(variants)):
            raise ValueError(f"duplicate outcome permutation for {condition}:{scene_id}")

        score_values: dict[str, list[float]] = {
            semantic: [] for semantic in SEMANTIC_OUTCOMES
        }
        assignment_counts: Counter[tuple[str, str]] = Counter()
        predicted_labels: Counter[str] = Counter()
        predicted_semantics = []
        raw_correct = []
        for record in group:
            mapping = record["label_to_semantic"]
            scores = record["candidate_logprobs"]
            if set(mapping) != set(scores):
                raise ValueError(
                    f"candidate labels and semantic mapping disagree for {record['example_id']}"
                )
            if set(mapping.values()) != set(SEMANTIC_OUTCOMES):
                raise ValueError(
                    f"outcome semantics are incomplete for {record['example_id']}"
                )
            for label, semantic in mapping.items():
                score_values[str(semantic)].append(float(scores[label]))
                assignment_counts[(str(semantic), str(label))] += 1
            predicted_labels[str(record["predicted_label"])] += 1
            predicted_semantics.append(str(record["predicted_semantic"]))
            raw_correct.append(float(bool(record["likelihood_correct"])))

        semantic_scores = {
            semantic: fmean(values) for semantic, values in score_values.items()
        }
        predicted = max(
            sorted(semantic_scores), key=lambda semantic: semantic_scores[semantic]
        )
        gold = str(_one_value(group, "gold_semantic"))
        alternatives = [
            score for semantic, score in semantic_scores.items() if semantic != gold
        ]
        complete_six = len(group) == 6 and all(
            assignment_counts[(semantic, label)] == 2
            for semantic in SEMANTIC_OUTCOMES
            for label in ("A", "B", "C")
        )
        aggregated.append(
            {
                "schema_version": "behavior-comparison-2.0",
                "record_type": "permutation_aggregated_prediction",
                "model_id": _one_value(group, "model_id"),
                "model_revision": _one_value(group, "model_revision"),
                "condition": condition,
                "scene_id": scene_id,
                "split": _one_value(group, "split"),
                "problem_type": _one_value(group, "problem_type"),
                "question_kind": "outcome",
                "gold_semantic": gold,
                "net_torque": int(_one_value(group, "net_torque")),
                "variant_count": len(group),
                "complete_six_permutations": complete_six,
                "raw_variant_accuracy": fmean(raw_correct),
                "raw_semantic_consistent": len(set(predicted_semantics)) == 1,
                "all_variants_correct": all(value == 1.0 for value in raw_correct),
                "raw_predicted_label_counts": dict(sorted(predicted_labels.items())),
                "semantic_scores": semantic_scores,
                "aggregated_predicted_semantic": predicted,
                "aggregated_correct": predicted == gold,
                "aggregated_gold_margin": semantic_scores[gold] - max(alternatives),
                "aggregated_signed_left_right_margin": (
                    semantic_scores["left"] - semantic_scores["right"]
                ),
                "scoring_modes": sorted(
                    {str(record["scoring_mode"]) for record in group}
                ),
            }
        )
    return aggregated


def _summarize_model(
    records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int,
    rng: random.Random,
) -> dict[str, Any]:
    correctness = [float(record["aggregated_correct"]) for record in records]
    by_problem_type: dict[str, list[Mapping[str, Any]]] = {}
    by_split: dict[str, list[Mapping[str, Any]]] = {}
    label_counts: Counter[str] = Counter()
    for record in records:
        by_problem_type.setdefault(str(record["problem_type"]), []).append(record)
        by_split.setdefault(str(record["split"]), []).append(record)
        label_counts.update(record["raw_predicted_label_counts"])

    problem_report = {}
    problem_correctness = {}
    for name, group in sorted(by_problem_type.items()):
        values = [float(record["aggregated_correct"]) for record in group]
        problem_correctness[name] = values
        problem_report[name] = {
            "scene_count": len(group),
            "aggregated_accuracy": fmean(values),
            "raw_variant_accuracy": fmean(
                float(record["raw_variant_accuracy"]) for record in group
            ),
        }

    torques = [float(record["net_torque"]) for record in records]
    margins = [
        float(record["aggregated_signed_left_right_margin"]) for record in records
    ]
    total_labels = sum(label_counts.values())
    return {
        "scene_count": len(records),
        "aggregated_accuracy": fmean(correctness),
        "aggregated_accuracy_scene_bootstrap_95_ci": _bootstrap_mean_interval(
            correctness, replicates=bootstrap_replicates, rng=rng
        ),
        "problem_type_macro_accuracy": fmean(
            report["aggregated_accuracy"] for report in problem_report.values()
        ),
        "problem_type_macro_accuracy_stratified_bootstrap_95_ci": (
            _bootstrap_macro_interval(
                problem_correctness,
                replicates=bootstrap_replicates,
                rng=rng,
            )
        ),
        "raw_variant_accuracy": fmean(
            float(record["raw_variant_accuracy"]) for record in records
        ),
        "all_six_correct_rate": fmean(
            float(record["all_variants_correct"]) for record in records
        ),
        "raw_semantic_consistency_rate": fmean(
            float(record["raw_semantic_consistent"]) for record in records
        ),
        "complete_six_permutation_rate": fmean(
            float(record["complete_six_permutations"]) for record in records
        ),
        "raw_predicted_label_counts": dict(sorted(label_counts.items())),
        "raw_predicted_label_shares": {
            label: count / total_labels for label, count in sorted(label_counts.items())
        },
        "torque_margin_correlation": {
            "pearson": _pearson(torques, margins),
            "spearman": _pearson(_ranks(torques), _ranks(margins)),
        },
        "by_problem_type": problem_report,
        "by_split": {
            name: {
                "scene_count": len(group),
                "aggregated_accuracy": fmean(
                    float(record["aggregated_correct"]) for record in group
                ),
            }
            for name, group in sorted(by_split.items())
        },
    }


def _paired_model_report(
    left_name: str,
    left_records: Sequence[Mapping[str, Any]],
    right_name: str,
    right_records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int,
    rng: random.Random,
) -> dict[str, Any]:
    def index(records: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
        return {
            (str(record["condition"]), str(record["scene_id"])): record
            for record in records
        }

    left_index, right_index = index(left_records), index(right_records)
    if set(left_index) != set(right_index):
        left_only = len(set(left_index) - set(right_index))
        right_only = len(set(right_index) - set(left_index))
        raise ValueError(
            "primary model runs do not cover the same scenes "
            f"({left_name}-only={left_only}, {right_name}-only={right_only})"
        )
    pairs = [(left_index[key], right_index[key]) for key in sorted(left_index)]
    deltas = [
        float(left["aggregated_correct"]) - float(right["aggregated_correct"])
        for left, right in pairs
    ]
    quadrants: Counter[str] = Counter()
    for left, right in pairs:
        left_status = "correct" if left["aggregated_correct"] else "wrong"
        right_status = "correct" if right["aggregated_correct"] else "wrong"
        quadrants[f"{left_name}_{left_status}__{right_name}_{right_status}"] += 1

    report = {
        "scene_count": len(pairs),
        "left_model": left_name,
        "right_model": right_name,
        "left_accuracy": fmean(float(left["aggregated_correct"]) for left, _ in pairs),
        "right_accuracy": fmean(float(right["aggregated_correct"]) for _, right in pairs),
        "left_minus_right_accuracy": fmean(deltas),
        "left_minus_right_paired_scene_bootstrap_95_ci": _bootstrap_mean_interval(
            deltas, replicates=bootstrap_replicates, rng=rng
        ),
        "semantic_agreement_rate": fmean(
            float(
                left["aggregated_predicted_semantic"]
                == right["aggregated_predicted_semantic"]
            )
            for left, right in pairs
        ),
        "quadrants": dict(sorted(quadrants.items())),
        "by_problem_type": {},
    }
    problem_types = sorted({str(left["problem_type"]) for left, _ in pairs})
    for problem_type in problem_types:
        group = [pair for pair in pairs if pair[0]["problem_type"] == problem_type]
        report["by_problem_type"][problem_type] = {
            "scene_count": len(group),
            "left_accuracy": fmean(
                float(left["aggregated_correct"]) for left, _ in group
            ),
            "right_accuracy": fmean(
                float(right["aggregated_correct"]) for _, right in group
            ),
        }
    return report


def _paired_condition_report(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    bootstrap_replicates: int,
    rng: random.Random,
    image_correct_key: str,
    symbolic_correct_key: str,
    image_semantic_key: str,
    symbolic_semantic_key: str,
) -> dict[str, Any]:
    if not pairs:
        return {
            "pair_count": 0,
            "scene_count": 0,
            "image_accuracy": None,
            "symbolic_accuracy": None,
            "symbolic_minus_image_accuracy": None,
            "symbolic_minus_image_scene_bootstrap_95_ci": None,
            "semantic_agreement_rate": None,
            "quadrants": {},
        }
    by_scene: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
    quadrants: Counter[str] = Counter()
    for image, symbolic in pairs:
        by_scene.setdefault(str(image["scene_id"]), []).append((image, symbolic))
        image_correct = bool(image[image_correct_key])
        symbolic_correct = bool(symbolic[symbolic_correct_key])
        quadrants[
            ("image_correct" if image_correct else "image_wrong")
            + "__"
            + ("symbolic_correct" if symbolic_correct else "symbolic_wrong")
        ] += 1
    scene_deltas = [
        fmean(
            float(symbolic[symbolic_correct_key]) - float(image[image_correct_key])
            for image, symbolic in group
        )
        for group in by_scene.values()
    ]
    return {
        "pair_count": len(pairs),
        "scene_count": len(by_scene),
        "image_accuracy": fmean(float(image[image_correct_key]) for image, _ in pairs),
        "symbolic_accuracy": fmean(
            float(symbolic[symbolic_correct_key]) for _, symbolic in pairs
        ),
        "symbolic_minus_image_accuracy": fmean(scene_deltas),
        "symbolic_minus_image_scene_bootstrap_95_ci": _bootstrap_mean_interval(
            scene_deltas, replicates=bootstrap_replicates, rng=rng
        ),
        "semantic_agreement_rate": fmean(
            float(image[image_semantic_key] == symbolic[symbolic_semantic_key])
            for image, symbolic in pairs
        ),
        "quadrants": dict(sorted(quadrants.items())),
    }


def _control_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int,
    rng: random.Random,
) -> dict[str, Any]:
    by_question: dict[str, dict[str, Mapping[str, Any]]] = {}
    for record in rows:
        condition = str(record.get("condition"))
        if condition in ("image_rule", "symbolic_rule"):
            by_question.setdefault(str(record["question_id"]), {})[condition] = record
    pairs = [
        (conditions["image_rule"], conditions["symbolic_rule"])
        for conditions in by_question.values()
        if "image_rule" in conditions and "symbolic_rule" in conditions
    ]

    def report(group: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> dict[str, Any]:
        return _paired_condition_report(
            group,
            bootstrap_replicates=bootstrap_replicates,
            rng=rng,
            image_correct_key="likelihood_correct",
            symbolic_correct_key="likelihood_correct",
            image_semantic_key="predicted_semantic",
            symbolic_semantic_key="predicted_semantic",
        )

    image_aggregated = aggregate_permuted_outcomes(
        [record for record in rows if record.get("condition") == "image_rule"]
    )
    symbolic_aggregated = aggregate_permuted_outcomes(
        [record for record in rows if record.get("condition") == "symbolic_rule"]
    )
    symbolic_by_scene = {record["scene_id"]: record for record in symbolic_aggregated}
    outcome_pairs = [
        (record, symbolic_by_scene[record["scene_id"]])
        for record in image_aggregated
        if record["scene_id"] in symbolic_by_scene
    ]
    outcome_aggregated_report = _paired_condition_report(
        outcome_pairs,
        bootstrap_replicates=bootstrap_replicates,
        rng=rng,
        image_correct_key="aggregated_correct",
        symbolic_correct_key="aggregated_correct",
        image_semantic_key="aggregated_predicted_semantic",
        symbolic_semantic_key="aggregated_predicted_semantic",
    )

    question_kinds = sorted({str(image["question_kind"]) for image, _ in pairs})
    problem_types = sorted({str(image["problem_type"]) for image, _ in pairs})
    return {
        "overall_raw_question_pairs": report(pairs),
        "outcome_permutation_aggregated_scene_pairs": outcome_aggregated_report,
        "by_question_kind": {
            name: report([pair for pair in pairs if pair[0]["question_kind"] == name])
            for name in question_kinds
        },
        "by_problem_type": {
            name: report([pair for pair in pairs if pair[0]["problem_type"] == name])
            for name in problem_types
        },
    }


def _full_symbolic_report(
    image_records: Sequence[Mapping[str, Any]],
    symbolic_records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int,
    rng: random.Random,
) -> dict[str, Any]:
    image_index = {str(record["scene_id"]): record for record in image_records}
    symbolic_index = {str(record["scene_id"]): record for record in symbolic_records}
    common = sorted(set(image_index) & set(symbolic_index))
    pairs = [(image_index[scene], symbolic_index[scene]) for scene in common]

    def report(group: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> dict[str, Any]:
        return _paired_condition_report(
            group,
            bootstrap_replicates=bootstrap_replicates,
            rng=rng,
            image_correct_key="aggregated_correct",
            symbolic_correct_key="aggregated_correct",
            image_semantic_key="aggregated_predicted_semantic",
            symbolic_semantic_key="aggregated_predicted_semantic",
        )

    return {
        "overall": report(pairs),
        "by_problem_type": {
            problem_type: report(
                [pair for pair in pairs if pair[0]["problem_type"] == problem_type]
            )
            for problem_type in sorted(
                {str(image["problem_type"]) for image, _ in pairs}
            )
        },
        "image_only_scene_count": len(set(image_index) - set(symbolic_index)),
        "symbolic_only_scene_count": len(set(symbolic_index) - set(image_index)),
    }


def _compare(left: int, right: int) -> str:
    if left > right:
        return "left"
    if left < right:
        return "right"
    return "balanced"


def _scene_heuristics(scene: Mapping[str, Any]) -> dict[str, str]:
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


def _stable_key(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _activation_manifest(
    dataset_dir: Path,
    model_records: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    per_stratum: int,
    seed: int,
) -> list[dict[str, Any]]:
    scenes = _read_jsonl(dataset_dir / "scenes.jsonl")
    pairs = _read_jsonl(dataset_dir / "counterfactual_pairs.jsonl")
    indexes = {
        name: {str(record["scene_id"]): record for record in records}
        for name, records in model_records.items()
    }
    common = set.intersection(*(set(index) for index in indexes.values()))
    relationships: dict[str, list[dict[str, Any]]] = {}
    for pair in pairs:
        for role, scene_key, other_key in (
            ("source", "source_scene_id", "target_scene_id"),
            ("target", "target_scene_id", "source_scene_id"),
        ):
            relationships.setdefault(str(pair[scene_key]), []).append(
                {
                    "pair_id": pair["pair_id"],
                    "role": role,
                    "other_scene_id": pair[other_key],
                    "edit_type": pair["edit_type"],
                    "delta_net_torque": pair["delta_net_torque"],
                    "intervention_split": pair["split"],
                }
            )

    candidates: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for scene in scenes:
        scene_id = str(scene["scene_id"])
        if scene_id not in common or scene["split"] not in ("probe_train", "probe_dev"):
            continue
        heuristics = _scene_heuristics(scene)
        if scene["problem_type"] == "multi_stack":
            suffix = (
                "weight_shortcut_agrees"
                if heuristics["total_weight"] == scene["derived"]["outcome"]
                else "weight_shortcut_conflicts"
            )
            stratum = f"multi_stack__{suffix}"
        else:
            stratum = str(scene["problem_type"])
        pattern = "__".join(
            f"{name}={'correct' if indexes[name][scene_id]['aggregated_correct'] else 'wrong'}"
            for name in sorted(indexes)
        )
        model_predictions = {
            name: {
                "predicted_semantic": indexes[name][scene_id][
                    "aggregated_predicted_semantic"
                ],
                "correct": indexes[name][scene_id]["aggregated_correct"],
                "gold_margin": indexes[name][scene_id]["aggregated_gold_margin"],
                "signed_left_right_margin": indexes[name][scene_id][
                    "aggregated_signed_left_right_margin"
                ],
                "semantic_scores": indexes[name][scene_id]["semantic_scores"],
                "raw_semantic_consistent": indexes[name][scene_id][
                    "raw_semantic_consistent"
                ],
            }
            for name in sorted(indexes)
        }
        candidate = {
            "schema_version": "activation-pilot-1.0",
            "record_type": "activation_pilot_scene",
            "scene_id": scene_id,
            "split": scene["split"],
            "analysis_stratum": stratum,
            "cross_model_pattern": pattern,
            "problem_type": scene["problem_type"],
            "image_path": scene["image_path"],
            "stacks": scene["stacks"],
            "derived": scene["derived"],
            "heuristics": heuristics,
            "model_predictions": model_predictions,
            "counterfactual_relationships": sorted(
                relationships.get(scene_id, []), key=lambda item: item["pair_id"]
            ),
        }
        candidates.setdefault((stratum, pattern), []).append(candidate)

    for (stratum, pattern), group in candidates.items():
        group.sort(
            key=lambda record: _stable_key(
                f"{stratum}:{pattern}:{record['scene_id']}", seed
            )
        )

    selected = []
    strata = sorted({key[0] for key in candidates})
    for stratum in strata:
        pattern_groups = {
            pattern: list(candidates[(candidate_stratum, pattern)])
            for candidate_stratum, pattern in candidates
            if candidate_stratum == stratum
        }
        while len([item for item in selected if item["analysis_stratum"] == stratum]) < per_stratum:
            progressed = False
            for pattern in sorted(pattern_groups):
                if pattern_groups[pattern]:
                    selected.append(pattern_groups[pattern].pop(0))
                    progressed = True
                    if len(
                        [
                            item
                            for item in selected
                            if item["analysis_stratum"] == stratum
                        ]
                    ) >= per_stratum:
                        break
            if not progressed:
                break
    selected.sort(key=lambda record: (record["analysis_stratum"], record["scene_id"]))
    return selected


def analyze_behavior_comparison(
    dataset_dir: str | Path,
    models: Sequence[BehaviorModelArtifacts],
    output_dir: str | Path,
    *,
    bootstrap_replicates: int = 5000,
    activation_scenes_per_stratum: int = 30,
    seed: int = 20260817,
) -> dict[str, str]:
    """Build paper-facing cross-model reports without running either model."""

    if len(models) < 2:
        raise ValueError("comparison requires at least two model artifacts")
    if len({model.name for model in models}) != len(models):
        raise ValueError("model artifact names must be unique")
    if bootstrap_replicates < 0:
        raise ValueError("bootstrap_replicates must be non-negative")
    if activation_scenes_per_stratum <= 0:
        raise ValueError("activation_scenes_per_stratum must be positive")

    dataset_root = Path(dataset_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    aggregated_by_model = {
        model.name: aggregate_permuted_outcomes(model.primary_predictions)
        for model in models
    }
    for name, records in aggregated_by_model.items():
        for record in records:
            record["model_name"] = name

    summaries = {
        name: _summarize_model(
            records,
            bootstrap_replicates=bootstrap_replicates,
            rng=rng,
        )
        for name, records in aggregated_by_model.items()
    }
    paired = {}
    for left, right in combinations(models, 2):
        key = f"{left.name}__vs__{right.name}"
        paired[key] = _paired_model_report(
            left.name,
            aggregated_by_model[left.name],
            right.name,
            aggregated_by_model[right.name],
            bootstrap_replicates=bootstrap_replicates,
            rng=rng,
        )

    controls = {}
    for model in models:
        result: dict[str, Any] = {}
        if model.control_predictions is not None:
            result["matched_subset"] = _control_report(
                _read_jsonl(model.control_predictions),
                bootstrap_replicates=bootstrap_replicates,
                rng=rng,
            )
        if model.full_symbolic_predictions is not None:
            symbolic = aggregate_permuted_outcomes(model.full_symbolic_predictions)
            result["full_outcome"] = _full_symbolic_report(
                aggregated_by_model[model.name],
                symbolic,
                bootstrap_replicates=bootstrap_replicates,
                rng=rng,
            )
        controls[model.name] = result

    activation_manifest = _activation_manifest(
        dataset_root,
        aggregated_by_model,
        per_stratum=activation_scenes_per_stratum,
        seed=seed,
    )
    all_aggregated = [
        record
        for name in sorted(aggregated_by_model)
        for record in aggregated_by_model[name]
    ]

    summary_path = destination / "model_summary.json"
    paired_path = destination / "paired_model_comparison.json"
    control_path = destination / "perception_computation_diagnostics.json"
    predictions_path = destination / "scene_level_predictions.jsonl"
    activation_path = destination / "activation_pilot_manifest.jsonl"
    summary_csv_path = destination / "model_summary.csv"
    problem_csv_path = destination / "problem_type_comparison.csv"
    control_csv_path = destination / "control_by_question_kind.csv"
    manifest_path = destination / "analysis_manifest.json"

    _write_json(summary_path, summaries)
    _write_json(paired_path, paired)
    _write_json(control_path, controls)
    _write_jsonl(predictions_path, all_aggregated)
    _write_jsonl(activation_path, activation_manifest)

    _write_csv(
        summary_csv_path,
        [
            {
                "model": name,
                "scene_count": report["scene_count"],
                "aggregated_accuracy": report["aggregated_accuracy"],
                "aggregated_ci_low": report[
                    "aggregated_accuracy_scene_bootstrap_95_ci"
                ][0],
                "aggregated_ci_high": report[
                    "aggregated_accuracy_scene_bootstrap_95_ci"
                ][1],
                "problem_type_macro_accuracy": report[
                    "problem_type_macro_accuracy"
                ],
                "raw_variant_accuracy": report["raw_variant_accuracy"],
                "all_six_correct_rate": report["all_six_correct_rate"],
                "semantic_consistency_rate": report[
                    "raw_semantic_consistency_rate"
                ],
                "torque_pearson": report["torque_margin_correlation"]["pearson"],
                "torque_spearman": report["torque_margin_correlation"]["spearman"],
                **{
                    f"predicted_{label}_share": share
                    for label, share in report["raw_predicted_label_shares"].items()
                },
            }
            for name, report in sorted(summaries.items())
        ],
    )
    problem_rows = []
    for name, report in sorted(summaries.items()):
        for problem_type, values in report["by_problem_type"].items():
            problem_rows.append(
                {"model": name, "problem_type": problem_type, **values}
            )
    _write_csv(problem_csv_path, problem_rows)

    control_rows = []
    for name, report in sorted(controls.items()):
        matched = report.get("matched_subset", {})
        for question_kind, values in matched.get("by_question_kind", {}).items():
            control_rows.append(
                {
                    "model": name,
                    "question_kind": question_kind,
                    **{
                        key: value
                        for key, value in values.items()
                        if key != "quadrants"
                    },
                }
            )
    _write_csv(control_csv_path, control_rows)

    outputs = {
        "model_summary": str(summary_path),
        "paired_model_comparison": str(paired_path),
        "perception_computation_diagnostics": str(control_path),
        "scene_level_predictions": str(predictions_path),
        "activation_pilot_manifest": str(activation_path),
        "model_summary_csv": str(summary_csv_path),
        "problem_type_comparison_csv": str(problem_csv_path),
        "control_by_question_kind_csv": str(control_csv_path),
    }
    _write_json(
        manifest_path,
        {
            "schema_version": "behavior-comparison-2.0",
            "dataset_dir": str(dataset_root),
            "models": [
                {
                    "name": model.name,
                    "primary_predictions": str(model.primary_predictions),
                    "control_predictions": (
                        str(model.control_predictions)
                        if model.control_predictions is not None
                        else None
                    ),
                    "full_symbolic_predictions": (
                        str(model.full_symbolic_predictions)
                        if model.full_symbolic_predictions is not None
                        else None
                    ),
                }
                for model in models
            ],
            "bootstrap_replicates": bootstrap_replicates,
            "activation_scenes_per_stratum": activation_scenes_per_stratum,
            "activation_manifest_scene_count": len(activation_manifest),
            "activation_manifest_allowed_splits": ["probe_train", "probe_dev"],
            "seed": seed,
            "outputs": outputs,
        },
    )
    outputs["analysis_manifest"] = str(manifest_path)
    return outputs

