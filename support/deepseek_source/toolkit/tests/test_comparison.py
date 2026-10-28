import itertools
import json
import tempfile
import unittest
from pathlib import Path

from visual_torque_toolkit.behavior import (
    BehaviorModelArtifacts,
    aggregate_permuted_outcomes,
    analyze_behavior_comparison,
)


SEMANTICS = ("left", "right", "balanced")
LABELS = ("A", "B", "C")


def prediction_records(
    *,
    model: str,
    condition: str,
    favored_semantic: str,
    label_bias: dict[str, float] | None = None,
) -> list[dict]:
    label_bias = label_bias or {}
    records = []
    for index, permutation in enumerate(itertools.permutations(SEMANTICS)):
        mapping = dict(zip(LABELS, permutation))
        scores = {
            label: (2.0 if semantic == favored_semantic else 0.0)
            + label_bias.get(label, 0.0)
            for label, semantic in mapping.items()
        }
        predicted_label = max(sorted(scores), key=lambda label: scores[label])
        predicted_semantic = mapping[predicted_label]
        records.append(
            {
                "example_id": f"{condition}:scene_1:permutation_{index}",
                "model_id": model,
                "model_revision": "test",
                "condition": condition,
                "question_id": f"scene_1:outcome:permutation_{index}",
                "scene_id": "scene_1",
                "question_kind": "outcome",
                "variant": f"permutation_{index}",
                "split": "probe_train",
                "problem_type": "conflict_distance",
                "gold_outcome": "left",
                "net_torque": 1,
                "label_to_semantic": mapping,
                "gold_label": next(
                    label for label, semantic in mapping.items() if semantic == "left"
                ),
                "gold_semantic": "left",
                "candidate_logprobs": scores,
                "scoring_mode": "test",
                "predicted_label": predicted_label,
                "predicted_semantic": predicted_semantic,
                "likelihood_correct": predicted_semantic == "left",
            }
        )
    return records


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class ComparisonTests(unittest.TestCase):
    def test_permutation_aggregation_cancels_fixed_label_bias(self) -> None:
        rows = prediction_records(
            model="qwen",
            condition="image_rule",
            favored_semantic="left",
            label_bias={"C": 3.0},
        )
        result = aggregate_permuted_outcomes(rows)[0]
        self.assertAlmostEqual(result["raw_variant_accuracy"], 1 / 3)
        self.assertEqual(result["raw_predicted_label_counts"], {"C": 6})
        self.assertTrue(result["complete_six_permutations"])
        self.assertEqual(result["aggregated_predicted_semantic"], "left")
        self.assertTrue(result["aggregated_correct"])

    def test_comparison_writes_scene_reports_and_discovery_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            scene = {
                "scene_id": "scene_1",
                "split": "probe_train",
                "problem_type": "conflict_distance",
                "image_path": "images/scene_1.png",
                "stacks": [
                    {
                        "stack_id": "l0",
                        "side": "left",
                        "units": 1,
                        "distance": 3,
                        "torque": 3,
                    },
                    {
                        "stack_id": "r0",
                        "side": "right",
                        "units": 2,
                        "distance": 1,
                        "torque": 2,
                    },
                ],
                "derived": {
                    "outcome": "left",
                    "net_torque": 1,
                    "left_torque": 3,
                    "right_torque": 2,
                    "left_units": 1,
                    "right_units": 2,
                    "stack_count": 2,
                    "max_distance": 3,
                },
            }
            write_jsonl(dataset / "scenes.jsonl", [scene])
            write_jsonl(dataset / "counterfactual_pairs.jsonl", [])

            qwen_primary = root / "qwen_primary.jsonl"
            gemma_primary = root / "gemma_primary.jsonl"
            qwen_control = root / "qwen_control.jsonl"
            gemma_control = root / "gemma_control.jsonl"
            write_jsonl(
                qwen_primary,
                prediction_records(
                    model="qwen",
                    condition="image_rule",
                    favored_semantic="left",
                    label_bias={"C": 3.0},
                ),
            )
            write_jsonl(
                gemma_primary,
                prediction_records(
                    model="gemma",
                    condition="image_rule",
                    favored_semantic="right",
                ),
            )
            write_jsonl(
                qwen_control,
                prediction_records(
                    model="qwen",
                    condition="image_rule",
                    favored_semantic="right",
                )
                + prediction_records(
                    model="qwen",
                    condition="symbolic_rule",
                    favored_semantic="left",
                ),
            )
            write_jsonl(
                gemma_control,
                prediction_records(
                    model="gemma",
                    condition="image_rule",
                    favored_semantic="right",
                )
                + prediction_records(
                    model="gemma",
                    condition="symbolic_rule",
                    favored_semantic="left",
                ),
            )

            output = root / "comparison"
            paths = analyze_behavior_comparison(
                dataset,
                (
                    BehaviorModelArtifacts(
                        "Qwen",
                        qwen_primary,
                        qwen_control,
                    ),
                    BehaviorModelArtifacts(
                        "Gemma",
                        gemma_primary,
                        gemma_control,
                    ),
                ),
                output,
                bootstrap_replicates=20,
                activation_scenes_per_stratum=1,
            )
            for path in paths.values():
                self.assertTrue(Path(path).is_file(), path)

            summaries = json.loads(
                (output / "model_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summaries["Qwen"]["aggregated_accuracy"], 1.0)
            self.assertEqual(summaries["Gemma"]["aggregated_accuracy"], 0.0)

            controls = json.loads(
                (output / "perception_computation_diagnostics.json").read_text(
                    encoding="utf-8"
                )
            )
            qwen_outcome = controls["Qwen"]["matched_subset"][
                "outcome_permutation_aggregated_scene_pairs"
            ]
            self.assertEqual(qwen_outcome["image_accuracy"], 0.0)
            self.assertEqual(qwen_outcome["symbolic_accuracy"], 1.0)

            manifest = [
                json.loads(line)
                for line in (output / "activation_pilot_manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(manifest), 1)
            self.assertEqual(manifest[0]["split"], "probe_train")
            self.assertIn("Qwen", manifest[0]["model_predictions"])


if __name__ == "__main__":
    unittest.main()

