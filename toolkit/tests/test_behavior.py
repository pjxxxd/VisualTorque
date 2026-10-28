import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from visual_torque_toolkit.behavior.adapter import (
    CandidateScoreResult,
    HuggingFaceMultimodalAdapter,
)
from visual_torque_toolkit.behavior.data import load_behavior_examples
from visual_torque_toolkit.behavior.parsing import parse_choice_label
from visual_torque_toolkit.behavior.runner import run_behavior_evaluation
from visual_torque_toolkit.behavior.schema import BehaviorRunConfig, PromptCondition
from visual_torque_toolkit.dataset import build_dataset
from visual_torque_toolkit.generate import GenerationConfig


class AlwaysGoldAdapter:
    @property
    def metadata(self):
        return {
            "adapter": "AlwaysGoldAdapter",
            "model_id": "mock/gold",
            "requested_revision": "test",
            "resolved_revision": "test-commit",
        }

    def score_batch(self, examples):
        results = []
        for example in examples:
            scores = {label: -3.0 for label in example.labels}
            scores[example.gold_label] = -0.1
            results.append(CandidateScoreResult(scores, "mock", 0.0))
        return results

    def generate_batch(self, examples):
        return [example.gold_label for example in examples]


class BehaviorTests(unittest.TestCase):
    def test_multimodal_template_nests_processor_kwargs(self) -> None:
        class FakeProcessor:
            def apply_chat_template(self, conversations, **kwargs):
                self.conversations = conversations
                self.kwargs = kwargs
                return {"input_ids": [[1]], "attention_mask": [[1]]}

        adapter = HuggingFaceMultimodalAdapter.__new__(
            HuggingFaceMultimodalAdapter
        )
        adapter.model_id = "Qwen/Qwen3.5-9B"
        adapter.processor = FakeProcessor()
        result = adapter._template([[{"role": "user", "content": "hi"}]], add_generation_prompt=True)
        self.assertIn("input_ids", result)
        self.assertEqual(adapter.processor.kwargs["processor_kwargs"], {"padding": True})
        self.assertNotIn("padding", adapter.processor.kwargs)

    def test_generation_skips_unconfigured_response_parser(self) -> None:
        class FakeTokenizer:
            response_template = None
            response_schema = None

            def parse_response(self, *_args, **_kwargs):
                raise AssertionError("unconfigured parser must not be called")

        class FakeProcessor:
            tokenizer = FakeTokenizer()

            def decode(self, _tokens, *, skip_special_tokens):
                return "B" if skip_special_tokens else "<assistant>B"

        adapter = HuggingFaceMultimodalAdapter.__new__(
            HuggingFaceMultimodalAdapter
        )
        adapter.processor = FakeProcessor()
        parsed = adapter._parse_generated_content(
            "<assistant>B",
            prompt_tokens=[1, 2],
            generated_tokens=[3],
        )
        self.assertEqual(parsed, "B")

    def test_literal_candidate_plan_supports_non_prefix_chat_templates(self) -> None:
        class FakeTokenizer:
            token_by_label = {"A": 101, "B": 102, "C": 103}

            def __call__(self, text, *, add_special_tokens):
                self.add_special_tokens = add_special_tokens
                return {"input_ids": [self.token_by_label[text]]}

        class FakeProcessor:
            tokenizer = FakeTokenizer()

        adapter = HuggingFaceMultimodalAdapter.__new__(
            HuggingFaceMultimodalAdapter
        )
        adapter.processor = FakeProcessor()
        plan = adapter._literal_candidate_plan(("A", "B", "C"))
        self.assertEqual(plan, ((), {"A": 101, "B": 102, "C": 103}))
        self.assertFalse(adapter.processor.tokenizer.add_special_tokens)

    def test_literal_candidate_plan_defers_multitoken_labels(self) -> None:
        class FakeTokenizer:
            def __call__(self, text, *, add_special_tokens):
                return {"input_ids": [ord(character) for character in text]}

        class FakeProcessor:
            tokenizer = FakeTokenizer()

        adapter = HuggingFaceMultimodalAdapter.__new__(
            HuggingFaceMultimodalAdapter
        )
        adapter.processor = FakeProcessor()
        self.assertIsNone(adapter._literal_candidate_plan(("AA", "B")))

    def test_strict_choice_parser(self) -> None:
        labels = ("A", "B", "C")
        self.assertEqual(parse_choice_label("B", labels), "B")
        self.assertEqual(parse_choice_label('{"answer": "C"}', labels), "C")
        self.assertEqual(parse_choice_label("The answer is A.", labels), "A")
        self.assertIsNone(parse_choice_label("A or B", labels))

    def test_conditions_build_distinct_grounded_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "dataset"
            build_dataset(
                dataset,
                GenerationConfig(
                    max_units=1,
                    max_distance=2,
                    multi_scene_count=0,
                ),
            )
            config = BehaviorRunConfig(
                dataset_dir=dataset,
                output_dir=Path(temporary) / "results",
                model_id="mock/gold",
                conditions=(PromptCondition.IMAGE_RULE, PromptCondition.SYMBOLIC_RULE),
                max_scenes=1,
            )
            examples = load_behavior_examples(config)
            image = next(item for item in examples if item.condition is PromptCondition.IMAGE_RULE)
            symbolic = next(item for item in examples if item.condition is PromptCondition.SYMBOLIC_RULE)
            self.assertIsNotNone(image.image_path)
            self.assertIsNone(symbolic.image_path)
            self.assertIn("multiplied by its distance", image.prompt)
            self.assertIn("Exact symbolic scene description", symbolic.prompt)

    def test_runner_resume_and_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            output = root / "results"
            build_dataset(
                dataset,
                GenerationConfig(
                    max_units=1,
                    max_distance=2,
                    multi_scene_count=0,
                ),
            )
            config = BehaviorRunConfig(
                dataset_dir=dataset,
                output_dir=output,
                model_id="mock/gold",
                conditions=(PromptCondition.IMAGE_RULE, PromptCondition.SYMBOLIC_RULE),
                question_kinds=("outcome",),
                batch_size=3,
                max_scenes=2,
                generation_every=1,
            )
            first = run_behavior_evaluation(config, AlwaysGoldAdapter())
            second = run_behavior_evaluation(config, AlwaysGoldAdapter())
            self.assertGreater(first["progress"]["completed"], 0)
            self.assertEqual(second["progress"]["newly_completed"], 0)

            # A Colab session may extract the exact same Dataset archive to a
            # different temporary path. Also exercise one-time migration from
            # the old path-based run identity.
            copied_dataset = root / "dataset_copy"
            shutil.copytree(dataset, copied_dataset)
            run_config_path = output / "run_config.json"
            legacy_run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
            legacy_identity = legacy_run_config["identity"]
            legacy_identity.pop("dataset_fingerprint")
            legacy_identity["dataset_dir"] = str(dataset.resolve())
            run_config_path.write_text(
                json.dumps(legacy_run_config),
                encoding="utf-8",
            )
            moved_config = replace(config, dataset_dir=copied_dataset)
            moved = run_behavior_evaluation(moved_config, AlwaysGoldAdapter())
            self.assertEqual(moved["progress"]["newly_completed"], 0)
            migrated_run_config = json.loads(
                run_config_path.read_text(encoding="utf-8")
            )
            self.assertIn("dataset_fingerprint", migrated_run_config["identity"])
            self.assertNotIn("dataset_dir", migrated_run_config["identity"])

            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["overall"]["likelihood_accuracy"], 1.0)
            self.assertEqual(metrics["overall"]["generation_parse_rate"], 1.0)
            self.assertTrue((output / "counterfactual_consistency.json").is_file())


if __name__ == "__main__":
    unittest.main()
