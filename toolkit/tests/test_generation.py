import unittest

from visual_torque_toolkit.edits import validate_pair
from visual_torque_toolkit.generate import (
    GenerationConfig,
    generate_counterfactual_pairs,
    generate_scenes,
)
from visual_torque_toolkit.prompts import QuestionKind, generate_questions
from visual_torque_toolkit.splits import SplitConfig, build_splits


class GenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = GenerationConfig(
            max_units=2,
            max_distance=3,
            multi_max_units=2,
            multi_scene_count=20,
            seed=7,
        )

    def test_generation_is_exhaustive_and_deterministic(self) -> None:
        first = generate_scenes(self.config)
        second = generate_scenes(self.config)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2 * 3 * 2 * 3 + 20)
        self.assertEqual(len({scene.canonical_signature for scene in first}), len(first))

    def test_all_generated_pairs_are_executable(self) -> None:
        scenes = generate_scenes(self.config)
        pairs = generate_counterfactual_pairs(scenes)
        self.assertGreater(len(pairs), 0)
        for pair in pairs:
            validate_pair(pair)
        self.assertIn("mirror", {pair.edit_type for pair in pairs})

    def test_outcome_options_cover_all_permutations(self) -> None:
        scene = generate_scenes(self.config)[0]
        questions = generate_questions(scene)
        outcomes = [q for q in questions if q.kind is QuestionKind.OUTCOME]
        self.assertEqual(len(outcomes), 6)
        self.assertEqual(len({q.option_semantics for q in outcomes}), 6)
        self.assertTrue(all(q.semantic_answer == scene.outcome.value for q in outcomes))
        self.assertIn(QuestionKind.LEFT_DISTANCE, {q.kind for q in questions})
        self.assertIn(QuestionKind.RIGHT_DISTANCE, {q.kind for q in questions})

    def test_strict_split_invariants(self) -> None:
        scenes = generate_scenes(self.config)
        pairs = generate_counterfactual_pairs(scenes)
        questions = tuple(q for scene in scenes for q in generate_questions(scene))
        splits = build_splits(scenes, pairs, questions, self.config, SplitConfig(seed=7))
        heldout = set(splits.heldout_unit_distance)
        for scene in scenes:
            if splits.scene_split[scene.scene_id] != "probe_test":
                self.assertFalse(any((s.units, s.distance) in heldout for s in scene.stacks))
        family_splits = {}
        for pair in pairs:
            value = splits.pair_split[pair.pair_id]
            self.assertEqual(family_splits.setdefault(pair.equivalence_key, value), value)


if __name__ == "__main__":
    unittest.main()
