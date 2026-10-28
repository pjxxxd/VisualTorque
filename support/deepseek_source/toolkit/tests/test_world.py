import unittest

from visual_torque_toolkit.classify import classify_problem_type
from visual_torque_toolkit.world import ProblemType, Scene, Side, Stack, make_demo_world


class WorldTests(unittest.TestCase):
    def test_demo_covers_every_problem_type(self) -> None:
        scenes, _ = make_demo_world()
        self.assertEqual({scene.problem_type for scene in scenes}, set(ProblemType))

    def test_scene_infers_and_rejects_wrong_type(self) -> None:
        stacks = (
            Stack("l", Side.LEFT, 2, 3),
            Stack("r", Side.RIGHT, 2, 1),
        )
        self.assertIs(classify_problem_type(stacks), ProblemType.WEIGHT)
        self.assertIs(Scene("auto", stacks).problem_type, ProblemType.WEIGHT)
        with self.assertRaises(ValueError):
            Scene("wrong", stacks, ProblemType.DISTANCE)

    def test_mirror_reverses_net_torque(self) -> None:
        scene = Scene(
            "original",
            (Stack("l", Side.LEFT, 3, 2), Stack("r", Side.RIGHT, 1, 1)),
        )
        mirror = scene.mirrored()
        self.assertEqual(mirror.net_torque, -scene.net_torque)
        self.assertEqual(mirror.mirrored("roundtrip").canonical_signature, scene.canonical_signature)

    def test_invalid_geometry_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Stack("bad", Side.LEFT, 0, 1)
        with self.assertRaises(ValueError):
            Scene(
                "collision",
                (Stack("a", Side.LEFT, 1, 1), Stack("b", Side.LEFT, 1, 2)),
            )


if __name__ == "__main__":
    unittest.main()
