import unittest

from visual_torque_toolkit.edits import (
    EditKind,
    add_unit,
    make_pair,
    redecompose_side,
    validate_pair,
)
from visual_torque_toolkit.world import Scene, Side, Stack


class EditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = Scene(
            "base",
            (Stack("l", Side.LEFT, 2, 2), Stack("r", Side.RIGHT, 3, 1)),
        )

    def test_executable_unit_edit_has_exact_delta(self) -> None:
        source = add_unit(self.scene, "l", "source")
        pair = make_pair("pair", self.scene, source, EditKind.ADD_UNIT)
        validate_pair(pair)
        self.assertEqual(pair.delta_net_torque, 2)
        self.assertEqual(pair.changed_stack_ids, ("l",))

    def test_bad_edit_declaration_is_rejected(self) -> None:
        source = add_unit(self.scene, "l", "source")
        with self.assertRaises(ValueError):
            make_pair("pair", self.scene, source, EditKind.MOVE_STACK)

    def test_redecomposition_preserves_outcome(self) -> None:
        multi = Scene(
            "multi",
            (
                Stack("l0", Side.LEFT, 1, 2),
                Stack("l1", Side.LEFT, 3, 2),
                Stack("r0", Side.RIGHT, 2, 3),
            ),
        )
        source = redecompose_side(
            multi,
            Side.LEFT,
            (Stack("l0", Side.LEFT, 2, 4),),
            "recomposed",
        )
        pair = make_pair("recompose", multi, source, EditKind.REDECOMPOSE)
        self.assertEqual(pair.delta_net_torque, 0)
        self.assertEqual(source.outcome, multi.outcome)


if __name__ == "__main__":
    unittest.main()
