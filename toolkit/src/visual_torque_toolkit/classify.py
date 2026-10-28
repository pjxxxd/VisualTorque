"""Automatic classification of balance-scale problem structure."""

from __future__ import annotations

from collections.abc import Sequence

from .world import ProblemType, Side, Stack


def classify_problem_type(stacks: Sequence[Stack]) -> ProblemType:
    """Infer the structural problem type from exact scene geometry."""

    left = [stack for stack in stacks if stack.side is Side.LEFT]
    right = [stack for stack in stacks if stack.side is Side.RIGHT]
    if len(left) != 1 or len(right) != 1:
        return ProblemType.MULTI_STACK

    left_stack, right_stack = left[0], right[0]
    same_units = left_stack.units == right_stack.units
    same_distance = left_stack.distance == right_stack.distance
    if same_units and same_distance:
        return ProblemType.BALANCE
    if same_distance:
        return ProblemType.WEIGHT
    if same_units:
        return ProblemType.DISTANCE

    unit_direction = (left_stack.units > right_stack.units) - (left_stack.units < right_stack.units)
    distance_direction = (left_stack.distance > right_stack.distance) - (left_stack.distance < right_stack.distance)
    if unit_direction == distance_direction:
        return ProblemType.COMPATIBLE_WEIGHT_DISTANCE

    net_torque = left_stack.torque - right_stack.torque
    if net_torque == 0:
        return ProblemType.CONFLICT_BALANCE
    winning_direction = 1 if net_torque > 0 else -1
    if winning_direction == unit_direction:
        return ProblemType.CONFLICT_WEIGHT
    return ProblemType.CONFLICT_DISTANCE
