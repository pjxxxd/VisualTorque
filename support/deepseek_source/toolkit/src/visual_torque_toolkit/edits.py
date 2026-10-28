"""Executable scene edits and strict counterfactual validation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .world import CounterfactualPair, Scene, Side, Stack


class EditKind(str, Enum):
    ADD_UNIT = "add_unit"
    REMOVE_UNIT = "remove_unit"
    MOVE_STACK = "move_stack"
    MIRROR = "mirror"
    REDECOMPOSE = "redecompose"


@dataclass(frozen=True, slots=True)
class StackChange:
    stack_id: str
    before: Stack | None
    after: Stack | None


@dataclass(frozen=True, slots=True)
class SceneDifference:
    changes: tuple[StackChange, ...]

    @property
    def changed_stack_ids(self) -> tuple[str, ...]:
        return tuple(change.stack_id for change in self.changes)


def _stack_map(scene: Scene) -> dict[str, Stack]:
    return {stack.stack_id: stack for stack in scene.stacks}


def diff_scenes(target: Scene, source: Scene) -> SceneDifference:
    target_by_id = _stack_map(target)
    source_by_id = _stack_map(source)
    changes = []
    for stack_id in sorted(target_by_id.keys() | source_by_id.keys()):
        before = target_by_id.get(stack_id)
        after = source_by_id.get(stack_id)
        if before != after:
            changes.append(StackChange(stack_id, before, after))
    return SceneDifference(tuple(changes))


def _replace_stack(scene: Scene, replacement: Stack, scene_id: str) -> Scene:
    if replacement.stack_id not in _stack_map(scene):
        raise KeyError(f"unknown stack_id: {replacement.stack_id}")
    stacks = tuple(replacement if stack.stack_id == replacement.stack_id else stack for stack in scene.stacks)
    return Scene(scene_id, stacks, metadata=(("edited_from", scene.scene_id),))


def add_unit(scene: Scene, stack_id: str, scene_id: str) -> Scene:
    stack = _stack_map(scene).get(stack_id)
    if stack is None:
        raise KeyError(f"unknown stack_id: {stack_id}")
    return _replace_stack(scene, Stack(stack.stack_id, stack.side, stack.distance, stack.units + 1), scene_id)


def remove_unit(scene: Scene, stack_id: str, scene_id: str) -> Scene:
    stack = _stack_map(scene).get(stack_id)
    if stack is None:
        raise KeyError(f"unknown stack_id: {stack_id}")
    if stack.units == 1:
        raise ValueError("remove_unit cannot remove the last unit in a stack")
    return _replace_stack(scene, Stack(stack.stack_id, stack.side, stack.distance, stack.units - 1), scene_id)


def move_stack(scene: Scene, stack_id: str, distance: int, scene_id: str) -> Scene:
    stack = _stack_map(scene).get(stack_id)
    if stack is None:
        raise KeyError(f"unknown stack_id: {stack_id}")
    if distance == stack.distance:
        raise ValueError("move_stack must change the distance")
    return _replace_stack(scene, Stack(stack.stack_id, stack.side, distance, stack.units), scene_id)


def redecompose_side(scene: Scene, side: Side, replacement: tuple[Stack, ...], scene_id: str) -> Scene:
    """Replace one side while preserving that side's total torque."""

    if not replacement or any(stack.side is not side for stack in replacement):
        raise ValueError("replacement must contain stacks from exactly the chosen side")
    old_torque = sum(stack.torque for stack in scene.stacks if stack.side is side)
    new_torque = sum(stack.torque for stack in replacement)
    if old_torque != new_torque:
        raise ValueError("redecomposition must preserve side torque")
    retained = tuple(stack for stack in scene.stacks if stack.side is not side)
    return Scene(scene_id, retained + replacement, metadata=(("redecomposed_from", scene.scene_id), ("side", side.value)))


def make_pair(
    pair_id: str,
    target: Scene,
    source: Scene,
    edit_type: EditKind,
    *,
    equivalence_key: str | None = None,
) -> CounterfactualPair:
    changed = validate_edit(target, source, edit_type)
    metadata = (("equivalence_key", equivalence_key),) if equivalence_key else ()
    return CounterfactualPair(pair_id, target, source, edit_type.value, changed, metadata)


def validate_pair(pair: CounterfactualPair) -> None:
    try:
        edit_type = EditKind(pair.edit_type)
    except ValueError as error:
        raise ValueError(f"unknown edit_type: {pair.edit_type}") from error
    actual = validate_edit(pair.target, pair.source, edit_type)
    if pair.changed_stack_ids and tuple(sorted(pair.changed_stack_ids)) != tuple(sorted(actual)):
        raise ValueError(f"declared changed_stack_ids {pair.changed_stack_ids} do not match {actual}")


def validate_edit(target: Scene, source: Scene, edit_type: EditKind) -> tuple[str, ...]:
    difference = diff_scenes(target, source)
    changes = difference.changes
    if not changes:
        raise ValueError("counterfactual scenes must differ")

    if edit_type in {EditKind.ADD_UNIT, EditKind.REMOVE_UNIT, EditKind.MOVE_STACK}:
        if len(changes) != 1:
            raise ValueError(f"{edit_type.value} must change exactly one stack")
        change = changes[0]
        if change.before is None or change.after is None:
            raise ValueError(f"{edit_type.value} cannot add or delete a stack")
        before, after = change.before, change.after
        if before.side is not after.side:
            raise ValueError(f"{edit_type.value} cannot change stack side")
        if edit_type is EditKind.ADD_UNIT:
            valid = after.distance == before.distance and after.units == before.units + 1
        elif edit_type is EditKind.REMOVE_UNIT:
            valid = after.distance == before.distance and after.units == before.units - 1
        else:
            valid = after.units == before.units and after.distance != before.distance
        if not valid:
            raise ValueError(f"scenes do not implement a valid {edit_type.value} edit")

    elif edit_type is EditKind.MIRROR:
        target_shape = sorted((stack.side.opposite.value, stack.distance, stack.units) for stack in target.stacks)
        source_shape = sorted((stack.side.value, stack.distance, stack.units) for stack in source.stacks)
        if target_shape != source_shape:
            raise ValueError("source is not an exact left/right mirror of target")

    elif edit_type is EditKind.REDECOMPOSE:
        changed_sides = {
            stack.side
            for change in changes
            for stack in (change.before, change.after)
            if stack is not None
        }
        if len(changed_sides) != 1:
            raise ValueError("redecompose must alter exactly one side")
        side = next(iter(changed_sides))
        target_torque = sum(stack.torque for stack in target.stacks if stack.side is side)
        source_torque = sum(stack.torque for stack in source.stacks if stack.side is side)
        if target_torque != source_torque:
            raise ValueError("redecompose must preserve changed-side torque")
        target_other = sorted((stack.stack_id, stack.distance, stack.units) for stack in target.stacks if stack.side is not side)
        source_other = sorted((stack.stack_id, stack.distance, stack.units) for stack in source.stacks if stack.side is not side)
        if target_other != source_other:
            raise ValueError("redecompose cannot change the opposite side")

    return difference.changed_stack_ids
