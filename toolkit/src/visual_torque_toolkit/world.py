"""Exact world records and the deterministic torque oracle."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


SCHEMA_VERSION = "1.0"


class Side(str, Enum):
    LEFT = "left"
    RIGHT = "right"

    @property
    def opposite(self) -> "Side":
        return Side.RIGHT if self is Side.LEFT else Side.LEFT


class Outcome(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    BALANCED = "balanced"


class ProblemType(str, Enum):
    BALANCE = "balance"
    WEIGHT = "weight"
    DISTANCE = "distance"
    COMPATIBLE_WEIGHT_DISTANCE = "compatible_weight_distance"
    CONFLICT_WEIGHT = "conflict_weight"
    CONFLICT_DISTANCE = "conflict_distance"
    CONFLICT_BALANCE = "conflict_balance"
    MULTI_STACK = "multi_stack"


def _require_positive_int(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class Stack:
    stack_id: str
    side: Side
    distance: int
    units: int

    def __post_init__(self) -> None:
        if not isinstance(self.stack_id, str) or not self.stack_id.strip():
            raise ValueError("stack_id must be a non-empty string")
        if not isinstance(self.side, Side):
            raise TypeError("side must be a Side value")
        _require_positive_int(self.distance, "distance")
        _require_positive_int(self.units, "units")

    @property
    def torque(self) -> int:
        return self.distance * self.units

    @property
    def signed_torque(self) -> int:
        return self.torque if self.side is Side.LEFT else -self.torque

    def to_record(self) -> dict[str, Any]:
        return {
            "stack_id": self.stack_id,
            "side": self.side.value,
            "distance": self.distance,
            "units": self.units,
            "torque": self.torque,
        }


@dataclass(frozen=True, slots=True)
class Scene:
    scene_id: str
    stacks: tuple[Stack, ...]
    problem_type: ProblemType | None = None
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.scene_id, str) or not self.scene_id.strip():
            raise ValueError("scene_id must be a non-empty string")
        if not isinstance(self.stacks, tuple) or not self.stacks:
            raise ValueError("stacks must be a non-empty tuple")
        if any(not isinstance(stack, Stack) for stack in self.stacks):
            raise TypeError("every scene entry must be a Stack")
        stack_ids = [stack.stack_id for stack in self.stacks]
        if len(stack_ids) != len(set(stack_ids)):
            raise ValueError("stack_id values must be unique within a scene")
        positions = [(stack.side, stack.distance) for stack in self.stacks]
        if len(positions) != len(set(positions)):
            raise ValueError("combine blocks at the same side and distance into one stack")
        if any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not all(isinstance(value, str) for value in item)
            for item in self.metadata
        ):
            raise TypeError("metadata must contain (str, str) tuples")

        # A local import avoids a module-level cycle: the classifier uses these
        # world enums, while every scene verifies its own structural label.
        from .classify import classify_problem_type

        inferred = classify_problem_type(self.stacks)
        if self.problem_type is None:
            object.__setattr__(self, "problem_type", inferred)
        elif not isinstance(self.problem_type, ProblemType):
            raise TypeError("problem_type must be a ProblemType value")
        elif self.problem_type is not inferred:
            raise ValueError(
                f"problem_type={self.problem_type.value!r} disagrees with "
                f"the scene geometry ({inferred.value!r})"
            )

    @property
    def left_torque(self) -> int:
        return sum(stack.torque for stack in self.stacks if stack.side is Side.LEFT)

    @property
    def right_torque(self) -> int:
        return sum(stack.torque for stack in self.stacks if stack.side is Side.RIGHT)

    @property
    def net_torque(self) -> int:
        return self.left_torque - self.right_torque

    @property
    def outcome(self) -> Outcome:
        if self.net_torque > 0:
            return Outcome.LEFT
        if self.net_torque < 0:
            return Outcome.RIGHT
        return Outcome.BALANCED

    @property
    def left_units(self) -> int:
        return sum(stack.units for stack in self.stacks if stack.side is Side.LEFT)

    @property
    def right_units(self) -> int:
        return sum(stack.units for stack in self.stacks if stack.side is Side.RIGHT)

    @property
    def max_distance(self) -> int:
        return max(stack.distance for stack in self.stacks)

    @property
    def canonical_signature(self) -> tuple[tuple[str, int, int], ...]:
        """A stable, ID-independent description of scene geometry."""

        return tuple(
            sorted(
                (stack.side.value, stack.distance, stack.units)
                for stack in self.stacks
            )
        )

    def mirrored(self, scene_id: str | None = None) -> "Scene":
        """Return a left/right reflection with stable stack identities."""

        return Scene(
            scene_id=scene_id or f"{self.scene_id}__mirror",
            stacks=tuple(
                Stack(
                    stack_id=stack.stack_id,
                    side=stack.side.opposite,
                    distance=stack.distance,
                    units=stack.units,
                )
                for stack in self.stacks
            ),
            metadata=self.metadata + (("mirrored_from", self.scene_id),),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "record_type": "scene",
            "scene_id": self.scene_id,
            "problem_type": self.problem_type.value,
            "stacks": [stack.to_record() for stack in self.stacks],
            "derived": {
                "left_torque": self.left_torque,
                "right_torque": self.right_torque,
                "net_torque": self.net_torque,
                "outcome": self.outcome.value,
                "left_units": self.left_units,
                "right_units": self.right_units,
                "stack_count": len(self.stacks),
                "max_distance": self.max_distance,
            },
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class CounterfactualPair:
    pair_id: str
    target: Scene
    source: Scene
    edit_type: str
    changed_stack_ids: tuple[str, ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.pair_id, str) or not self.pair_id.strip():
            raise ValueError("pair_id must be a non-empty string")
        if not isinstance(self.edit_type, str) or not self.edit_type.strip():
            raise ValueError("edit_type must be a non-empty string")
        if self.target.scene_id == self.source.scene_id:
            raise ValueError("target and source must be distinct scenes")
        if len(self.changed_stack_ids) != len(set(self.changed_stack_ids)):
            raise ValueError("changed_stack_ids must not contain duplicates")

    @property
    def delta_net_torque(self) -> int:
        return self.source.net_torque - self.target.net_torque

    @property
    def equivalence_key(self) -> str:
        metadata = dict(self.metadata)
        return metadata.get(
            "equivalence_key", f"{self.edit_type}:delta={self.delta_net_torque}"
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "record_type": "counterfactual_pair",
            "pair_id": self.pair_id,
            "target_scene_id": self.target.scene_id,
            "source_scene_id": self.source.scene_id,
            "edit_type": self.edit_type,
            "changed_stack_ids": list(self.changed_stack_ids),
            "delta_net_torque": self.delta_net_torque,
            "target_outcome": self.target.outcome.value,
            "source_outcome": self.source.outcome.value,
            "equivalence_key": self.equivalence_key,
            "metadata": dict(self.metadata),
        }


def _stack(stack_id: str, side: Side, distance: int, units: int) -> Stack:
    return Stack(stack_id=stack_id, side=side, distance=distance, units=units)


def make_demo_world() -> tuple[tuple[Scene, ...], tuple[CounterfactualPair, ...]]:
    """Return a few hand-checked scenes and counterfactual relationships."""

    scenes = (
        Scene("balance", (_stack("l", Side.LEFT, 2, 2), _stack("r", Side.RIGHT, 2, 2)), ProblemType.BALANCE),
        Scene("weight", (_stack("l", Side.LEFT, 2, 3), _stack("r", Side.RIGHT, 2, 2)), ProblemType.WEIGHT),
        Scene("distance", (_stack("l", Side.LEFT, 3, 2), _stack("r", Side.RIGHT, 2, 2)), ProblemType.DISTANCE),
        Scene("compatible", (_stack("l", Side.LEFT, 3, 3), _stack("r", Side.RIGHT, 2, 2)), ProblemType.COMPATIBLE_WEIGHT_DISTANCE),
        Scene("conflict_weight", (_stack("l", Side.LEFT, 2, 4), _stack("r", Side.RIGHT, 3, 2)), ProblemType.CONFLICT_WEIGHT),
        Scene("conflict_distance", (_stack("l", Side.LEFT, 2, 3), _stack("r", Side.RIGHT, 4, 2)), ProblemType.CONFLICT_DISTANCE),
        Scene("conflict_balance", (_stack("l", Side.LEFT, 2, 4), _stack("r", Side.RIGHT, 4, 2)), ProblemType.CONFLICT_BALANCE),
        Scene(
            "multi_stack",
            (
                _stack("l_near", Side.LEFT, 3, 2),
                _stack("l_far", Side.LEFT, 5, 1),
                _stack("r_near", Side.RIGHT, 2, 3),
                _stack("r_far", Side.RIGHT, 4, 1),
            ),
            ProblemType.MULTI_STACK,
        ),
        Scene("weight_edit_plus_2", (_stack("l", Side.LEFT, 2, 5), _stack("r", Side.RIGHT, 3, 2)), ProblemType.CONFLICT_WEIGHT),
        Scene("distance_edit_plus_2", (_stack("l", Side.LEFT, 2, 4), _stack("r", Side.RIGHT, 2, 2)), ProblemType.WEIGHT),
        Scene(
            "multi_redecomposition",
            (
                _stack("l_near", Side.LEFT, 1, 1),
                _stack("l_far", Side.LEFT, 5, 2),
                _stack("r_near", Side.RIGHT, 2, 3),
                _stack("r_far", Side.RIGHT, 4, 1),
            ),
            ProblemType.MULTI_STACK,
        ),
    )
    by_id = {scene.scene_id: scene for scene in scenes}
    pairs = (
        CounterfactualPair("same_delta_via_weight", by_id["conflict_weight"], by_id["weight_edit_plus_2"], "add_unit", ("l",)),
        CounterfactualPair("same_delta_via_distance", by_id["conflict_weight"], by_id["distance_edit_plus_2"], "move_stack", ("r",)),
        CounterfactualPair("same_torque_new_decomposition", by_id["multi_stack"], by_id["multi_redecomposition"], "redecompose", ("l_near", "l_far")),
    )
    return scenes, pairs
