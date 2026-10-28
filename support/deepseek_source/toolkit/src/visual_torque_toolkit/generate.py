"""Deterministic scene and counterfactual generation for Dataset v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
import json

from .edits import EditKind, make_pair
from .world import CounterfactualPair, Scene, Side, Stack


SideShape = tuple[tuple[int, int], ...]
SceneShape = tuple[SideShape, SideShape]


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    max_units: int = 5
    max_distance: int = 6
    multi_max_units: int = 4
    multi_scene_count: int = 1200
    seed: int = 20260817

    def __post_init__(self) -> None:
        for name in ("max_units", "max_distance", "multi_max_units"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_distance < 2 and self.multi_scene_count:
            raise ValueError("multi-stack scenes require max_distance >= 2")
        if self.multi_scene_count < 0:
            raise ValueError("multi_scene_count must be non-negative")

    def to_record(self) -> dict[str, int]:
        return asdict(self)


def _stable_digest(value: object, seed: int = 0) -> str:
    payload = json.dumps([seed, value], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _scene_id(shape: SceneShape, prefix: str = "multi") -> str:
    return f"{prefix}_{_stable_digest(shape)[:16]}"


def _shape_from_scene(scene: Scene) -> SceneShape:
    left = tuple(sorted((s.distance, s.units) for s in scene.stacks if s.side is Side.LEFT))
    right = tuple(sorted((s.distance, s.units) for s in scene.stacks if s.side is Side.RIGHT))
    return left, right


def _scene_from_shape(shape: SceneShape, scene_id: str | None = None) -> Scene:
    left, right = shape
    stacks = tuple(
        Stack(f"l{index}", Side.LEFT, distance, units)
        for index, (distance, units) in enumerate(left)
    ) + tuple(
        Stack(f"r{index}", Side.RIGHT, distance, units)
        for index, (distance, units) in enumerate(right)
    )
    return Scene(scene_id or _scene_id(shape), stacks)


def generate_single_stack_scenes(config: GenerationConfig) -> tuple[Scene, ...]:
    """Exhaustively enumerate the full bounded single-stack grid."""

    scenes = []
    for left_units, left_distance, right_units, right_distance in itertools.product(
        range(1, config.max_units + 1),
        range(1, config.max_distance + 1),
        range(1, config.max_units + 1),
        range(1, config.max_distance + 1),
    ):
        scene_id = (
            f"single_lu{left_units}_ld{left_distance}_"
            f"ru{right_units}_rd{right_distance}"
        )
        scenes.append(
            Scene(
                scene_id,
                (
                    Stack("l0", Side.LEFT, left_distance, left_units),
                    Stack("r0", Side.RIGHT, right_distance, right_units),
                ),
                metadata=(("family", "single_stack"),),
            )
        )
    return tuple(scenes)


def _side_shapes(config: GenerationConfig) -> tuple[SideShape, ...]:
    one_stack = [
        ((distance, units),)
        for distance in range(1, config.max_distance + 1)
        for units in range(1, config.multi_max_units + 1)
    ]
    two_stack = [
        ((near, near_units), (far, far_units))
        for near, far in itertools.combinations(range(1, config.max_distance + 1), 2)
        for near_units in range(1, config.multi_max_units + 1)
        for far_units in range(1, config.multi_max_units + 1)
    ]
    return tuple(one_stack + two_stack)


def generate_multi_stack_scenes(config: GenerationConfig) -> tuple[Scene, ...]:
    """Select a deterministic, mirror-closed sample of multi-stack scenes."""

    if config.multi_scene_count == 0:
        return ()
    side_shapes = _side_shapes(config)
    candidates = [
        (left, right)
        for left in side_shapes
        for right in side_shapes
        if len(left) > 1 or len(right) > 1
    ]
    if config.multi_scene_count > len(candidates):
        raise ValueError(
            f"requested {config.multi_scene_count} multi-stack scenes, but the "
            f"configured bounds contain only {len(candidates)}"
        )
    candidates.sort(key=lambda shape: _stable_digest(shape, config.seed))

    selected: dict[SceneShape, None] = {}
    for shape in candidates:
        if len(selected) >= config.multi_scene_count:
            break
        mirror = (shape[1], shape[0])
        selected.setdefault(shape, None)
        if len(selected) < config.multi_scene_count:
            selected.setdefault(mirror, None)

    scenes = [_scene_from_shape(shape) for shape in selected]
    scenes.sort(key=lambda scene: scene.scene_id)
    return tuple(scenes)


def generate_scenes(config: GenerationConfig = GenerationConfig()) -> tuple[Scene, ...]:
    scenes = generate_single_stack_scenes(config) + generate_multi_stack_scenes(config)
    signatures = [scene.canonical_signature for scene in scenes]
    if len(signatures) != len(set(signatures)):
        raise RuntimeError("generator produced duplicate scene geometry")
    return scenes


def _pair_id(target: Scene, source: Scene, edit_type: EditKind) -> str:
    key = (target.scene_id, source.scene_id, edit_type.value)
    return f"pair_{_stable_digest(key)[:18]}"


def generate_counterfactual_pairs(
    scenes: tuple[Scene, ...],
) -> tuple[CounterfactualPair, ...]:
    """Create only counterfactuals whose edited endpoint exists in ``scenes``."""

    by_shape = {_shape_from_scene(scene): scene for scene in scenes}
    pairs: dict[tuple[str, str, str], CounterfactualPair] = {}

    def register(target: Scene, source: Scene, kind: EditKind) -> None:
        key = (target.scene_id, source.scene_id, kind.value)
        if key in pairs:
            return
        delta = source.net_torque - target.net_torque
        equivalence_key = (
            f"torque_delta={delta:+d}"
            if kind in {EditKind.ADD_UNIT, EditKind.REMOVE_UNIT, EditKind.MOVE_STACK}
            else f"{kind.value}:delta={delta:+d}"
        )
        pairs[key] = make_pair(
            _pair_id(target, source, kind),
            target,
            source,
            kind,
            equivalence_key=equivalence_key,
        )

    for target in scenes:
        left_shape, right_shape = _shape_from_scene(target)

        # Unit and one-tick distance edits. Requiring the source to be present
        # makes every declared intervention independently renderable.
        for side_index, side_shape in enumerate((left_shape, right_shape)):
            for stack_index, (distance, units) in enumerate(side_shape):
                modified_sides = [list(left_shape), list(right_shape)]
                modified_sides[side_index][stack_index] = (distance, units + 1)
                modified_shape = (
                    tuple(sorted(modified_sides[0])),
                    tuple(sorted(modified_sides[1])),
                )
                source = by_shape.get(modified_shape)
                if source is not None:
                    register(target, source, EditKind.ADD_UNIT)

                for new_distance in (distance - 1, distance + 1):
                    if new_distance <= 0:
                        continue
                    occupied = {d for index, (d, _) in enumerate(side_shape) if index != stack_index}
                    if new_distance in occupied:
                        continue
                    moved_sides = [list(left_shape), list(right_shape)]
                    moved_sides[side_index][stack_index] = (new_distance, units)
                    moved_shape = (
                        tuple(sorted(moved_sides[0])),
                        tuple(sorted(moved_sides[1])),
                    )
                    source = by_shape.get(moved_shape)
                    if source is not None:
                        register(target, source, EditKind.MOVE_STACK)

        mirror = by_shape.get((right_shape, left_shape))
        if mirror is not None and target.scene_id < mirror.scene_id:
            register(target, mirror, EditKind.MIRROR)

    # Same-torque, different-decomposition pairs with an identical opposite side.
    for changed_side in (Side.LEFT, Side.RIGHT):
        groups: dict[tuple[SideShape, int], list[Scene]] = {}
        for scene in scenes:
            left_shape, right_shape = _shape_from_scene(scene)
            changed_shape = left_shape if changed_side is Side.LEFT else right_shape
            unchanged_shape = right_shape if changed_side is Side.LEFT else left_shape
            torque = sum(distance * units for distance, units in changed_shape)
            groups.setdefault((unchanged_shape, torque), []).append(scene)
        for group in groups.values():
            group.sort(key=lambda scene: scene.scene_id)
            for target, source in zip(group[::2], group[1::2]):
                register(target, source, EditKind.REDECOMPOSE)

    return tuple(sorted(pairs.values(), key=lambda pair: pair.pair_id))
