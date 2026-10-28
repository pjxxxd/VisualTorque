"""Model-independent records for later probe and intervention code."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ActivationSite:
    model_family: str
    normalized_depth: float
    semantic_role: str
    module_hint: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.normalized_depth <= 1.0:
            raise ValueError("normalized_depth must lie in [0, 1]")
        if not self.model_family or not self.semantic_role:
            raise ValueError("model_family and semantic_role must not be empty")


@dataclass(frozen=True, slots=True)
class CaptureRequest:
    scene_id: str
    image_path: Path
    prompt: str
    sites: tuple[ActivationSite, ...]


@dataclass(frozen=True, slots=True)
class InterventionRequest:
    target_scene_id: str
    source_scene_id: str
    site: ActivationSite
    alpha: float = 1.0
    expected_delta_net_torque: int | None = None


@dataclass(frozen=True, slots=True)
class ProbeTarget:
    name: str
    manifest_field: str
    target_kind: str

