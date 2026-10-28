"""Model-agnostic dataset and experiment building blocks for VisualTorque."""

from .classify import classify_problem_type
from .contracts import ActivationSite, CaptureRequest, InterventionRequest, ProbeTarget
from .dataset import DatasetSummary, build_dataset, compute_statistics, validate_dataset
from .edits import (
    EditKind,
    SceneDifference,
    StackChange,
    add_unit,
    diff_scenes,
    make_pair,
    move_stack,
    redecompose_side,
    remove_unit,
    validate_pair,
)
from .generate import (
    GenerationConfig,
    generate_counterfactual_pairs,
    generate_multi_stack_scenes,
    generate_scenes,
    generate_single_stack_scenes,
)
from .prompts import QuestionKind, QuestionRecord, generate_all_questions, generate_questions
from .render import RenderStyle, render_scene
from .splits import SplitConfig, SplitManifest, build_splits, validate_splits
from .world import (
    CounterfactualPair,
    Outcome,
    ProblemType,
    Scene,
    Side,
    Stack,
    make_demo_world,
)

__all__ = [
    "ActivationSite",
    "CaptureRequest",
    "CounterfactualPair",
    "DatasetSummary",
    "EditKind",
    "GenerationConfig",
    "InterventionRequest",
    "Outcome",
    "ProbeTarget",
    "ProblemType",
    "QuestionKind",
    "QuestionRecord",
    "RenderStyle",
    "Scene",
    "SceneDifference",
    "Side",
    "SplitConfig",
    "SplitManifest",
    "Stack",
    "StackChange",
    "add_unit",
    "build_dataset",
    "build_splits",
    "classify_problem_type",
    "compute_statistics",
    "diff_scenes",
    "generate_all_questions",
    "generate_counterfactual_pairs",
    "generate_multi_stack_scenes",
    "generate_questions",
    "generate_scenes",
    "generate_single_stack_scenes",
    "make_demo_world",
    "make_pair",
    "move_stack",
    "redecompose_side",
    "remove_unit",
    "render_scene",
    "validate_dataset",
    "validate_pair",
    "validate_splits",
]
