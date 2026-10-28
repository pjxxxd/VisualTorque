"""Behavioral evaluation utilities for multimodal torque reasoning."""

from .adapter import HuggingFaceMultimodalAdapter, ModelAdapter
from .comparison import (
    BehaviorModelArtifacts,
    aggregate_permuted_outcomes,
    analyze_behavior_comparison,
)
from .data import BehaviorExample, load_behavior_examples
from .metrics import analyze_behavior_run
from .parsing import parse_choice_label
from .runner import run_behavior_evaluation
from .schema import BehaviorRunConfig, PromptCondition

__all__ = [
    "BehaviorExample",
    "BehaviorModelArtifacts",
    "BehaviorRunConfig",
    "HuggingFaceMultimodalAdapter",
    "ModelAdapter",
    "PromptCondition",
    "analyze_behavior_run",
    "aggregate_permuted_outcomes",
    "analyze_behavior_comparison",
    "load_behavior_examples",
    "parse_choice_label",
    "run_behavior_evaluation",
]
