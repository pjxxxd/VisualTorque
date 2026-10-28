"""Command-line interface for VisualTorque behavior experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .behavior import (
    BehaviorRunConfig,
    HuggingFaceMultimodalAdapter,
    PromptCondition,
    analyze_behavior_run,
    run_behavior_evaluation,
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="visual-torque-behavior")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run or resume model inference")
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--revision", default="main")
    run.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32", "auto"))
    run.add_argument(
        "--conditions",
        nargs="+",
        default=[PromptCondition.IMAGE_RULE.value],
        choices=[condition.value for condition in PromptCondition],
    )
    run.add_argument("--question-kinds", nargs="+", default=["outcome"])
    run.add_argument(
        "--splits",
        nargs="+",
        default=["probe_train", "probe_dev", "probe_test"],
    )
    run.add_argument("--batch-size", type=int, default=8)
    run.add_argument("--max-scenes", type=int)
    run.add_argument("--scenes-per-stratum", type=int)
    run.add_argument("--max-examples", type=int)
    run.add_argument("--generation-every", type=int, default=0)
    run.add_argument("--seed", type=int, default=20260817)
    run.add_argument("--no-resume", action="store_true")

    analyze = subparsers.add_parser("analyze", help="rebuild reports without inference")
    analyze.add_argument("--dataset", type=Path, required=True)
    analyze.add_argument("--predictions", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "analyze":
        result = analyze_behavior_run(args.dataset, args.predictions, args.output)
    else:
        config = BehaviorRunConfig(
            dataset_dir=args.dataset,
            output_dir=args.output,
            model_id=args.model,
            model_revision=args.revision,
            conditions=tuple(PromptCondition(value) for value in args.conditions),
            question_kinds=tuple(args.question_kinds),
            splits=tuple(args.splits),
            batch_size=args.batch_size,
            max_scenes=args.max_scenes,
            scenes_per_stratum=args.scenes_per_stratum,
            max_examples=args.max_examples,
            generation_every=args.generation_every,
            seed=args.seed,
            resume=not args.no_resume,
        )
        adapter = HuggingFaceMultimodalAdapter(
            args.model,
            revision=args.revision,
            dtype=args.dtype,
        )
        result = run_behavior_evaluation(config, adapter)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
