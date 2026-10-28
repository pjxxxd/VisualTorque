"""Command-line entry point for VisualTorque dataset tools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .dataset import build_dataset, validate_dataset
from .demo import build_demo
from .generate import GenerationConfig
from .splits import SplitConfig


def _generation_from_args(args: argparse.Namespace) -> GenerationConfig:
    return GenerationConfig(
        max_units=args.max_units,
        max_distance=args.max_distance,
        multi_max_units=args.multi_max_units,
        multi_scene_count=args.multi_scenes,
        seed=args.seed,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="visual-torque")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="build and validate Dataset v1")
    generate.add_argument("--output", type=Path, default=Path("data/dataset_v1"))
    generate.add_argument("--max-units", type=int, default=5)
    generate.add_argument("--max-distance", type=int, default=6)
    generate.add_argument("--multi-max-units", type=int, default=4)
    generate.add_argument("--multi-scenes", type=int, default=1200)
    generate.add_argument("--seed", type=int, default=20260817)
    generate.add_argument("--no-render", action="store_true")

    validate = subparsers.add_parser("validate", help="validate an existing dataset")
    validate.add_argument("dataset", type=Path)
    validate.add_argument("--no-images", action="store_true")

    demo = subparsers.add_parser("demo", help="write the original hand-authored demo")
    demo.add_argument("--output", type=Path, default=Path("visual_torque_demo"))

    args = parser.parse_args(argv)
    if args.command == "generate":
        generation = _generation_from_args(args)
        summary = build_dataset(
            args.output,
            generation,
            SplitConfig(seed=args.seed),
            render_images=not args.no_render,
        )
        result = {
            "output_dir": str(summary.output_dir),
            "scenes": summary.scene_count,
            "pairs": summary.pair_count,
            "questions": summary.question_count,
            "images": summary.image_count,
        }
    elif args.command == "validate":
        result = validate_dataset(args.dataset, verify_images=not args.no_images)
    else:
        manifest, scene_count, pair_count = build_demo(args.output)
        result = {"manifest": str(manifest), "scenes": scene_count, "pairs": pair_count}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
