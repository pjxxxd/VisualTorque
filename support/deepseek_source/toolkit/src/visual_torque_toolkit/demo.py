"""Write a small set of rendered examples and a JSONL manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .render import render_scene
from .world import make_demo_world


def build_demo(output_dir: str | Path) -> tuple[Path, int, int]:
    destination = Path(output_dir).expanduser().resolve()
    image_dir = destination / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    scenes, pairs = make_demo_world()
    records: list[dict[str, object]] = []
    for scene in scenes:
        image_path = render_scene(scene, image_dir / f"{scene.scene_id}.png")
        record = scene.to_record()
        record["image_path"] = str(image_path.relative_to(destination))
        records.append(record)
    records.extend(pair.to_record() for pair in pairs)

    manifest_path = destination / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return manifest_path, len(scenes), len(pairs)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("visual_torque_demo"),
        help="directory for PNG files and manifest.jsonl",
    )
    args = parser.parse_args(argv)
    manifest, scene_count, pair_count = build_demo(args.output)
    print(f"wrote {scene_count} scenes and {pair_count} pairs to {manifest}")


if __name__ == "__main__":
    main()

