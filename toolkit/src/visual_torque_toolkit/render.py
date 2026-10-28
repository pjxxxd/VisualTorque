"""A deliberately plain renderer for VisualTorque scenes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from .world import Scene, Side


@dataclass(frozen=True, slots=True)
class RenderStyle:
    width: int = 896
    height: int = 448
    ticks_per_side: int = 6
    beam_y: int = 286
    block_size: int = 30
    block_gap: int = 2
    background: tuple[int, int, int] = (248, 247, 244)
    ink: tuple[int, int, int] = (40, 43, 48)
    block_fill: tuple[int, int, int] = (83, 122, 156)


def render_scene(
    scene: Scene,
    output_path: str | Path,
    style: RenderStyle = RenderStyle(),
) -> Path:
    """Render one scene as a PNG and return its resolved output path."""

    if style.ticks_per_side <= 0:
        raise ValueError("ticks_per_side must be positive")
    max_distance = max(stack.distance for stack in scene.stacks)
    if max_distance > style.ticks_per_side:
        raise ValueError(
            f"scene uses distance {max_distance}, but style only draws "
            f"{style.ticks_per_side} ticks"
        )

    image = Image.new("RGB", (style.width, style.height), style.background)
    draw = ImageDraw.Draw(image)
    center_x = style.width // 2
    half_span = int(style.width * 0.38)
    spacing = half_span / style.ticks_per_side
    beam_left = center_x - half_span
    beam_right = center_x + half_span

    draw.line(
        (beam_left, style.beam_y, beam_right, style.beam_y),
        fill=style.ink,
        width=7,
    )
    for signed_distance in range(-style.ticks_per_side, style.ticks_per_side + 1):
        x = round(center_x + signed_distance * spacing)
        tick_height = 16 if signed_distance else 22
        draw.line(
            (x, style.beam_y - tick_height // 2, x, style.beam_y + tick_height // 2),
            fill=style.ink,
            width=3,
        )

    fulcrum_top = style.beam_y + 8
    fulcrum_bottom = style.beam_y + 104
    draw.polygon(
        (
            (center_x, fulcrum_top),
            (center_x - 66, fulcrum_bottom),
            (center_x + 66, fulcrum_bottom),
        ),
        fill=(214, 171, 79),
        outline=style.ink,
    )

    for stack in scene.stacks:
        sign = -1 if stack.side is Side.LEFT else 1
        x_center = round(center_x + sign * stack.distance * spacing)
        for level in range(stack.units):
            bottom = style.beam_y - 7 - level * (style.block_size + style.block_gap)
            top = bottom - style.block_size
            left = x_center - style.block_size // 2
            right = left + style.block_size
            draw.rectangle(
                (left, top, right, bottom),
                fill=style.block_fill,
                outline=style.ink,
                width=2,
            )

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG")
    return destination

