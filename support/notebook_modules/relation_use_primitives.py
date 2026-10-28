import hashlib,itertools,json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
SEMANTICS=("left","balanced","right")
def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)

def freeze(path, obj):
    if Path(path).exists():
        assert read_json(path) == obj, f'Frozen configuration changed: {path}. Use a new RUN_ID.'
    else:
        save_json(path, obj)
    return digest(obj)

def signature(stacks):
    direct = tuple(sorted((s['side'], int(s['distance']), int(s['units'])) for s in stacks))
    mirror = tuple(sorted(('right' if s == 'left' else 'left', d, w) for s, d, w in direct))
    return min(direct, mirror)

def geometries_in(obj):
    found = set()
    if isinstance(obj, dict):
        if isinstance(obj.get('stacks'), list) and obj['stacks']:
            stacks = obj['stacks']
            if all(isinstance(s, dict) and {'side', 'distance', 'units'} <= s.keys() for s in stacks):
                found.add(signature(stacks))
        for key, value in obj.items():
            if key == 'signatures' and isinstance(value, list):
                for sig in value:
                    found.add(signature([dict(side=s, distance=d, units=w) for s, d, w in sig]))
            elif key != 'stacks':
                found.update(geometries_in(value))
    elif isinstance(obj, list):
        for item in obj:
            found.update(geometries_in(item))
    return found

def scene_features(stacks):
    w = {s: sum(x['units'] for x in stacks if x['side'] == s) for s in ('left', 'right')}
    torque = {s: sum(x['units'] * x['distance'] for x in stacks if x['side'] == s)
              for s in ('left', 'right')}
    delta = torque['left'] - torque['right']
    weight = w['left'] - w['right']
    outcome = lambda x: 'left' if x > 0 else 'right' if x < 0 else 'balanced'
    return dict(left_torque=torque['left'], right_torque=torque['right'], net_torque=delta,
                weight_score=weight, outcome=outcome(delta), weight_outcome=outcome(weight))

def mappings_for(family_id):
    offset = int(digest([family_id, 'mapping'])[:8], 16) % 3
    return [tuple(SEMANTICS[(i + k + offset) % 3] for i in range(3)) for k in range(3)]

def render_images(families, image_dir):
    from visual_torque_toolkit.render import RenderStyle, render_scene
    from visual_torque_toolkit.world import Scene, Side, Stack
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    lock = image_dir.parent / 'image_manifest.json'
    if lock.exists():
        manifest = read_json(lock)
        expected = {s['scene_id'] + suffix for f in families for s in f['scenes']
                    for suffix in ('__plain.png', '__ids.png')}
        assert set(manifest) == expected
        for name, entry in manifest.items():
            assert file_hash(image_dir / name) == entry['sha256'], 'Frozen image missing/changed: ' + name
        return manifest
    style = RenderStyle()
    # Built-in scalable Pillow font travels with the pinned Pillow package.
    font = ImageFont.load_default(size=18)
    manifest = {}
    for family in families:
        for record in family['scenes']:
            scene = Scene(record['scene_id'], tuple(Stack(s['stack_id'], Side(s['side']), s['distance'], s['units'])
                                                   for s in record['stacks']))
            plain = image_dir / (record['scene_id'] + '__plain.png')
            render_scene(scene, plain, style)
            with Image.open(plain) as source:
                labeled = source.convert('RGB')
            draw = ImageDraw.Draw(labeled)
            spacing = int(style.width * .38) / style.ticks_per_side
            boxes = []
            for s in scene.stacks:
                x = round(style.width // 2 + (-1 if s.side is Side.LEFT else 1) * s.distance * spacing)
                y = style.beam_y + 25
                box = draw.textbbox((x, y), s.stack_id, font=font, anchor='mt')
                assert 0 <= box[0] < box[2] < style.width and 0 <= box[1] < box[3] < style.height
                assert all(box[2] <= b[0] or box[0] >= b[2] or box[3] <= b[1] or box[1] >= b[3] for b in boxes)
                boxes.append(box)
                draw.text((x, y), s.stack_id, font=font, anchor='mt', fill=style.ink)
            labeled.save(image_dir / (record['scene_id'] + '__ids.png'))
            for name in (plain.name, record['scene_id'] + '__ids.png'):
                manifest[name] = dict(sha256=file_hash(image_dir / name), width=style.width, height=style.height)
    save_json(lock, manifest)
    return manifest

def holm(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    previous = 0.
    for rank, i in enumerate(order):
        previous = max(previous, min(1., p[i] * (len(p) - rank)))
        adjusted[i] = previous
    return adjusted

def paired_stats(values, key, replicates=10000):
    x = np.asarray(values, dtype=float)
    assert x.ndim == 1 and len(x) and np.isfinite(x).all()
    rng = np.random.default_rng(int(digest(key)[:16], 16))
    draws = np.concatenate([rng.choice(x, (min(250, replicates-i), len(x)), replace=True).mean(axis=1)
                            for i in range(0, replicates, 250)])
    nonzero = x[x != 0]
    observed = abs(float(x.sum()))
    if len(nonzero) <= 16:
        permuted = np.array([sum(a*b for a, b in zip(signs, nonzero))
                             for signs in itertools.product((-1., 1.), repeat=len(nonzero))])
        p = float(np.mean(np.abs(permuted) >= observed - 1e-12))
    else:
        extreme = 0
        for i in range(0, replicates, 250):
            signs = rng.choice((-1., 1.), size=(min(250, replicates-i), len(nonzero)))
            extreme += int((np.abs(signs @ nonzero) >= observed - 1e-12).sum())
        p = (extreme + 1) / (replicates + 1)
    return dict(n_families=int(len(x)), mean=float(x.mean()),
                lower=float(np.quantile(draws, .025)), upper=float(np.quantile(draws, .975)), p_value=p)
