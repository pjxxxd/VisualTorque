"""CPU design, rendering, prompts and family-paired analysis for information support."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

SCHEMA = 'visual-information-support-v1'
CONDITIONS = ('image', 'image_weight', 'image_distance', 'image_full', 'text_full', 'legacy_image')
STRATA = ('congruent', 'balanced', 'equal_weight_distance', 'weight_conflict', 'binding_swap')
LABELS = ('A', 'B', 'C')
SEMANTICS = ('left', 'balanced', 'right')
SEMANTIC_TEXT = {'left': 'left side', 'balanced': 'equal', 'right': 'right side'}
RULE_TEXT = ('Each square is one unit of weight. Distance is measured in tick '
             'intervals from the center pivot. Torque equals weight times distance. '
             'Total torque on a side is the sum over stacks on that side.')
REPLY_TEXT = ('Do not explain your reasoning. Reply with exactly one option letter '
              'and nothing else. Valid replies are: A, B, C.')
MODEL_SPECS = {
    'gemma12b': dict(model_id='google/gemma-4-12B-it',
                    revision='707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7'),
    'qwen27b': dict(model_id='Qwen/Qwen3.5-27B',
                   revision='373612cb4f9288fcf550b526d86d85febedd9801'),
}


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


def load_history(project, run_root, source_run='main', extra_files=(), progress=print):
    """Read only explicit old supports/snapshots and shallow dataset manifests; no recursive Drive walk."""
    project, run_root = Path(project), Path(run_root)
    lock = run_root / 'historical_exclusion.json'
    if lock.exists():
        old = read_json(lock)
        assert old['source_run'] == source_run and old['extra_files'] == [str(p) for p in extra_files]
        for name, sha in old['files'].items():
            assert file_hash(name) == sha, 'Historical input changed: ' + name
        progress(f"Reusing frozen exclusions: {len(old['signatures'])} geometries; no directory scan.")
        return geometries_in(old)
    paths = set()
    data = project / 'data'
    if data.exists():
        paths.update(data.glob('*/scenes.jsonl'))
        if (data / 'scenes.jsonl').is_file():
            paths.add(data / 'scenes.jsonl')
    results = project / 'results'
    if results.exists():
        # Only top-level experiment directories and this known run, never activation shards.
        for experiment in results.iterdir():
            if experiment.is_dir() and experiment.resolve() != run_root.parent.resolve():
                for folder in (experiment, experiment / source_run):
                    for name in ('historical_geometry_snapshot.json', 'support.json', 'scenes.jsonl'):
                        if (folder / name).is_file():
                            paths.add(folder / name)
    for name in extra_files:
        p = Path(name)
        assert p.is_file(), f'EXTRA_HISTORY_FILES must name files: {p}'
        paths.add(p)
    assert paths, 'No historical manifests found. Restore old support/snapshot files, or set EXTRA_HISTORY_FILES.'
    signatures, hashes = set(), {}
    for i, path in enumerate(sorted(paths)):
        progress(f'Read historical manifest {i+1}/{len(paths)}: {path.name}')
        h = hashlib.sha256()
        with path.open('rb') as stream:
            if path.suffix == '.jsonl':
                for line in stream:
                    h.update(line)
                    if line.strip():
                        signatures.update(geometries_in(json.loads(line.decode('utf-8-sig'))))
            else:
                raw = stream.read()
                h.update(raw)
                signatures.update(geometries_in(json.loads(raw.decode('utf-8-sig'))))
        hashes[str(path)] = h.hexdigest()
    assert signatures, 'No usable historical geometry was found in the selected manifests.'
    save_json(lock, dict(source_run=source_run, extra_files=[str(p) for p in extra_files],
                         files=hashes, signatures=[[list(t) for t in sig] for sig in sorted(signatures)],
                         scope='Listed manifests/snapshots only; unseen ZIPs or unlisted runs are not certified disjoint.'))
    return signatures


def scene_features(stacks):
    w = {s: sum(x['units'] for x in stacks if x['side'] == s) for s in ('left', 'right')}
    torque = {s: sum(x['units'] * x['distance'] for x in stacks if x['side'] == s)
              for s in ('left', 'right')}
    delta = torque['left'] - torque['right']
    weight = w['left'] - w['right']
    outcome = lambda x: 'left' if x > 0 else 'right' if x < 0 else 'balanced'
    return dict(left_torque=torque['left'], right_torque=torque['right'], net_torque=delta,
                weight_score=weight, outcome=outcome(delta), weight_outcome=outcome(weight))


def allocation(n):
    assert isinstance(n, int) and n >= 32 and n % 8 == 0
    return dict(congruent=n//8, balanced=n//8, equal_weight_distance=n//4,
                weight_conflict=n//4, binding_swap=n//4)


def _candidate(rng, stratum):
    dl, dr = [sorted(rng.choice(np.arange(1, 7), 2, replace=False).tolist()) for _ in range(2)]
    wl, wr = rng.integers(1, 7, size=(2, 2)).tolist()
    if stratum == 'equal_weight_distance':
        wr = wl.copy()
        if not (all(a > b for a, b in zip(dl, dr)) or all(a < b for a, b in zip(dl, dr))):
            return None
    stacks = [dict(side=side, distance=d, units=w) for side, distances, weights in
              [('left', dl, wl), ('right', dr, wr)] for d, w in zip(distances, weights)]
    f = scene_features(stacks)
    if stratum == 'congruent':
        distance_delta = sum(dl) - sum(dr)
        if not (f['weight_score'] * f['net_torque'] > 0 and distance_delta * f['net_torque'] > 0):
            return None
    elif stratum == 'balanced' and f['net_torque'] != 0:
        return None
    elif stratum == 'weight_conflict' and f['weight_score'] * f['net_torque'] >= 0:
        return None
    roles = {'base': stacks}
    if stratum == 'binding_swap':
        swapped = [dict(s) for s in stacks]
        swapped[0]['units'], swapped[1]['units'] = swapped[1]['units'], swapped[0]['units']
        if f['net_torque'] * scene_features(swapped)['net_torque'] >= 0:
            return None
        roles['swapped'] = swapped
    return roles


def generate_families(n, seed, excluded, phase):
    rng = np.random.default_rng(seed)
    used = set(excluded)
    families = []
    for stratum, count in allocation(n).items():
        accepted = 0
        for attempt in range(300000):
            roles = _candidate(rng, stratum)
            if roles is None:
                continue
            sigs = {signature(s) for s in roles.values()}
            if len(sigs) != len(roles) or sigs & used:
                continue
            fid = 'vis_' + digest([SCHEMA, phase, stratum, sorted(sigs)])[:16]
            slot_ids = np.random.default_rng(int(digest([fid, 'ids'])[:16], 16)).permutation(4).tolist()
            records = []
            for role, stacks in roles.items():
                for mirror in (False, True):
                    actual = [dict(s, stack_id=f'S{slot_ids[i]+1}',
                                   side=('right' if s['side'] == 'left' else 'left') if mirror else s['side'])
                              for i, s in enumerate(stacks)]
                    sid = f'{fid}__{role}__m{int(mirror)}'
                    records.append(dict(scene_id=sid, role=role, mirror=mirror,
                                        stacks=actual, features=scene_features(actual)))
            families.append(dict(family_id=fid, stratum=stratum, phase=phase, scenes=records))
            used.update(sigs)
            accepted += 1
            if accepted == count:
                break
        assert accepted == count, f'Insufficient unused {stratum} geometries: {accepted}/{count}; do not relax exclusions.'
    families.sort(key=lambda f: digest([seed, f['family_id']]))
    validate_families(families, excluded)
    return families


def validate_families(families, excluded=()):
    used = set(excluded)
    assert len({f['family_id'] for f in families}) == len(families)
    for family in families:
        base = [s for s in family['scenes'] if not s['mirror']]
        sigs = {signature(s['stacks']) for s in base}
        assert len(sigs) == len(base) and not sigs & used
        used.update(sigs)
        for scene in family['scenes']:
            assert scene_features(scene['stacks']) == scene['features']
            assert len(scene['stacks']) == 4
            assert len({(s['side'], s['distance']) for s in scene['stacks']}) == 4
            assert len({s['stack_id'] for s in scene['stacks']}) == 4
            assert all(1 <= s['units'] <= 6 and 1 <= s['distance'] <= 6 for s in scene['stacks'])
        if family['stratum'] == 'binding_swap':
            a, b = base
            assert a['features']['net_torque'] * b['features']['net_torque'] < 0
            for side in ('left', 'right'):
                aa = [s for s in a['stacks'] if s['side'] == side]
                bb = [s for s in b['stacks'] if s['side'] == side]
                assert sorted(s['units'] for s in aa) == sorted(s['units'] for s in bb)
                assert sorted(s['distance'] for s in aa) == sorted(s['distance'] for s in bb)
            assert {s['stack_id']: (s['side'], s['distance']) for s in a['stacks']} == {
                s['stack_id']: (s['side'], s['distance']) for s in b['stacks']}


def mappings_for(family_id):
    offset = int(digest([family_id, 'mapping'])[:8], 16) % 3
    return [tuple(SEMANTICS[(i + k + offset) % 3] for i in range(3)) for k in range(3)]


def public_fields(scene, condition):
    assert condition in CONDITIONS
    full = condition in ('image_full', 'text_full')
    return [dict(stack_id=s['stack_id'], side=s['side'],
                 weight=s['units'] if full or condition == 'image_weight' else 'not provided',
                 distance=s['distance'] if full or condition == 'image_distance' else 'not provided')
            for s in sorted(scene['stacks'], key=lambda s: s['stack_id'])]


def prompt_for(scene, condition, semantics):
    options = '\n'.join(f'{label}. {SEMANTIC_TEXT[meaning]}' for label, meaning in zip(LABELS, semantics))
    if condition == 'legacy_image':
        parts = ['Answer this question using only the supplied balance-scale scene.', RULE_TEXT]
    else:
        fields = public_fields(scene, condition)
        table = '\n'.join(f"{r['stack_id']}: side={r['side']}; weight={r['weight']}; distance={r['distance']}" for r in fields)
        parts = ['Answer this question using the supplied scene information.', RULE_TEXT,
                 'Stack IDs identify positions; they do not encode weight or distance. '
                 'Use any supplied image and provided values. A field marked not provided supplies no numeric value.',
                 'Provided stack information:\n' + table]
    parts += ['Question: Which way will the balance scale tip?', 'Options:\n' + options, REPLY_TEXT]
    return '\n\n'.join(parts)


def trial_specs(families):
    for family in families:
        for scene in family['scenes']:
            for mapping_index, semantics in enumerate(mappings_for(family['family_id'])):
                for condition in CONDITIONS:
                    eid = scene['scene_id'] + f'__map{mapping_index}__{condition}'
                    yield dict(trial_id=eid, family_id=family['family_id'], stratum=family['stratum'],
                               scene_id=scene['scene_id'], role=scene['role'], mirror=scene['mirror'],
                               mapping_index=mapping_index, semantics=list(semantics), condition=condition,
                               gold_semantic=scene['features']['outcome'],
                               gold_label=LABELS[semantics.index(scene['features']['outcome'])],
                               weight_outcome=scene['features']['weight_outcome'])


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


def generation_audit_ids(families):
    selected = set()
    for stratum in STRATA:
        choices = sorted((f for f in families if f['stratum'] == stratum), key=lambda f: digest(['generation', f['family_id']]))[:2]
        for f in choices:
            scene = next(s for s in f['scenes'] if s['role'] == 'base' and not s['mirror'])
            k = int(digest(['audit-map', f['family_id']])[:8], 16) % 3
            selected.update(scene['scene_id'] + f'__map{k}__{c}' for c in CONDITIONS)
    return selected


def strict_label(text):
    value = text.strip()
    return value if value in LABELS else None


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


def analyze_rows(rows, families, outdir, model_key, phase, replicates=10000):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    expected = {s['trial_id'] for s in trial_specs(families)}
    assert df.trial_id.is_unique and set(df.trial_id) == expected, 'Analysis requires every planned trial.'
    df.to_json(outdir / 'trial_results.jsonl', orient='records', lines=True, force_ascii=False)
    metrics = ['correct', 'gold_margin', 'predicted_balance', 'wrong_follows_weight']
    fam = df.groupby(['family_id', 'stratum', 'condition'])[metrics].mean().reset_index()
    fam.to_csv(outdir / 'family_metrics.csv', index=False)
    summaries = []
    for scope in ('all', *STRATA):
        subset = fam if scope == 'all' else fam[fam.stratum == scope]
        for condition, group in subset.groupby('condition'):
            row = dict(model=model_key, phase=phase, scope=scope, condition=condition,
                       **paired_stats(group.correct, [model_key, phase, scope, condition], replicates))
            row.pop('p_value')
            row.update(gold_margin=float(group.gold_margin.mean()),
                       balance_answer_rate=float(group.predicted_balance.mean()),
                       wrong_weight_response_rate=float(group.wrong_follows_weight.mean()))
            summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(outdir / 'accuracy_by_condition.csv', index=False)
    pair_rows = []
    subset = df[df.stratum == 'binding_swap']
    for keys, group in subset.groupby(['family_id', 'condition', 'mirror', 'mapping_index']):
        assert set(group.role) == {'base', 'swapped'} and len(group) == 2
        a, b = [group[group.role == role].iloc[0] for role in ('base', 'swapped')]
        pair_rows.append(dict(family_id=keys[0], condition=keys[1], mirror=keys[2], mapping_index=keys[3],
                              both_correct=int(a.correct and b.correct),
                              answer_changed=int(a.predicted_semantic != b.predicted_semantic)))
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(outdir / 'binding_pair_trials.csv', index=False)
    pair_fam = pairs.groupby(['family_id', 'condition'])[['both_correct', 'answer_changed']].mean().reset_index()
    pair_fam.to_csv(outdir / 'binding_pair_families.csv', index=False)
    definitions = [
        ('weight_support_all', 'all', [('image_weight', 1), ('image', -1)], 'correct'),
        ('distance_support_all', 'all', [('image_distance', 1), ('image', -1)], 'correct'),
        ('full_support_all', 'all', [('image_full', 1), ('image', -1)], 'correct'),
        ('joint_support_interaction', 'all', [('image_full', 1), ('image_weight', -1), ('image_distance', -1), ('image', 1)], 'correct'),
        ('distance_equal_weight', 'equal_weight_distance', [('image_distance', 1), ('image', -1)], 'correct'),
        ('full_weight_conflict', 'weight_conflict', [('image_full', 1), ('image', -1)], 'correct'),
        ('full_binding_pair', 'binding_swap', [('image_full', 1), ('image', -1)], 'both_correct'),
    ]
    comparisons = []
    for name, scope, coefficients, metric in definitions:
        source = pair_fam if metric == 'both_correct' else (fam if scope == 'all' else fam[fam.stratum == scope])
        pivot = source.pivot(index='family_id', columns='condition', values=metric)
        delta = sum(coef * pivot[condition] for condition, coef in coefficients)
        comparisons.append(dict(model=model_key, phase=phase, comparison=name, metric=metric, scope=scope,
                                **paired_stats(delta.to_numpy(), [model_key, phase, name], replicates)))
    tests = pd.DataFrame(comparisons)
    tests['holm_within_model'] = holm(tests.p_value)
    tests['inference_status'] = 'development_descriptive_only' if phase == 'development' else 'pending_joint_14_test_correction'
    tests.to_csv(outdir / 'primary_comparisons.csv', index=False)
    event_rows = []
    keys = ['family_id', 'scene_id', 'role', 'mirror', 'mapping_index']
    base = df[df.condition == 'image'][keys + ['correct']].rename(columns={'correct': 'baseline_correct'})
    for c in CONDITIONS:
        joined = df[df.condition == c].merge(base, on=keys, validate='one_to_one')
        joined['repaired'] = (joined.baseline_correct == 0) & (joined.correct == 1)
        joined['harmed'] = (joined.baseline_correct == 1) & (joined.correct == 0)
        joined['original_error'] = joined.baseline_correct == 0
        joined['original_correct'] = joined.baseline_correct == 1
        for scope in ('all', *STRATA):
            g = joined if scope == 'all' else joined[joined.stratum == scope]
            event_rows.append(dict(condition=c, scope=scope, n_trials=len(g),
                                   repairs=int(g.repaired.sum()), harms=int(g.harmed.sum()),
                                   original_errors=int(g.original_error.sum()), original_correct=int(g.original_correct.sum()),
                                   repairs_per_original_error=float(g.repaired.sum()/g.original_error.sum()) if g.original_error.sum() else None,
                                   harms_per_original_correct=float(g.harmed.sum()/g.original_correct.sum()) if g.original_correct.sum() else None))
    pd.DataFrame(event_rows).to_csv(outdir / 'repair_harm_counts.csv', index=False)
    audits = df[df.generation_audited == True]
    audit_summary = []
    for condition, group in audits.groupby('condition'):
        valid = group.generated_label.isin(LABELS)
        agreed = valid & (group.generated_label == group.predicted_label)
        audit_summary.append(dict(condition=condition, n=len(group), valid=int(valid.sum()),
                                  agrees=int(agreed.sum()), agreement_all=float(agreed.mean()),
                                  agreement_valid=float(agreed.sum()/valid.sum()) if valid.sum() else None))
    pd.DataFrame(audit_summary).to_csv(outdir / 'generation_scoring_agreement.csv', index=False)
    return summary, tests
