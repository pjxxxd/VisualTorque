"""Pure CPU utilities embedded in the follow-up notebooks; no model execution."""
import hashlib
import json
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations, product
import numpy as np
import pandas as pd


def plain(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=plain, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=plain, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def freeze(path, identity):
    identity = json.loads(json.dumps(identity, default=plain, sort_keys=True, allow_nan=False))
    if Path(path).is_file():
        assert read_json(path) == identity, f'Configuration changed; preserve the old run and use a new RUN_ID: {path}'
    else:
        save_json(path, identity)
    return digest(identity)


def cached_json(directory, key, identity, compute):
    path = Path(directory) / (digest(str(key))[:24] + '.json')
    expected = digest(identity)
    if path.is_file():
        old = read_json(path)
        assert old['key'] == str(key) and old['identity'] == expected, f'Incompatible cache: {path}'
        return old['result']
    value = compute()
    save_json(path, dict(key=str(key), identity=expected, result=value))
    return value


def bootstrap(values, key='', replicates=10000):
    x = np.asarray(values, dtype=float)
    assert x.ndim == 1 and len(x) and np.isfinite(x).all()
    rng = np.random.default_rng(int(digest(str(key))[:16], 16))
    draws = np.concatenate([rng.choice(x, (min(250, replicates-i), len(x)), replace=True).mean(1)
                            for i in range(0, replicates, 250)])
    return dict(n=len(x), mean=float(x.mean()), lower=float(np.quantile(draws, .025)),
                upper=float(np.quantile(draws, .975)),
                p_value=float((1 + (draws <= 0).sum()) / (replicates + 1)))


def bh(values):
    p = np.asarray(values, dtype=float)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    order = np.argsort(p)
    q = np.empty_like(p)
    previous = 1.
    for rank in range(len(p)-1, -1, -1):
        i = order[rank]
        previous = min(previous, p[i]*len(p)/(rank+1))
        q[i] = previous
    return q


def paired(left, right, id_col, value):
    assert left[id_col].is_unique and right[id_col].is_unique
    assert set(left[id_col]) == set(right[id_col]), 'Different paired support'
    out = left[[id_col, value]].merge(right[[id_col, value]], on=id_col, validate='one_to_one', suffixes=('_l', '_r'))
    return out[value+'_l'].to_numpy(float) - out[value+'_r'].to_numpy(float)


def semantic_scores(record):
    if 'semantic_logprobs' in record:
        return {str(k): float(v) for k, v in record['semantic_logprobs'].items()}
    values = record['candidate_logprobs']
    if isinstance(values, dict):
        return {str(record['label_to_semantic'][k]): float(v) for k, v in values.items()}
    return dict(zip(map(str, record['candidate_semantics']), map(float, values)))


def correct_margin(record):
    values = semantic_scores(record)
    gold = str(record['gold_semantic'])
    assert gold in values and len(values) >= 2
    return values[gold] - max(v for k, v in values.items() if k != gold)


def random_unit(shape, key):
    r = np.random.default_rng(int(digest(key)[:16], 16)).normal(size=shape)
    return r / np.linalg.norm(r)


def coordinate_clamp(pre, untreated, axis, cast):
    h, base, u = [np.asarray(x, dtype=np.float64) for x in (pre, untreated, axis)]
    u = u / np.linalg.norm(u)
    a = float(np.sum((h-base)*u))
    return np.asarray(cast(h-a*u), dtype=np.float64), a


def native_norm_edit(pre, direction, norm, cast, tolerance=.02, steps=48):
    h = np.asarray(pre, dtype=np.float64)
    r = np.asarray(direction, dtype=np.float64)
    assert np.isfinite(h).all() and np.isfinite(r).all() and norm >= 0 and np.linalg.norm(r) > 0
    r /= np.linalg.norm(r)
    if norm == 0:
        return h.copy(), dict(actual_norm=0., target_norm=0., relative_error=0., matched=True)
    lo, hi = 0., max(float(norm), 1e-12)
    for _ in range(32):
        if np.linalg.norm(np.asarray(cast(h+hi*r))-h) >= norm:
            break
        hi *= 2
    best, error = h.copy(), 1.
    for _ in range(steps):
        mid = (lo+hi)/2
        candidate = np.asarray(cast(h+mid*r), dtype=np.float64)
        actual = float(np.linalg.norm(candidate-h))
        current = abs(actual-norm)/norm
        if current < error:
            best, error = candidate.copy(), current
        if error <= tolerance:
            break
        if actual < norm:
            lo = mid
        else:
            hi = mid
    return best, dict(actual_norm=float(np.linalg.norm(best-h)), target_norm=float(norm),
                      relative_error=float(error), matched=bool(error <= tolerance))


def equal_removal(pre, untreated, axis, random_direction, cast, tolerance=.02):
    target, coefficient = coordinate_clamp(pre, untreated, axis, cast)
    target_norm = float(np.linalg.norm(target-np.asarray(pre)))
    sign = 1. if coefficient >= 0 else -1.
    post, audit = native_norm_edit(pre, -sign*np.asarray(random_direction), target_norm, cast, tolerance)
    audit.update(reference_coefficient=coefficient, reference_actual_norm=target_norm)
    return post, audit


def geometry_signature(stacks):
    a = tuple(sorted((str(s['side']), int(s['distance']), int(s['units'])) for s in stacks))
    b = tuple(sorted(('right' if side == 'left' else 'left', d, u) for side, d, u in a))
    return min(a, b)  # mirrors belong to the same physical family


def geometries_in(value):
    found = set()
    if isinstance(value, dict):
        stacks = value.get('stacks')
        if isinstance(stacks, list) and stacks and all(isinstance(s, dict) and {'side','distance','units'} <= set(s) for s in stacks):
            found.add(geometry_signature(stacks))
        # Some prior matched cohorts store geometry instead of stack records.
        geom = value.get('pair_geometry')
        if isinstance(geom, (list, tuple)) and len(geom) == 4 and all(isinstance(x, (int,float)) for x in geom):
            u1,d1,u2,d2 = map(int, geom)
            found.add(geometry_signature([dict(side='left',distance=d1,units=u1),dict(side='right',distance=d2,units=u2)]))
        if isinstance(geom, (list, tuple)) and len(geom)==2 and all(isinstance(x,(list,tuple)) and len(x)==2 for x in geom):
            (u1,d1),(u2,d2)=geom
            found.add(geometry_signature([dict(side='left',distance=int(d1),units=int(u1)),dict(side='right',distance=int(d2),units=int(u2))]))
        for child in value.values():
            found.update(geometries_in(child))
    elif isinstance(value, list):
        for child in value:
            found.update(geometries_in(child))
    return found


def physics(stacks):
    units = {s:sum(int(x['units']) for x in stacks if x['side']==s) for s in ('left','right')}
    dist = {s:sum(int(x['distance']) for x in stacks if x['side']==s) for s in ('left','right')}
    torque = {s:sum(int(x['units'])*int(x['distance']) for x in stacks if x['side']==s) for s in ('left','right')}
    out = lambda x:'left' if x>0 else 'right' if x<0 else 'balanced'
    net=torque['left']-torque['right']; w=units['left']-units['right']; d=dist['left']-dist['right']
    return dict(net_torque=net,outcome=out(net),weight_score=w,distance_score=d,
                normalized_margin=abs(net)/max(1,sum(torque.values())),
                left_units=units['left'],right_units=units['right'],left_torque=torque['left'],right_torque=torque['right'],
                strategy_predictions=dict(exact_torque=out(net),weight_only=out(w),distance_only=out(d)))


def exclusion_snapshot(project, followup_root, dataset_dir=None):
    lock=Path(followup_root)/'prior_geometry_snapshot.json'
    if lock.is_file():
        old=read_json(lock)
        for path, sha in old['files'].items():
            assert file_hash(Path(path))==sha, f'Prior input changed: {path}'
        return {tuple(tuple(x) for x in sig) for sig in old['signatures']}, old
    paths={Path(dataset_dir or project/'data/dataset_v1')/'scenes.jsonl'}
    for pattern in ('*manifest*.jsonl','*support*.jsonl','*geometries*.jsonl'):
        paths.update(p for p in (project/'results').rglob(pattern) if Path(followup_root) not in p.parents)
    excluded=set(); hashes={}
    for path in sorted(paths):
        assert path.is_file(), f'Missing required geometry input: {path}'
        excluded.update(geometries_in(read_jsonl(path)))
        hashes[str(path.resolve())]=file_hash(path)
    assert excluded, 'No prior physical geometry was indexed'
    old=dict(files=hashes,signatures=sorted(excluded),mirror_canonical=True)
    save_json(lock,old)
    return excluded,old


def inverse_families(n, seed, excluded):
    assert n%4==0
    pool=defaultdict(list)
    for dl in combinations(range(1,7),2):
        for dr in combinations(range(1,7),2):
            for units in product(range(1,6), repeat=4):
                stacks=[dict(side=side,distance=d,units=u) for side,d,u in zip(('left','left','right','right'),dl+dr,units)]
                f=physics(stacks)
                pool[dl+dr].append((stacks,f))
    candidates=[]
    for distances,rows in pool.items():
        by_net=defaultdict(list)
        for stacks,f in rows:
            if f['outcome']==f['strategy_predictions']['weight_only'] and f['outcome']!='balanced':
                by_net[f['net_torque']].append((stacks,f))
        for stacks,f in rows:
            margin=f['normalized_margin']; difficulty='hard' if .08<=margin<.22 else 'moderate' if .22<=margin<.45 else None
            if (difficulty and f['outcome']!='balanced' and f['strategy_predictions']['weight_only'] not in ('balanced',f['outcome'])
                    and f['strategy_predictions']['distance_only']==f['outcome'] and abs(f['distance_score'])>=2):
                sources=by_net.get(f['net_torque'],[])
                if sources:candidates.append((digest([seed,stacks]),difficulty,stacks,f,sources))
    candidates.sort(key=lambda r:r[0]); used=set(excluded); counts=Counter(); result=[]
    for _,difficulty,stacks,f,sources in candidates:
        stratum=(difficulty,f['outcome'])
        if counts[stratum]>=n//4 or geometry_signature(stacks) in used:continue
        valid=[(s,g) for s,g in sources if geometry_signature(s) not in used|{geometry_signature(stacks)}]
        if not valid:continue
        valid.sort(key=lambda x:(sum(abs(a['units']-b['units']) for a,b in zip(stacks,x[0])),digest([seed,x[0]])))
        src,sf=valid[0];fid='fixed_'+digest([seed,stacks,src])[:20]
        roles={name:dict(scene_id=fid+'__'+name,image_filename=fid+'__'+name+'.png',stacks=s,features=g)
               for name,s,g in [('base',stacks,f),('aligned_source',src,sf)]}
        result.append(dict(family_id=fid,remapping_split='new_fixed_confirmation',roles=roles,margin_bin=difficulty))
        used.update([geometry_signature(stacks),geometry_signature(src)]);counts[stratum]+=1
        if len(result)==n:break
    assert len(result)==n and set(counts.values())=={n//4}, (len(result),counts)
    return sorted(result,key=lambda r:r['family_id'])


def repair_families(n, seed, excluded):
    assert n%4==0
    candidates=defaultdict(list)
    for ul,dl,ur,dr in product(range(1,9),range(1,7),range(1,9),range(1,7)):
        s=[dict(side='left',units=ul,distance=dl),dict(side='right',units=ur,distance=dr)]
        f=physics(s)
        if f['net_torque']==0:continue
        if ul==ur and dl!=dr:kind='equal_weight_distance'
        elif f['strategy_predictions']['weight_only'] not in ('balanced',f['outcome']):kind='weight_distance_conflict'
        elif f['strategy_predictions']['weight_only']==f['outcome'] and f['strategy_predictions']['distance_only']==f['outcome']:kind='aligned_cues'
        else:continue
        candidates[kind].append(s)
    for units in product(range(1,6),repeat=4):
        for distances in [((1,3),(2,5)),((2,5),(1,4)),((1,6),(3,5)),((2,4),(3,6))]:
            s=[dict(side=side,distance=d,units=u) for side,d,u in zip(('left','left','right','right'),distances[0]+distances[1],units)]
            if physics(s)['net_torque']!=0:candidates['multi_stack_sum'].append(s)
    used=set(excluded); result=[]
    for kind in ('equal_weight_distance','weight_distance_conflict','multi_stack_sum','aligned_cues'):
        rows=sorted(candidates[kind],key=lambda s:digest([seed,kind,s])); count=0
        for stacks in rows:
            sig=geometry_signature(stacks)
            if sig in used:continue
            fid='repair_'+digest([seed,kind,sig])[:20];scenes=[]
            for mirror in (False,True):
                current=[dict(s,side=('right' if s['side']=='left' else 'left')) if mirror else dict(s) for s in stacks]
                for style in ('standard','alternate'):
                    sid=f'{fid}__m{int(mirror)}__{style}'
                    scenes.append(dict(scene_id=sid,image_filename=sid+'.png',stacks=current,features=physics(current),style=style,mirror=mirror))
            result.append(dict(family_id=fid,stratum=kind,scenes=scenes));used.add(sig);count+=1
            if count==n//4:break
        assert count==n//4,(kind,count,'Insufficient unused geometries; do not silently change task ranges')
    return sorted(result,key=lambda r:r['family_id'])
