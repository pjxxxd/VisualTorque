"""Frozen paired tests of physical relations, answer format, and supplied intermediates.

CPU-only design/analysis. Scientific evidence comes from generated model responses,
never from these geometry generators or the unit-test fixtures.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from relation_use_primitives import (digest, file_hash, read_json, save_json, freeze,
    signature, geometries_in, scene_features, mappings_for, render_images, holm, paired_stats)

SCHEMA = 'relation-use-v1'
MODEL_SPECS = {
    'gemma12b': dict(model_id='google/gemma-4-12B-it', revision='707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7'),
    'qwen27b': dict(model_id='Qwen/Qwen3.5-27B', revision='373612cb4f9288fcf550b526d86d85febedd9801'),
}
SEMANTICS = ('left', 'balanced', 'right')
WORDS = {'left': 'left side down', 'balanced': 'balanced', 'right': 'right side down'}
LABELS = ('A', 'B', 'C')
LEVELS = ('image', 'facts', 'products', 'totals')
STRATA = ('swap_flip', 'swap_stay', 'weight_conflict', 'equal_weight_distance', 'balanced')
SEEDS = {'development': 2026101401, 'validation': 2026101402}
COUNTS = {'development': 24, 'validation': 96}
RULE = ('Each square represents one unit of weight. Distance is measured in tick intervals from the central pivot. '
        'The torque of a stack is its weight multiplied by its distance. Sum the torques on each side. '
        'The side with the larger total torque goes down; equal totals mean balanced.')
PRIMARY = (
    ('remove_letters', 'facts__semantic', 'facts__abc'),
    ('supply_weight_distance', 'facts__semantic', 'image__semantic'),
    ('supply_local_products', 'products__semantic', 'facts__semantic'),
    ('supply_side_totals', 'totals__semantic', 'products__semantic'),
    ('request_calculation', 'facts__structured', 'facts__semantic'),
)


def protocol(implementation):
    return dict(schema=SCHEMA, implementation=implementation, model_specs=MODEL_SPECS,
        seeds=SEEDS, families=COUNTS, levels=list(LEVELS), primary=[list(x) for x in PRIMARY],
        primary_scope='validation swap_flip and swap_stay; family mean both-partners-correct',
        n_primary_across_models=10, bootstrap_replicates=10000,
        generation=dict(direct_tokens=32, structured_tokens=192, component_tokens=160,
                        do_sample=False, num_beams=1, use_cache=True, enable_thinking=False),
        controls=['mirror', 'outcome-preserving swap', 'three cyclic ABC mappings',
                  'text-only facts', 'text-only totals', 'independent component tasks'],
        inference='Invalid or truncated answers count as incorrect; no accuracy-based exclusions.',
        interpretation='Intermediate-support effects locate useful supplied information, not unique neural modules.')


class Archive:
    """Consume checksummed named members without extracting paths or scanning Drive."""
    def __init__(self, path):
        self.path = Path(path)
        self.z = zipfile.ZipFile(path)
        names = self.z.namelist()
        assert len(names) == len(set(names)), 'Duplicate archive paths'
        ms = [n for n in names if n == 'export_manifest.json' or n.endswith('/export_manifest.json')]
        assert len(ms) == 1, 'Expected one export_manifest.json'
        self.prefix = ms[0][:-len('export_manifest.json')]
        self.manifest = json.loads(self.z.read(ms[0]))
        self.consumed = {}

    def raw(self, name):
        assert not name.startswith(('/', '\\')) and '..' not in name.replace('\\', '/').split('/')
        raw = self.z.read(self.prefix + name)
        sha = hashlib.sha256(raw).hexdigest()
        assert self.manifest['files'].get(name) == sha, 'Checksum mismatch: ' + name
        self.consumed[name] = sha
        return raw

    def json(self, name):
        return json.loads(self.raw(name))

    def close(self):
        self.z.close()


def find_history_source(project, explicit=''):
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(str(p))
        return p
    root = Path(project) / 'results/frozen_visual_validation_v1/main'
    # The two models used identical scene supports; either export is sufficient.
    for model in MODEL_SPECS:
        name = f'VisualTorque_Frozen_Visual_Validation_{model}_Output.zip'
        for p in (Path('/content')/name, root/name, Path(project)/name):
            if p.is_file():
                return p
    if (root/'shared/frozen_validation/support.json').is_file() and (root/'historical_exclusion.json').is_file():
        return root
    raise FileNotFoundError('Set SOURCE_VISUAL_ZIP to either Frozen_Visual_Validation output ZIP, '
                           'or restore its original results/frozen_visual_validation_v1/main directory. '
                           'No recursive Drive scan is performed.')


def read_history(source, extra_files=()):
    source = Path(source)
    names = ('historical_exclusion.json', 'shared/frozen_validation/support.json')
    values, hashes = {}, {}
    if source.is_dir():
        for name in names:
            values[name] = read_json(source/name)
            hashes[name] = file_hash(source/name)
    else:
        a = Archive(source)
        try:
            values = {name: a.json(name) for name in names}
            hashes = dict(a.consumed)
        finally:
            a.close()
    old = values[names[1]]
    assert old['n_families'] == 96 and len(old['families']) == 96
    excluded = set().union(*(geometries_in(obj) for obj in values.values()))
    for name in extra_files:
        p = Path(name)
        assert p.is_file(), 'EXTRA_HISTORY_FILES must name existing JSON/JSONL files'
        hashes['extra:'+str(p)] = file_hash(p)
        if p.suffix == '.jsonl':
            with p.open(encoding='utf-8-sig') as stream:
                for line in stream:
                    if line.strip():
                        excluded.update(geometries_in(json.loads(line)))
        else:
            excluded.update(geometries_in(read_json(p)))
    assert excluded, 'No prior geometries found'
    snapshot = dict(signatures=[[list(x) for x in sig] for sig in sorted(excluded)],
        scope='Disjoint from the listed archive snapshot, its 96 validation families, and explicit extra files only.')
    return excluded, snapshot, dict(source_name=source.name, files=hashes)


def allocation(n):
    assert n in (24, 96)
    return dict(swap_flip=n//3, swap_stay=n//3, weight_conflict=n//6,
                equal_weight_distance=n//12, balanced=n//12)


def candidate(rng, stratum):
    dl, dr = [sorted(map(int, rng.choice(range(1, 7), 2, replace=False))) for _ in range(2)]
    wl, wr = rng.integers(1, 7, size=(2, 2)).tolist()
    if stratum == 'equal_weight_distance':
        wr = wl.copy()
        if not (all(a>b for a,b in zip(dl,dr)) or all(a<b for a,b in zip(dl,dr))):
            return None
    stacks = [dict(side=side, distance=d, units=w) for side, ds, ws in
              [('left',dl,wl), ('right',dr,wr)] for d,w in zip(ds,ws)]
    f = scene_features(stacks)
    if stratum == 'balanced' and f['net_torque'] != 0:
        return None
    if stratum == 'weight_conflict' and f['net_torque'] * f['weight_score'] >= 0:
        return None
    roles = {'base': stacks}
    if stratum.startswith('swap_'):
        changed = [dict(s) for s in stacks]
        changed[0]['units'], changed[1]['units'] = changed[1]['units'], changed[0]['units']
        g = scene_features(changed)
        product = f['net_torque'] * g['net_torque']
        if wl[0] == wl[1] or (stratum == 'swap_flip' and product >= 0) or (stratum == 'swap_stay' and product <= 0):
            return None
        roles['swapped'] = changed
    return roles


def generate_families(phase, excluded, progress=print):
    rng = np.random.default_rng(SEEDS[phase])
    used = set(excluded)
    families = []
    for stratum, n in allocation(COUNTS[phase]).items():
        accepted = 0
        for attempt in range(300000):
            roles = candidate(rng, stratum)
            if roles is None:
                continue
            sigs = {signature(s) for s in roles.values()}
            if len(sigs) != len(roles) or used.intersection(sigs):
                continue
            fid = 'rel_' + digest([SCHEMA, phase, stratum, sorted(sigs)])[:16]
            ids = np.random.default_rng(int(digest([fid,'slots'])[:16],16)).permutation(4)
            scenes = []
            for role, stacks in roles.items():
                for mirror in (False, True):
                    ss = [dict(s, stack_id=f'S{int(ids[i])+1}',
                        side=('right' if s['side']=='left' else 'left') if mirror else s['side'])
                        for i,s in enumerate(stacks)]
                    scenes.append(dict(scene_id=f'{fid}__{role}__m{int(mirror)}', role=role,
                        mirror=mirror, stacks=ss, features=scene_features(ss)))
            families.append(dict(family_id=fid, phase=phase, stratum=stratum, scenes=scenes))
            used.update(sigs)
            accepted += 1
            if accepted == n:
                break
        assert accepted == n, f'Insufficient unused {stratum} families; preserve exclusions'
        progress(f'{phase}: {stratum} {n}/{n} families generated')
    families.sort(key=lambda f:digest([SEEDS[phase],f['family_id']]))
    validate_families(families, excluded)
    return families


def validate_families(families, excluded=()):
    seen = set(excluded)
    assert len({f['family_id'] for f in families}) == len(families)
    for f in families:
        original = [s for s in f['scenes'] if not s['mirror']]
        sigs = {signature(s['stacks']) for s in original}
        assert len(sigs) == len(original) and not sigs.intersection(seen)
        seen.update(sigs)
        for s in f['scenes']:
            assert scene_features(s['stacks']) == s['features']
            assert len(s['stacks']) == 4
            assert {x['stack_id'] for x in s['stacks']} == {'S1','S2','S3','S4'}
            assert len({(x['side'],x['distance']) for x in s['stacks']}) == 4
            assert all(1<=x['units']<=6 and 1<=x['distance']<=6 for x in s['stacks'])
            other = next(t for t in f['scenes'] if t['role']==s['role'] and t['mirror']!=s['mirror'])
            assert other['features']['net_torque'] == -s['features']['net_torque']
            assert {x['stack_id']:(x['units'],x['distance']) for x in s['stacks']} == {
                x['stack_id']:(x['units'],x['distance']) for x in other['stacks']}
        if f['stratum'].startswith('swap_'):
            a,b = original
            for side in ('left','right'):
                for key in ('units','distance'):
                    assert sorted(x[key] for x in a['stacks'] if x['side']==side) == sorted(
                        x[key] for x in b['stacks'] if x['side']==side)
            assert {x['stack_id']:(x['side'],x['distance']) for x in a['stacks']} == {
                x['stack_id']:(x['side'],x['distance']) for x in b['stacks']}
            sign = a['features']['net_torque'] * b['features']['net_torque']
            assert (sign<0 if f['stratum']=='swap_flip' else sign>0)
        if f['stratum']=='balanced':
            assert all(s['features']['net_torque']==0 for s in f['scenes'])
        if f['stratum']=='weight_conflict':
            assert all(s['features']['net_torque']*s['features']['weight_score']<0 for s in f['scenes'])


def make_support(snapshot, implementation, progress=print):
    excluded = geometries_in(snapshot)
    dev = generate_families('development', excluded, progress)
    val = generate_families('validation', excluded | geometries_in(dev), progress)
    return dict(schema=SCHEMA, implementation=implementation, history_sha256=digest(snapshot),
                families=dev+val, counts=COUNTS)


def trials(family):
    for scene in family['scenes']:
        for level in LEVELS:
            for i,mapping in enumerate(mappings_for(family['family_id'])):
                yield trial(family,scene,level,'abc',i,list(mapping))
            yield trial(family,scene,level,'semantic')
        yield trial(family,scene,'facts','structured')
        for level in ('facts','totals'):
            yield trial(family,scene,level,'semantic',text_only=True)
    # Independent prompts: do not feed their answers to any physical-judgment trial.
    scene = next(s for s in family['scenes'] if s['role']=='base' and not s['mirror'])
    for task in ('reading','multiply','sum_compare'):
        yield trial(family,scene,'image' if task=='reading' else 'facts','component',task=task,
                    text_only=task!='reading')


def trial(family,scene,level,mode,mapping_index=-1,mapping=None,text_only=False,task='judgment'):
    arm = ('text_' if text_only else '') + level+'__'+mode if task=='judgment' else task
    tid = scene['scene_id']+'__'+arm+(f'__m{mapping_index}' if mode=='abc' else '')
    return dict(trial_id=tid, family_id=family['family_id'],phase=family['phase'],stratum=family['stratum'],
        scene_id=scene['scene_id'],role=scene['role'],mirror=scene['mirror'],level=level,mode=mode,
        mapping_index=mapping_index,mapping=mapping,arm=arm,task=task,text_only=text_only)


def truth(scene):
    rows = sorted(scene['stacks'],key=lambda s:s['stack_id'])
    return dict(reading={s['stack_id']:[s['side'],s['units'],s['distance']] for s in rows},
        products={s['stack_id']:s['units']*s['distance'] for s in rows},
        totals={side:scene['features'][side+'_torque'] for side in ('left','right')},
        answer=scene['features']['outcome'])


def prompt_for(scene,t):
    rows = sorted(scene['stacks'],key=lambda s:s['stack_id'])
    if t['task']=='reading':
        return (RULE+'\n\nReport the side, unit weight, and distance of each labeled stack. '
            'Stack IDs identify positions. Do not predict the scale outcome. Return only a JSON object '
            'whose keys are S1, S2, S3, S4 and whose values are [side, weight, distance]. '
            'Use "left" or "right" for side and integers for the two values.')
    if t['task']=='multiply':
        pairs = '\n'.join(f"{s['stack_id']}: weight={s['units']}; distance={s['distance']}" for s in rows)
        return (RULE+'\n\n'+pairs+'\n\nCompute weight times distance for each row. '
            'Return only a JSON object with keys S1, S2, S3, S4 and integer products as values. '
            'No scale outcome is requested.')
    if t['task']=='sum_compare':
        values = '\n'.join(f"{s['stack_id']}: side={s['side']}; torque={s['units']*s['distance']}" for s in rows)
        return (RULE+'\n\n'+values+'\n\nAdd the torques on each side and determine the outcome. '
            'Return only JSON with keys "totals" and "answer". "totals" has integer keys-values '
            '"left" and "right"; "answer" is "left side down", "right side down", or "balanced".')
    lines=[]
    for s in rows:
        w = s['units'] if t['level']!='image' else 'not provided'
        d = s['distance'] if t['level']!='image' else 'not provided'
        p = s['units']*s['distance'] if t['level'] in ('products','totals') else 'not provided'
        lines.append(f"{s['stack_id']}: side={s['side']}; weight={w}; distance={d}; torque={p}")
    totals = '; '.join(f"{side}={scene['features'][side+'_torque'] if t['level']=='totals' else 'not provided'}"
                       for side in ('left','right'))
    parts=['Use the supplied scene information to answer the balance-scale question.',RULE,
        'Stack IDs identify positions, not weights. Provided values are exact. '
        'A field marked not provided supplies no numeric value.',
        'Stack information:\n'+'\n'.join(lines),'Side torque totals: '+totals,
        'Question: What happens to the balance scale?']
    if t['mode']=='abc':
        parts+=['Options:\n'+'\n'.join(f'{l}. {WORDS[s]}' for l,s in zip(LABELS,t['mapping'])),
                'Reply with exactly one option letter: A, B, or C. Do not include an explanation.']
    elif t['mode']=='semantic':
        parts+=['Reply with one short phrase describing the physical outcome. Do not use an option letter '
                'or include an explanation.']
    else:
        parts+=['Calculate explicitly before answering. Return only one JSON object with keys '
            '"products", "totals", "answer", in that order. "products" maps S1, S2, S3, S4 '
            'to integer weight-times-distance products. "totals" maps "left" and "right" to '
            'their integer sums. "answer" is "left side down", "right side down", or "balanced".']
    return '\n\n'.join(parts)


def strip_response(raw,model,completed):
    if not completed:
        return None,'truncated'
    ends = ('<turn|>','<eos>') if model=='gemma12b' else ('<|im_end|>','<|endoftext|>')
    text=raw.strip()
    end=next((s for s in ends if text.endswith(s)),None)
    if end is None or text.count(end)!=1:
        return None,'unrecognized_termination'
    body=text[:-len(end)].strip();form='direct'
    if model=='gemma12b':
        full=re.fullmatch(r'<\|channel>thought\s+([^<>]*?)<channel\|>\s*(.*)',body,re.S)
        closing=re.fullmatch(r'(?:thought\s*)?<channel\|>\s*(.*)',body,re.S)
        if full:
            body=full.group(2).strip();form='closed_thought_channel'
        elif closing:
            body=closing.group(1).strip();form='closing_channel'
    else:
        m=re.fullmatch(r'<think>\s*</think>\s*(.*)',body,re.S)
        if m:
            body=m.group(1).strip();form='empty_think'
    return body,form


def semantic(text):
    if not isinstance(text,str):
        return None
    value=text.strip().lower().rstrip('.!').strip()
    # Whole-response grammars only: never find a direction somewhere in an explanation.
    direction=re.fullmatch(r'(?:the )?(left|right) side(?: of the (?:balance )?scale)? '
        r'(?:goes down|will go down|tips down|will tip down|descends|will descend|sinks|will sink)',value)
    if direction:
        return direction.group(1)
    direction=re.fullmatch(r'(?:the )?(?:balance )?scale (?:tips|will tip|tilts|will tilt) '
        r'(?:to the )?(left|right)',value)
    if direction:
        return direction.group(1)
    values={
        'left':('left','left side','left side down','the left side goes down','left side goes down',
                'the left side will go down','left side descends','the left side descends',
                'the left side will descend','left side will descend','left side sinks','the left side sinks',
                'left side tips down','the left side tips down','the left side will tip down'),
        'right':('right','right side','right side down','the right side goes down','right side goes down',
                'the right side will go down','right side descends','the right side descends',
                'the right side will descend','right side will descend','right side sinks','the right side sinks',
                'right side tips down','the right side tips down','the right side will tip down'),
        'balanced':('balanced','balance','equal','the scale is balanced','the scale remains balanced',
                    'the balance scale is balanced','neither side goes down','neither side tips','it remains balanced'),
    }
    return next((s for s,words in values.items() if value in words),None)


def json_body(body):
    if body is None:
        return None
    m=re.fullmatch(r'```(?:json)?\s*(.*?)\s*```',body,re.S)
    if m:
        body=m.group(1)
    try:
        def unique_pairs(items):
            d={}
            for k,v in items:
                if k in d: raise ValueError('Duplicate JSON key')
                d[k]=v
            return d
        return json.loads(body,object_pairs_hook=unique_pairs)
    except (ValueError,TypeError):
        return None


def score_record(record,t,scene,model):
    body,form=strip_response(record['raw'],model,bool(record['completed']))
    obj=json_body(body);target=truth(scene);answer=None;valid=False
    component_correct=None;products_correct=None;totals_correct=None
    integer_map=lambda x,keys: isinstance(x,dict) and set(x)==set(keys) and all(type(v) is int and v>=0 for v in x.values())
    if t['task']=='judgment':
        if t['mode']=='abc' and body in LABELS:
            answer=t['mapping'][LABELS.index(body)];valid=True
        elif t['mode']=='semantic':
            answer=semantic(body);valid=answer is not None
        elif t['mode']=='structured' and isinstance(obj,dict):
            answer=semantic(obj.get('answer'));valid=answer is not None
            products_correct=integer_map(obj.get('products'),target['products']) and obj['products']==target['products']
            totals_correct=integer_map(obj.get('totals'),target['totals']) and obj['totals']==target['totals']
    elif t['task']=='reading':
        valid=isinstance(obj,dict) and set(obj)==set(target['reading']) and all(
            isinstance(v,list) and len(v)==3 and v[0] in ('left','right') and
            type(v[1]) is int and type(v[2]) is int for v in obj.values())
        component_correct=bool(valid and obj==target['reading'])
    elif t['task']=='multiply':
        valid=integer_map(obj,target['products']);component_correct=bool(valid and obj==target['products'])
    else:
        valid=isinstance(obj,dict) and integer_map(obj.get('totals'),target['totals']) and semantic(obj.get('answer')) is not None
        answer=semantic(obj.get('answer')) if isinstance(obj,dict) else None
        totals_correct=bool(valid and obj['totals']==target['totals'])
        component_correct=bool(totals_correct and answer==target['answer'])
    return dict(**t,model=model,valid=bool(valid),completed=bool(record['completed']),response_form=form,
        predicted_semantic=answer,gold=target['answer'],correct=bool(valid and answer==target['answer']),
        component_correct=component_correct,products_correct=products_correct,totals_correct=totals_correct,
        predicted_left=answer=='left',predicted_right=answer=='right',predicted_balanced=answer=='balanced',
        weight_outcome=scene['features']['weight_outcome'],
        follows_wrong_weight=answer==scene['features']['weight_outcome'] and scene['features']['weight_outcome']!=target['answer'])


def score_all(records,families,model):
    expected={t['trial_id']:t for f in families for t in trials(f)}
    scenes={s['scene_id']:s for f in families for s in f['scenes']}
    assert len(records)==len(expected) and {r['trial_id'] for r in records}==set(expected), 'Incomplete/duplicate trial records'
    rows=[]
    for r in records:
        t=expected[r['trial_id']];scene=scenes[t['scene_id']]
        assert r['prompt']==prompt_for(scene,t), 'Prompt differs from frozen definition'
        assert r['prompt_sha256']==hashlib.sha256(r['prompt'].encode()).hexdigest()
        rows.append(score_record(r,t,scene,model))
    return pd.DataFrame(rows)


def mean_interval(values,key,replicates):
    """Descriptive bootstrap only; do not enumerate unused permutation tests."""
    x=np.asarray(values,dtype=float)
    assert len(x) and np.isfinite(x).all()
    rng=np.random.default_rng(int(digest(key)[:16],16))
    draws=np.concatenate([rng.choice(x,(min(250,replicates-i),len(x)),replace=True).mean(axis=1)
                          for i in range(0,replicates,250)])
    return dict(n_families=len(x),mean=float(x.mean()),lower=float(np.quantile(draws,.025)),
                upper=float(np.quantile(draws,.975)))


def analyze(records,families,model,outdir,replicates=10000):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    df=score_all(records,families,model)
    df.to_csv(outdir/'trial_scores.csv',index=False)
    judged=df[df.task=='judgment'].copy()
    metrics=['correct','valid','completed','predicted_left','predicted_right','predicted_balanced','follows_wrong_weight']
    fam=judged.groupby(['phase','family_id','stratum','arm'])[metrics].mean().reset_index()
    fam.to_csv(outdir/'family_accuracy.csv',index=False)
    pairs=[]
    for keys,g in judged[judged.stratum.isin(['swap_flip','swap_stay'])].groupby(
            ['phase','family_id','stratum','arm','mirror','mapping_index']):
        assert len(g)==2 and set(g.role)=={'base','swapped'}
        a,b=[g[g.role==role].iloc[0] for role in ('base','swapped')]
        both_valid=bool(a.valid and b.valid)
        changed=bool(both_valid and a.predicted_semantic!=b.predicted_semantic)
        pairs.append(dict(zip(['phase','family_id','stratum','arm','mirror','mapping_index'],keys),
            both_correct=bool(a.correct and b.correct),both_valid=both_valid,
            appropriate_change=bool(both_valid and (changed if keys[2]=='swap_flip' else not changed))))
    pair=pd.DataFrame(pairs)
    pair.to_csv(outdir/'paired_trials.csv',index=False)
    pf=pair.groupby(['phase','family_id','stratum','arm'])[['both_correct','both_valid','appropriate_change']].mean().reset_index()
    pf.to_csv(outdir/'paired_families.csv',index=False)
    summaries=[]
    for frame,metric,kind in ((fam,'correct','single_accuracy'),(pf,'both_correct','pair_accuracy')):
        for phase in frame.phase.unique():
            sub=frame[frame.phase==phase]
            for scope in ['all',*sorted(sub.stratum.unique())]:
                selected=sub if scope=='all' else sub[sub.stratum==scope]
                for arm,g in selected.groupby('arm'):
                    st=mean_interval(g[metric].to_numpy(float),[model,phase,scope,arm,kind],replicates)
                    summaries.append(dict(model=model,phase=phase,scope=scope,arm=arm,metric=kind,**st))
    summary=pd.DataFrame(summaries);summary.to_csv(outdir/'summary.csv',index=False)
    tests=[]
    for phase in pf.phase.unique():
        pivot=pf[pf.phase==phase].pivot(index='family_id',columns='arm',values='both_correct')
        quality=pf[pf.phase==phase].groupby('arm').both_valid.mean()
        for name,new,old in PRIMARY:
            values=(pivot[new]-pivot[old]).to_numpy(float)
            st=paired_stats(values,[model,phase,name],replicates)
            tests.append(dict(model=model,phase=phase,test=name,new_arm=new,reference=old,
                new_pair_valid=float(quality[new]),reference_pair_valid=float(quality[old]),
                measurement_needs_review=bool(min(quality[new],quality[old])<.95),**st))
    tests=pd.DataFrame(tests)
    tests['holm_p']=np.nan
    valid=tests.phase=='validation'
    if valid.any():
        tests.loc[valid,'holm_p']=holm(list(tests.loc[valid,'p_value'])+[1.]*5)[:5]
    tests['status']=np.where(valid,'Other model pending; 10-test scope, missing tests p=1','Development descriptive only')
    tests.to_csv(outdir/'primary_comparisons.csv',index=False)
    comps=df[df.task!='judgment'].copy()
    comps.groupby(['phase','task'])[['valid','component_correct']].mean().reset_index().to_csv(outdir/'component_summary.csv',index=False)
    joined=comps.pivot(index=['phase','family_id'],columns='task',values='component_correct').astype(bool)
    joined['all_components_correct']=joined[['reading','multiply','sum_compare']].all(axis=1)
    base=judged[(judged.role=='base') & ~judged.mirror & (judged.arm=='facts__semantic')].set_index(['phase','family_id'])
    joined=joined.join(base[['correct','valid']].rename(columns={'correct':'joint_answer_correct','valid':'joint_answer_valid'}))
    joined['components_correct_joint_wrong']=joined.all_components_correct & joined.joint_answer_valid & ~joined.joint_answer_correct
    joined['components_correct_joint_invalid']=joined.all_components_correct & ~joined.joint_answer_valid
    joined.reset_index().to_csv(outdir/'component_integration.csv',index=False)
    judged[judged['mode']=='structured'][['phase','family_id','scene_id','valid','correct','products_correct','totals_correct']].to_csv(
        outdir/'structured_calculation.csv',index=False)
    judged.groupby(['phase','arm'])[['valid','completed']].mean().reset_index().to_csv(outdir/'generation_quality.csv',index=False)
    # Mirror consistency needs both answers to be valid; invalid/invalid is never a success.
    mirror=[]
    opposite={'left':'right','right':'left','balanced':'balanced'}
    for keys,g in judged.groupby(['phase','family_id','stratum','arm','role','mapping_index']):
        assert len(g)==2 and set(g.mirror)=={False,True}
        a,b=[g[g.mirror==x].iloc[0] for x in (False,True)]
        mirror.append(dict(zip(['phase','family_id','stratum','arm','role','mapping_index'],keys),
            mirror_consistent=bool(a.valid and b.valid and opposite[a.predicted_semantic]==b.predicted_semantic),
            both_correct=bool(a.correct and b.correct)))
    pd.DataFrame(mirror).to_csv(outdir/'mirror_pairs.csv',index=False)
    return summary,tests


def export_archive(root,model,output_path,images_zip=None):
    root=Path(root);output_path=Path(output_path)
    files={}
    for p in root.joinpath('shared').rglob('*'):
        if p.is_file() and not p.name.endswith('.tmp'):
            files[p.relative_to(root).as_posix()]=p
    for p in root.joinpath('models',model).rglob('*'):
        if p.is_file() and not p.name.endswith('.tmp'):
            files[p.relative_to(root).as_posix()]=p
    if images_zip is not None:
        files['shared/images.zip']=Path(images_zip)
    hashes={n:file_hash(p) for n,p in files.items()}
    manifest=dict(schema=SCHEMA,model=model,files=hashes)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    tmp=output_path.with_suffix('.tmp.zip')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for name,p in files.items(): z.write(p,name)
        z.writestr('export_manifest.json',json.dumps(manifest,sort_keys=True))
    tmp.replace(output_path)
    return output_path
