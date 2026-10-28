"""Frozen visual-mechanism protocols. CPU-only; no predictor fitting."""
import hashlib
import io
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from vt_frozen_primitives import (digest, file_hash, read_json, save_json, freeze,
    utc_now, center, bf16, state_hash, mapping_reference, mapping_conflict,
    predict_one, operation_key, paired_interval, holm, random_unit, native_norm_edit)

SCHEMA = 'visual-mechanism-frozen-v1'
LABELS = ('A', 'B', 'C')
MODEL_SPECS = {
    'qwen27b': dict(model_id='Qwen/Qwen3.5-27B', revision='373612cb4f9288fcf550b526d86d85febedd9801',
        expected_layers=64, content_layers=[49, 63], physical_layers=[48, 63]),
    'gemma12b': dict(model_id='google/gemma-4-12B-it', revision='707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7',
        expected_layers=48, content_layers=[37, 47], physical_layers=[37, 47]),
}
CONDITIONS = {'bridge': ('image', 'image_full'), 'visual': ('image', 'outline', 'layout')}
QUESTION = {'outcome': 'Which way will the balance scale tip?',
    'weight_more': 'Which side has more total weight?', 'weight_less': 'Which side has less total weight?'}
WORDS = {'left': 'left side', 'balanced': 'equal', 'right': 'right side'}
RULE = ('Each square is one unit of weight. Distance is measured in tick intervals from the center pivot. '
        'Torque equals weight times distance. Total torque on a side is the sum over stacks on that side.')
RENDER_RULES = dict(width=896, height=448, outline='background-filled unit squares with unchanged dark outlines',
    layout=dict(scale=.85, translation_x=35, translation_y=-12),
    interpretation='Same physical state; no crop, no added values, no outcome cue.')
RANDOM_SEEDS = (0, 1, 2, 3)


class VerifiedArchive:
    """Read only named, manifest-checked members; never recursively scan Drive."""
    def __init__(self, path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(str(self.path))
        self.zip = zipfile.ZipFile(self.path)
        names = self.zip.namelist()
        assert len(names) == len(set(names)), 'Duplicate ZIP members'
        manifests = [n for n in names if n == 'export_manifest.json' or n.endswith('/export_manifest.json')]
        assert len(manifests) == 1, 'Expected one export_manifest.json'
        self.prefix = manifests[0][:-len('export_manifest.json')]
        self.manifest = json.loads(self.zip.read(manifests[0]))
        self.hashes = {}

    def raw(self, name):
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        value = self.zip.read(self.prefix + name)
        actual = hashlib.sha256(value).hexdigest()
        assert self.manifest['files'].get(name) == actual, 'Source member checksum failed: ' + name
        self.hashes[name] = actual
        return value

    def json(self, name):
        return json.loads(self.raw(name))

    def close(self):
        self.zip.close()


def locate_source(project, model, kind, explicit=''):
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError('Explicit source does not exist: ' + str(p))
        return p
    if kind == 'prediction':
        name = f'VisualTorque_Intervention_Prediction_{model}_validation_Output.zip'
        folder = 'intervention_prediction_v1'
    else:
        name = f'VisualTorque_Frozen_Visual_Validation_{model}_Output.zip'
        folder = 'frozen_visual_validation_v1'
    candidates = [Path('/content')/name, Path(project)/'results'/folder/'main'/name, Path(project)/name]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError('Set SOURCE_' + ('PREDICTION_ZIP' if kind == 'prediction' else 'VISUAL_ZIP') +
        ' to the exact Drive ZIP path, or upload ' + name + ' to /content. No history scan is needed.')


def read_prediction_source(path, model):
    a = VerifiedArchive(path)
    try:
        prefix = f'models/{model}/'
        support = a.json('support.json'); protocol = a.json('protocol.json')
        frozen_raw = a.raw(prefix+'frozen_predictors.json'); frozen = json.loads(frozen_raw)
        runtime = a.json(prefix+'runtime_identity.json')
        complete = a.json(prefix+'validation/completion.json')
        identity = frozen['identity']
        assert identity['model_key'] == model and complete['status'] == 'heldout_validation_complete'
        assert identity['support_sha256'] == a.hashes['support.json']
        assert identity['runtime_sha256'] == a.hashes[prefix+'runtime_identity.json']
        assert identity['protocol_sha256'] == digest(protocol) == support['protocol_sha256']
        assert complete['n_families'] == 192 and complete['n_trials'] == 3456
        for k, v in MODEL_SPECS[model].items():
            assert protocol['model_specs'][model][k] == v, ('Model specification differs', k)
        families = [f for f in support['families'] if f['split'] == 'validation']
        assert Counter(f['module'] for f in families) == {'content':96, 'physical':96}
        assert len({f['family_id'] for f in families}) == 192
        for module in ('content','physical'):
            fids = {f['family_id'] for f in families if f['module'] == module}
            for layer in MODEL_SPECS[model][module+'_layers']:
                op = frozen['operations'][f'{module}:L{layer}']
                assert not fids.intersection(op['calibration_families'])
        # Retain a small authentic reference for CPU prediction parity, not for new input reuse.
        sealed = a.json(prefix+'validation/predictions_before_intervention.json')
        assert a.hashes[prefix+'validation/predictions_before_intervention.json'] == complete['prediction_sha256']
        image_manifest = a.json('image_manifest.json')
        return dict(support=support, families=families, protocol=protocol, frozen=frozen,
            frozen_bytes=frozen_raw, runtime=runtime, image_manifest=image_manifest,
            source_identity=dict(source_name=a.path.name, consumed_sha256=dict(a.hashes),
                protocol_identity=identity, scope='Existing 96 content and 96 physical validation families; new input conditions.'))
    finally:
        a.close()


def read_panel_source(path, model):
    a = VerifiedArchive(path)
    try:
        support = a.json('shared/frozen_validation/support.json')
        runtime = a.json(f'models/{model}/frozen_validation/runtime_identity.json')
        completion = a.json(f'models/{model}/frozen_validation/completion.json')
        assert support['n_families'] == 96 and len(support['families']) == 96
        assert completion['n_scored'] == 8640
        assert runtime['model_metadata']['resolved_revision'] == MODEL_SPECS[model]['revision']
        assert a.hashes['shared/frozen_validation/support.json'] == completion['support_sha256']
        assert sum(len(f['scenes']) for f in support['families']) == 240
        return dict(support=support, source_identity=dict(source_name=a.path.name, consumed_sha256=dict(a.hashes),
            scope='Reused visual-validation geometries; all conditions freshly captured under shared positional prompt.'))
    finally:
        a.close()


def ordered_stacks(role):
    return sorted(role['stacks'], key=lambda s: (s['side'] != 'left', s['distance']))


def public_table(role, full):
    lines = ['Side | Stack rank from pivot | Weight in units | Distance in tick intervals']
    for side in ('left','right'):
        stacks = [s for s in ordered_stacks(role) if s['side'] == side]
        for rank, s in enumerate(stacks,1):
            w, d = (str(s['units']),str(s['distance'])) if full else ('not provided','not provided')
            lines.append(f'{side} | {rank} | {w} | {d}')
    return '\n'.join(lines)


def question_gold(role, task):
    f = role['features']
    v = f['net_torque'] if task == 'outcome' else f['weight_score'] * (1 if task == 'weight_more' else -1)
    return 'left' if v > 0 else 'right' if v < 0 else 'balanced'


def prompt_for(role, task, mapping, condition):
    options = '\n'.join(f"{x['label']}. {WORDS[x['semantic']]}" for x in mapping)
    return '\n\n'.join(['Answer this question using the supplied balance-scale scene and any provided values.', RULE,
        'Within each side, stack ranks run from nearest to farthest from the pivot. '
        'Ranks identify objects, not weights or distances. A field marked not provided supplies no numerical value.',
        public_table(role, condition == 'image_full'), 'Question: '+QUESTION[task], 'Options:\n'+options,
        'Do not explain your reasoning. Reply with exactly one option letter and nothing else. Valid replies are: A, B, C.'])


def reading_prompt():
    return '\n\n'.join([RULE,
        'Read the weight and the distance of each of the four stacks from the image. '
        'List the left-side stacks nearest to farthest from the pivot, then the right-side stacks nearest to farthest.',
        'Return exactly a JSON array of four [weight, distance] pairs. Use integers only. '
        'Do not calculate torque or state which way the scale tips. Do not add an explanation.'])


def trial_specs(family, spec, condition):
    sm = mapping_reference(family['family_id'])
    if family['module'] == 'content':
        routes = [dict(source_before='base',source_after='aligned_source',recipient='base',task=t,tag=t)
                  for t in ('weight_more','weight_less')]
    else:
        routes = [dict(source_before=f'source_{ds}_before',source_after=f'source_{ds}_after',
            recipient=f'recipient_{dr}_before',cf_role=f'recipient_{dr}_after',task='outcome',tag=f'{ds}_{dr}',
            source_distance=family[ds],recipient_distance=family[dr]) for ds in ('near','far') for dr in ('near','far')]
    for route in routes:
        if family['module'] == 'content':
            focus = question_gold(family['roles'][route['source_after']], route['task'])
        else:
            focus = 'left' if family['dw']*(1 if family['target_side']=='left' else -1)>0 else 'right'
        mappings = [('identity',sm),('conflict_forward',mapping_conflict(sm,focus)),
                    ('conflict_reverse',mapping_conflict(sm,focus,True))]
        for name, rm in mappings:
            for layer in spec[family['module']+'_layers']:
                yield dict(route,trial_id=f"{family['family_id']}__{condition}__{route['tag']}__{name}__L{layer}",
                    condition=condition,layer=layer,source_mapping=sm,recipient_mapping=rm,mapping_name=name)


def primary_registry(stage, model):
    rows=[]; spec=MODEL_SPECS[model]
    conditions=CONDITIONS[stage] if stage=='bridge' else ('outline','layout')
    for layer in spec['content_layers']:
        selected = 'source_semantic' if model=='qwen27b' and layer==49 else 'source_letter'
        rival = 'source_letter' if selected=='source_semantic' else 'source_semantic'
        for c in conditions:
            for other in (rival,'null','confidence'):
                rows.append(dict(module='content',layer=layer,condition=c,preferred=selected,comparator=other))
    if stage=='bridge':
        for layer in spec['physical_layers']:
            for c in conditions:
                for other in ('behavior','behavior_plus_source_torque'):
                    rows.append(dict(module='physical',layer=layer,condition=c,
                        preferred='behavior_plus_weight',comparator=other))
    return rows


def protocol_for(stage, implementation):
    assert stage in CONDITIONS
    return dict(schema=SCHEMA,stage=stage,implementation=implementation,models=MODEL_SPECS,
        conditions=list(CONDITIONS[stage]),render=RENDER_RULES,seed=2026092001,
        modules=['content','physical'] if stage=='bridge' else ['content'],n_per_module=96,
        primary={m:primary_registry(stage,m) for m in MODEL_SPECS},random_seeds=list(RANDOM_SEEDS),
        random_controls='Original preselected quarter of families; four realized-magnitude-matched directions.',
        frozen='Original coefficients/scalers and unit-dose native BF16 source differences. No fitting or selection.',
        generation='Greedy cached generation, 16 new tokens; retain raw prefill logits, processed scores and output tokens.',
        measurement='First-position candidate metric and full-string generation answer are separate endpoints; never relabel wrapped output as first-position scoring.',
        statistical_unit='Physical family; group all maps, roles, mirrors, renderings. Two-sided family sign-flip and bootstrap.',
        correction='Per-stage planned tests jointly across both models. Incomplete model tests padded with p=1; each stage is a separate family.',
        secondary='Accuracy, mapping repair/harm, changed-answer subset, style error differences and reading cross-tabs are descriptive.',
        support='Previously observed geometries under new input conditions; no new-family claim.',
        reading='Separate call; not a measurement of the hidden state used by the original answer.',
        input_protocol='Same positional table scaffold; fresh image baseline, no archived-score substitution.')


def render_variant(role, condition, path):
    """Deterministic vector-like Pillow rendering; return auditable primitive bounds."""
    from PIL import Image, ImageDraw
    w,h=896,448; bg=(248,247,244); ink=(40,43,48); fill=(83,122,156)
    layout=condition=='layout'; scale=.85 if layout else 1.; tx=35 if layout else 0; ty=-12 if layout else 0
    def point(x,y): return (round(w/2+(x-w/2)*scale+tx),round(h/2+(y-h/2)*scale+ty))
    image=Image.new('RGB',(w,h),bg); draw=ImageDraw.Draw(image); bounds=[]
    def line(points,width): draw.line([point(*p) for p in points],fill=ink,width=max(1,round(width*scale)))
    cx=w//2; y=286; half=int(w*.38); spacing=half/6
    line([(cx-half,y),(cx+half,y)],7)
    for k in range(-6,7):
        tick=16 if k else 22; x=round(cx+k*spacing)
        line([(x,y-tick//2),(x,y+tick//2)],3)
    polygon=[point(cx,y+8),point(cx-66,y+104),point(cx+66,y+104)]
    draw.polygon(polygon,fill=(214,171,79),outline=ink); bounds.extend(polygon)
    for s in role['stacks']:
        assert s['side'] in ('left','right') and 1<=int(s['distance'])<=6 and 1<=int(s['units'])<=7
        x=round(cx+(-1 if s['side']=='left' else 1)*s['distance']*spacing)
        for level in range(int(s['units'])):
            bottom=y-7-level*32; a=point(x-15,bottom-30); b=point(x+15,bottom)
            bounds.extend((a,b)); draw.rectangle((*a,*b),fill=bg if condition=='outline' else fill,
                outline=ink,width=max(1,round(2*scale)))
    assert all(0<=x<w and 0<=y<h for x,y in bounds),'Clipped visual primitive'
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    raw=io.BytesIO(); image.save(raw,format='PNG'); data=raw.getvalue()
    if path.exists(): assert path.read_bytes()==data,'Frozen image differs: '+str(path)
    else: path.write_bytes(data)
    return dict(sha256=hashlib.sha256(data).hexdigest(),size=[w,h],
        pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),n_blocks=sum(s['units'] for s in role['stacks']),
        bounds=[min(p[0] for p in bounds),min(p[1] for p in bounds),max(p[0] for p in bounds),max(p[1] for p in bounds)],
        geometry_sha256=digest(role['stacks']))


def strip_completion(raw, model, completed):
    """Only known whole-turn wrappers, never search for a final answer substring."""
    if not completed: return None, 'incomplete'
    end = '<turn|>' if model=='gemma12b' else '<|im_end|>'
    text=raw.strip()
    if not text.endswith(end) or text.count(end)!=1: return None, 'unrecognized_end'
    body=text[:-len(end)].strip()
    form='direct'
    if model=='gemma12b':
        m=re.fullmatch(r'(?:thought\s*)?<channel\|>\s*(.*)',body,flags=re.S)
        structured=re.fullmatch(r'<\|channel>thought\s+([^<>]*?)<channel\|>\s*(.*)',body,flags=re.S)
        if structured: body=structured.group(2).strip(); form='closed_thought_wrapper'
        elif m: body=m.group(1).strip(); form='closing_channel_wrapper'
    else:
        m=re.fullmatch(r'<think>\s*</think>\s*(.*)',body,flags=re.S)
        if m: body=m.group(1).strip(); form='empty_think_wrapper'
    return body,form


def parse_generation(raw, model, completed, reading=False):
    body,form=strip_completion(raw,model,completed)
    result=dict(valid=False,strict_valid=False,form=form,label=None,values=None)
    if body is None: return result
    if not reading:
        if body in LABELS: result.update(valid=True,strict_valid=form=='direct',label=body)
    else:
        try: values=json.loads(body)
        except (ValueError,TypeError): return result
        if (isinstance(values,list) and len(values)==4 and all(isinstance(x,list) and len(x)==2
            and all(type(v) is int and v>=0 for v in x) for x in values)):
            result.update(valid=True,strict_valid=form=='direct',values=values)
    return result


def seal_payload(path, rows, fitted, identity):
    """All forecasts must exist before ANY target intervention in this stage."""
    predictions={}
    for r in rows:
        tid=r['trial_id']; assert tid not in predictions
        op=fitted[operation_key(r)]
        predictions[tid]={name:predict_one(r,model).tolist() for name,model in op['models'].items()}
    body=dict(identity=identity,rows_sha256=digest(rows),predictions=predictions)
    path=Path(path)
    if path.exists(): assert read_json(path)['payload']==body,'Frozen forecasts changed'
    else: save_json(path,dict(created_at_utc=utc_now(),payload=body))
    return file_hash(path)


def compare_primary(targets, sealed, stage, model, outdir, replicates=10000):
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    predictions=sealed['payload']['predictions']; records=[]
    for r in targets:
        actual=np.asarray(r['actual_delta']); post=r['post_record']; pre=r['pre_record']
        for name,pred in predictions[r['trial_id']].items():
            predicted_label=LABELS[int(np.argmax(np.asarray(r['s0'])+pred))]
            actual_label=post['predicted_label']; before=pre['predicted_label']; gen=post['generation']['parsed']['label']
            records.append(dict(model=model,stage=stage,family_id=r['family_id'],module=r['module'],layer=r['layer'],
                condition=r['condition'],mapping=r['mapping_name'],predictor=name,trial_id=r['trial_id'],
                mse=float(np.mean((actual-pred)**2)),choice_correct=predicted_label==actual_label,
                changed=actual_label!=before,generated_prediction_correct=predicted_label==gen,
                generation_valid=gen is not None,repair=before!=r['gold'] and actual_label==r['gold'],
                harm=before==r['gold'] and actual_label!=r['gold'],
                generated_repair=pre['generation']['parsed']['label'] is not None and
                    pre['generation']['parsed']['label']!=r['gold'] and gen==r['gold'],
                generated_harm=pre['generation']['parsed']['label']==r['gold'] and gen is not None and gen!=r['gold'],
                net_gain=int(actual_label==r['gold'])-int(before==r['gold']),
                pre_correct=before==r['gold'],post_correct=actual_label==r['gold'],
                pre_generated_correct=pre['generation']['parsed']['label']==r['gold'],
                post_generated_correct=gen==r['gold'],
                first_position_mass=post['generation']['first_candidate_mass'],
                generation_agreement=actual_label==gen))
    frame=pd.DataFrame(records); frame.to_csv(outdir/'trial_prediction_metrics.csv',index=False)
    keys=['model','stage','module','layer','condition','predictor']
    metrics=['mse','choice_correct','generated_prediction_correct','generation_valid','repair','harm','net_gain',
             'pre_correct','post_correct','pre_generated_correct','post_generated_correct','generation_agreement','first_position_mass']
    families=frame.groupby(keys+['family_id'])[metrics].mean().reset_index()
    families.to_csv(outdir/'prediction_family_metrics.csv',index=False)
    summary=families.groupby(keys)[metrics].mean().reset_index(); summary.to_csv(outdir/'prediction_summary.csv',index=False)
    def selected_name(module,layer):
        if module=='physical':return 'behavior_plus_weight'
        return 'source_semantic' if model=='qwen27b' and layer==49 else 'source_letter'
    selected=summary[[r.predictor==selected_name(r.module,r.layer) for r in summary.itertuples()]].copy()
    selected['generated_net_accuracy_change']=selected.post_generated_correct-selected.pre_generated_correct
    selected.to_csv(outdir/'key_results_summary.csv',index=False)
    changes=[]
    for (module,layer),sub in families.groupby(['module','layer']):
        sub=sub[sub.predictor==selected_name(module,layer)]
        anchor=sub[sub.condition=='image'].set_index('family_id')
        for condition in CONDITIONS[stage]:
            if condition=='image':continue
            new=sub[sub.condition==condition].set_index('family_id').loc[anchor.index]
            for metric in ('pre_generated_correct','mse'):
                delta=(new[metric]-anchor[metric]).to_numpy()
                stat=paired_interval(delta,[stage,model,module,int(layer),condition,metric],replicates)
                stat.pop('p_value')  # Descriptive intervals, outside the fixed primary test family.
                changes.append(dict(module=module,layer=layer,condition=condition,reference='image',metric=metric,**stat))
    pd.DataFrame(changes).to_csv(outdir/'condition_changes_descriptive.csv',index=False)
    changed=frame[frame.changed].groupby(keys).agg(n_changed=('trial_id','size'),n_prediction_correct=('choice_correct','sum'))
    changed['accuracy']=changed.n_prediction_correct/changed.n_changed
    changed.to_csv(outdir/'changed_answers_descriptive.csv')
    mapping=frame.groupby(keys+['mapping','family_id'])[['net_gain','repair','harm','generated_repair','generated_harm']].mean()
    mapping.groupby(keys+['mapping']).mean().to_csv(outdir/'mapping_gains_descriptive.csv')
    mapping_gaps=[]
    for (module,layer,condition),sub in frame.groupby(['module','layer','condition']):
        sub=sub[sub.predictor==selected_name(module,layer)]
        values=sub.groupby(['family_id','mapping']).net_gain.mean().unstack()
        gap=values['identity']-values[['conflict_forward','conflict_reverse']].mean(axis=1)
        stat=paired_interval(gap.to_numpy(),[stage,model,module,int(layer),condition,'mapping_gap'],replicates);stat.pop('p_value')
        mapping_gaps.append(dict(module=module,layer=layer,condition=condition,**stat))
    pd.DataFrame(mapping_gaps).to_csv(outdir/'mapping_gap_descriptive.csv',index=False)
    tests=[]
    for definition in primary_registry(stage,model):
        d=definition; sub=families[(families.module==d['module'])&(families.layer==d['layer'])&(families.condition==d['condition'])]
        pivot=sub.pivot(index='family_id',columns='predictor',values='mse')
        assert len(pivot)==96
        v=pivot[d['comparator']]-pivot[d['preferred']]
        tests.append(dict(model=model,stage=stage,**d,**paired_interval(v.to_numpy(),[stage,model,d],replicates)))
    tests=pd.DataFrame(tests)
    n_planned=sum(len(primary_registry(stage,m)) for m in MODEL_SPECS)
    tests['holm_p_planned_scope']=holm(list(tests.p_value)+[1.]*(n_planned-len(tests)))[:len(tests)]
    tests['correction_status']='Other model pending; missing planned tests conservatively p=1'
    tests.to_csv(outdir/'primary_comparisons.csv',index=False)
    return summary,tests


def panel_analysis(rows, reading_rows, families, outdir):
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    d=[]
    for r in rows:
        g=r['record']['generation']['parsed']; label=g['label']; sm=r['semantics']
        meaning=sm[LABELS.index(label)] if label in LABELS else None
        d.append({k:r[k] for k in ('trial_id','family_id','scene_id','role','mirror','stratum','mapping_index','condition')}|
            dict(correct=meaning==r['gold_semantic'],valid=g['valid'],semantic=meaning,
                scored_correct=r['record']['predicted_semantic']==r['gold_semantic'],
                generated_agrees=r['record']['predicted_label']==label))
    frame=pd.DataFrame(d); frame.to_csv(outdir/'panel_trials.csv',index=False)
    fm=frame.groupby(['condition','family_id','stratum'])[['correct','valid','scored_correct','generated_agrees']].mean().reset_index()
    fm.to_csv(outdir/'panel_family_metrics.csv',index=False)
    fm.groupby('condition')[['correct','valid','scored_correct','generated_agrees']].mean().to_csv(outdir/'panel_summary.csv')
    binding=[]
    for keys,g in frame[frame.stratum=='binding_swap'].groupby(['condition','family_id','mirror','mapping_index']):
        assert len(g)==2
        binding.append(dict(zip(['condition','family_id','mirror','mapping_index'],keys))|
            dict(both_correct=bool(g.correct.all()),both_valid=bool(g.valid.all()),answers_differ=g.semantic.nunique()==2))
    bf=pd.DataFrame(binding)
    bf.groupby(['condition','family_id'])[['both_correct','both_valid','answers_differ']].mean().groupby('condition').mean().to_csv(outdir/'binding_pairs.csv')
    reading=[]
    scene_truth={s['scene_id']:s for f in families for s in f['scenes']}
    for r in reading_rows:
        parsed=r['record']['generation']['parsed']; truth=ordered_stacks(scene_truth[r['scene_id']]); vals=parsed['values']
        w=bool(parsed['valid'] and all(x[0]==s['units'] for x,s in zip(vals,truth)))
        dist=bool(parsed['valid'] and all(x[1]==s['distance'] for x,s in zip(vals,truth)))
        reading.append(dict(family_id=r['family_id'],scene_id=r['scene_id'],condition=r['condition'],
            valid=parsed['valid'],weight_correct=w,distance_correct=dist,all_variables_correct=w and dist))
    rf=pd.DataFrame(reading); rf.to_csv(outdir/'reading_trials.csv',index=False)
    rf.groupby(['condition','family_id'])[['valid','weight_correct','distance_correct','all_variables_correct']].mean().groupby('condition').mean().to_csv(outdir/'reading_summary.csv')
    cross=frame.merge(rf,on=['family_id','scene_id','condition'],validate='many_to_one',suffixes=('','_reading'))
    for vars_ok in (False,True):
        for correct in (False,True):
            name=f'variables_{int(vars_ok)}__outcome_{int(correct)}'
            cross[name]=(cross.all_variables_correct==vars_ok)&(cross.correct==correct)
    cols=[c for c in cross if c.startswith('variables_')]
    cross.groupby(['condition','family_id'])[cols].mean().groupby('condition').mean().to_csv(outdir/'reading_answer_cross_table.csv')
    # A constant answer can be consistent. Always report joint correctness alongside agreement.
    anchor=frame[frame.condition=='image'].set_index(['scene_id','mapping_index'])
    comparisons=[]
    for c in ('outline','layout'):
        new=frame[frame.condition==c].set_index(['scene_id','mapping_index'])
        assert set(new.index)==set(anchor.index); new=new.loc[anchor.index]
        tmp=pd.DataFrame(dict(family_id=anchor.family_id,agreement=(new.semantic==anchor.semantic)&new.valid&anchor.valid,
            both_correct=new.correct&anchor.correct,accuracy_change=new.correct.astype(float)-anchor.correct.astype(float)))
        comparisons.append(dict(condition=c,**tmp.groupby('family_id')[['agreement','both_correct','accuracy_change']].mean().mean().to_dict()))
    pd.DataFrame(comparisons).to_csv(outdir/'style_behavior_descriptive.csv',index=False)
    return fm


def combined_tests(run_root, stage):
    root=Path(run_root); frames=[]; identities=[]
    for model in MODEL_SPECS:
        folder=root/'models'/model/stage
        if not (folder/'completion.json').exists(): continue
        complete=read_json(folder/'completion.json')
        if not complete.get('acquisition_complete'): continue
        p=folder/'analysis/primary_comparisons.csv'
        assert file_hash(p)==complete['primary_comparisons_sha256']
        d=pd.read_csv(p,keep_default_na=False)
        assert len(d)==len(primary_registry(stage,model)) and set(d.model)=={model}
        frames.append(d); identities.append((complete['protocol_sha256'],complete['support_sha256'],complete.get('panel_support_sha256')))
    if len(frames)!=2: return None
    assert len(set(identities))==1,'Different protocols; do not pool these runs'
    d=pd.concat(frames,ignore_index=True)
    d['holm_p_planned_scope']=holm(d.p_value);d['correction_status']='Both planned models complete'
    out=root/'analysis';out.mkdir(exist_ok=True)
    d.to_csv(out/f'{stage}_cross_model_primary.csv',index=False)
    return d
