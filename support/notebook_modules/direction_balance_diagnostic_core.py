"""Small, descriptive protocol diagnostic; does not modify the parent study."""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import visual_information_support_core as base
from visual_information_support_core import (
    digest, file_hash, read_json, save_json, freeze, MODEL_SPECS, LABELS,
    SEMANTIC_TEXT, strict_label,
)

SCHEMA = 'direction-balance-diagnostic-v1'
PHASE = 'diagnostic'
CONDITIONS = ('legacy_image', 'image', 'image_full', 'text_full')
WORDINGS = ('original', 'explicit_balance', 'neutral_question')
GROUPS = ('weight_direction', 'distance_direction', 'symmetric_balance', 'compensated_balance')


def build_controls():
    # Deliberate positive controls; familiar geometry is allowed, not a held-out sample.
    definitions = [
        ('weight_direction', [('left',2,3),('right',2,1)]),
        ('weight_direction', [('left',4,6),('right',4,2)]),
        ('distance_direction', [('left',4,2),('right',1,2)]),
        ('distance_direction', [('left',6,3),('right',2,3)]),
        ('symmetric_balance', [('left',2,2),('right',2,2)]),
        ('symmetric_balance', [('left',2,1),('left',4,2),('right',2,1),('right',4,2)]),
        ('compensated_balance', [('left',4,1),('right',2,2)]),
        ('compensated_balance', [('left',1,1),('left',3,2),('right',1,2),('right',5,1)]),
    ]
    families=[]
    for i,(group,tuples) in enumerate(definitions):
        fid=f'db_{i:02d}_{group}'
        ids=np.random.default_rng(int(digest([SCHEMA,fid])[:16],16)).permutation(len(tuples))+1
        scenes=[]
        for mirror in (False,True):
            stacks=[dict(side=('right' if side=='left' else 'left') if mirror else side,
                         distance=d,units=w,stack_id=f'S{ids[j]}') for j,(side,d,w) in enumerate(tuples)]
            scenes.append(dict(scene_id=f'{fid}__base__m{int(mirror)}',role='base',mirror=mirror,
                               stacks=stacks,features=base.scene_features(stacks)))
        families.append(dict(family_id=fid,stratum=group,scope='simple_controls',scenes=scenes))
    return families


def diagnostic_prompt(scene, condition, semantics, wording):
    prompt=base.prompt_for(scene,condition,semantics)
    if wording!='original':
        label=LABELS[semantics.index('balanced')]
        prompt=prompt.replace(f'{label}. equal',f'{label}. balanced (neither side tips)')
    if wording=='neutral_question':
        prompt=prompt.replace('Question: Which way will the balance scale tip?',
                              'Question: What happens to the balance scale?')
    assert wording in WORDINGS
    return prompt


def load_archives(paths, expected_implementation):
    snapshots={}
    pictures={}
    common=None
    families=None
    for key in MODEL_SPECS:
        path=Path(paths[key])
        assert path.is_file(), f'Missing prior development ZIP for {key}: {path}. Set SOURCE_ZIPS to its exact path.'
        with zipfile.ZipFile(path) as z:
            manifest=json.loads(z.read('export_manifest.json'))
            assert manifest['status']=='complete' and manifest['phase']=='development' and manifest['model']==key
            assert set(z.namelist())==set(manifest['files'])|{'export_manifest.json'}
            for name,sha in manifest['files'].items():
                assert hashlib.sha256(z.read(name)).hexdigest()==sha, 'Source ZIP hash mismatch: '+name
            support=json.loads(z.read('shared/development/support.json'))
            old_inputs=json.loads(z.read('shared/development/inputs.json'))
            selected=[f for f in support['families'] if f['stratum']=='balanced']
            assert len(selected)==4
            old_scene_ids={s['scene_id'] for f in selected for s in f['scenes']}
            root=f'models/{key}/development/'
            runtime=json.loads(z.read(root+'runtime_identity.json'))
            assert runtime['implementation']==expected_implementation, 'This diagnostic expects the reviewed information-support development protocol.'
            assert runtime['model_metadata']['resolved_revision']==MODEL_SPECS[key]['revision']
            identity={name:manifest['files'][name] for name in ('protocol.json','shared/development/support.json','shared/development/inputs.json')}
            if common is None:
                common=identity
                families=[dict(f,scope='archived_balance_replay',stratum='archived_balance') for f in selected]
            else:
                assert identity==common, 'The two development ZIPs used different shared inputs.'
            old_rows={}
            for name in z.namelist():
                if name.startswith(root+'families/') and name.endswith('.json'):
                    payload=json.loads(z.read(name))
                    assert digest(payload['rows'])==payload['rows_sha256']
                    old_rows.update({r['trial_id']:r for r in payload['rows'] if r['scene_id'] in old_scene_ids})
            chosen=[]
            for f in selected:
                # Chosen using identity only, not original outcomes or scores.
                mapping=int(digest(['replay-map',f['family_id']])[:8],16)%3
                chosen.extend(inp for inp in old_inputs if inp['family_id']==f['family_id']
                              and inp['mapping_index']==mapping and inp['condition'] in ('legacy_image','image_full','text_full'))
            assert len(chosen)==24
            for inp in chosen:
                assert inp['trial_id'] in old_rows
                if inp['image_filename']:
                    raw=z.read('shared/development/images/'+inp['image_filename'])
                    assert hashlib.sha256(raw).hexdigest()==inp['image_sha256']
                    if inp['image_filename'] in pictures:
                        assert pictures[inp['image_filename']]==raw
                    pictures[inp['image_filename']]=raw
            snapshots[key]=dict(zip_name=path.name,zip_sha256=file_hash(path),shared_identity=identity,
                runtime=runtime,inputs=chosen,rows={x['trial_id']:old_rows[x['trial_id']] for x in chosen})
    assert snapshots['gemma12b']['inputs']==snapshots['qwen27b']['inputs']
    return families,snapshots,pictures


def prepare_study(shared_root, paths, expected_implementation):
    shared_root=Path(shared_root)
    controls=build_controls()
    archived,snapshots,pictures=load_archives(paths,expected_implementation)
    freeze(shared_root/'source_snapshots.json',snapshots)
    families=controls+archived
    support=dict(schema=SCHEMA,scope='descriptive_diagnostic_only',families=families,
                 simple_families=8,archived_families=4,archive_overlap='intentional replay',
                 generalization_claim=False)
    freeze(shared_root/'support.json',support)
    image_dir=shared_root/'images'
    # The base renderer supports two- and four-stack scenes.
    # Keep its manifest in a separate folder to avoid mixing archive originals.
    control_dir=shared_root/'control_render/images'
    control_manifest=base.render_images(controls,control_dir)
    image_dir.mkdir(parents=True,exist_ok=True)
    for name in control_manifest:
        raw=(control_dir/name).read_bytes()
        target=image_dir/name
        if target.exists():
            assert target.read_bytes()==raw
        else:
            target.write_bytes(raw)
    for name,raw in pictures.items():
        target=image_dir/name
        if target.exists():
            assert target.read_bytes()==raw
        else:
            target.write_bytes(raw)
    inputs=[]
    first_per_group={group:next(f['family_id'] for f in controls if f['stratum']==group) for group in GROUPS}
    for f in controls:
        for s in f['scenes']:
            for mapping,semantics in enumerate(base.mappings_for(f['family_id'])):
                for condition in CONDITIONS:
                    for wording in WORDINGS:
                        prompt=diagnostic_prompt(s,condition,semantics,wording)
                        filename=None if condition=='text_full' else s['scene_id']+('__plain.png' if condition=='legacy_image' else '__ids.png')
                        tid=f"{s['scene_id']}__map{mapping}__{condition}__{wording}"
                        inputs.append(dict(trial_id=tid,family_id=f['family_id'],stratum=f['stratum'],scope=f['scope'],
                            scene_id=s['scene_id'],role=s['role'],mirror=s['mirror'],mapping_index=mapping,
                            semantics=list(semantics),condition=condition,wording=wording,
                            gold_semantic=s['features']['outcome'],gold_label=LABELS[semantics.index(s['features']['outcome'])],
                            weight_outcome=s['features']['weight_outcome'],prompt=prompt,
                            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),image_filename=filename,
                            image_sha256=file_hash(image_dir/filename) if filename else None,
                            generation_audit=(f['family_id']==first_per_group[f['stratum']] and mapping==0),
                            original_trial_id=None))
    for old in snapshots['gemma12b']['inputs']:
        inp=dict(old,trial_id='replay__'+old['trial_id'],original_trial_id=old['trial_id'],
                 scope='archived_balance_replay',stratum='archived_balance',wording='original',generation_audit=True)
        inputs.append(inp)
    assert len(inputs)==600 and sum(x['generation_audit'] for x in inputs)==120
    assert len({x['trial_id'] for x in inputs})==600
    freeze(shared_root/'inputs.json',inputs)
    freeze(shared_root/'image_manifest.json',{name:file_hash(image_dir/name) for name in sorted(p.name for p in image_dir.glob('*.png'))})
    return families,inputs,snapshots


def summarize_diagnostic(rows,inputs,snapshots,model_key,outdir):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows)
    expected={r['trial_id']:r for r in inputs}
    assert df.trial_id.is_unique and set(df.trial_id)==set(expected), 'Full analysis requires all 600 inputs.'
    for r in rows:
        i=expected[r['trial_id']]
        assert r['gold_label']==i['gold_label'] and r['prompt_sha256']==i['prompt_sha256']
        assert r['correct']==int(r['predicted_label']==i['gold_label'])
        assert r['generation_audited']==i['generation_audit']
    df.to_json(outdir/'trial_results.jsonl',orient='records',lines=True,force_ascii=False)
    for side in ('left','right','balanced'):
        df['answer_'+side]=(df.predicted_semantic==side).astype(int)
    for label in LABELS:
        df['label_'+label]=(df.predicted_label==label).astype(int)
    metrics=['correct','answer_left','answer_right','answer_balanced','label_A','label_B','label_C','gold_margin','top_score_tie']
    fam=df.groupby(['scope','stratum','family_id','condition','wording'])[metrics].mean().reset_index()
    fam.to_csv(outdir/'family_metrics.csv',index=False)
    summary=fam.groupby(['scope','stratum','condition','wording'])[metrics].mean().reset_index()
    counts=fam.groupby(['scope','stratum','condition','wording']).size().rename('n_families').reset_index()
    summary=summary.merge(counts)
    summary.to_csv(outdir/'diagnostic_summary.csv',index=False)
    # No overall number mixing hand-picked control groups with archive replay.
    mirrors=[]
    flip={'left':'right','right':'left','balanced':'balanced'}
    for k,g in df.groupby(['scope','stratum','family_id','condition','wording','mapping_index']):
        assert len(g)==2 and set(g.mirror)=={False,True}
        a=g[g.mirror==False].iloc[0];b=g[g.mirror==True].iloc[0]
        mirrors.append(dict(zip(['scope','stratum','family_id','condition','wording','mapping_index'],k),
             both_correct=int(a.correct and b.correct),semantic_flips_as_expected=int(flip[a.predicted_semantic]==b.predicted_semantic)))
    pd.DataFrame(mirrors).to_csv(outdir/'mirror_pairs.csv',index=False)
    maps=[]
    controls=df[df.scope=='simple_controls']
    for k,g in controls.groupby(['stratum','family_id','scene_id','condition','wording']):
        assert len(g)==3 and g.mapping_index.nunique()==3
        maps.append(dict(zip(['stratum','family_id','scene_id','condition','wording'],k),
                         semantic_invariant=int(g.predicted_semantic.nunique()==1),all_correct=int(g.correct.min())))
    pd.DataFrame(maps).to_csv(outdir/'mapping_invariance.csv',index=False)
    pair=[]
    for k,g in controls.groupby(['stratum','family_id','scene_id','condition','mapping_index']):
        by=g.set_index('wording')
        for name,a,b in [('balance_word','original','explicit_balance'),('question_wording','explicit_balance','neutral_question')]:
            pair.append(dict(zip(['stratum','family_id','scene_id','condition','mapping_index'],k),comparison=name,
                             accuracy_delta=int(by.loc[b,'correct'])-int(by.loc[a,'correct']),
                             semantic_changed=int(by.loc[a,'predicted_semantic']!=by.loc[b,'predicted_semantic'])))
    pd.DataFrame(pair).to_csv(outdir/'wording_pairs.csv',index=False)
    generated=[]
    for k,g in df[df.generation_audited].groupby(['scope','condition','wording']):
        valid=g.generated_label.isin(LABELS)
        agree=valid&(g.generated_label==g.predicted_label)
        generated.append(dict(zip(['scope','condition','wording'],k),n=len(g),valid=int(valid.sum()),
                              agrees=int(agree.sum()),agreement_all=float(agree.mean())))
    pd.DataFrame(generated).to_csv(outdir/'generation_agreement.csv',index=False)
    replays=[]
    for r in rows:
        if r['scope']!='archived_balance_replay':
            continue
        old=snapshots[model_key]['rows'][r['original_trial_id']]
        scores=np.array([r['candidate_logprobs'][x] for x in LABELS])
        before=np.array([old['candidate_logprobs'][x] for x in LABELS])
        replays.append(dict(trial_id=r['trial_id'],original_trial_id=r['original_trial_id'],condition=r['condition'],
            prompt_identical=r['prompt_sha256']==old['prompt_sha256'],
            prepared_inputs_identical=r['prepared_inputs']==old['prepared_inputs'],
            candidate_plan_identical=r['candidate_tokens']==old['candidate_tokens'] and r['candidate_prefix']==old['candidate_prefix'],
            max_centered_score_difference=float(np.max(np.abs((scores-scores.mean())-(before-before.mean())))),
            old_semantic=old['predicted_semantic'],new_semantic=r['predicted_semantic'],
            same_choice=r['predicted_label']==old['predicted_label'],old_correct=old['correct'],new_correct=r['correct']))
    pd.DataFrame(replays).to_csv(outdir/'archived_replay_comparison.csv',index=False)
    return summary
