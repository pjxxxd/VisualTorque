"""Frozen, family-disjoint visual-information and wording validation."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import visual_information_support_core as base
from visual_information_support_core import (
    digest, file_hash, read_json, save_json, freeze, MODEL_SPECS, LABELS,
    SEMANTIC_TEXT, CONDITIONS, STRATA, strict_label,
)

SCHEMA='frozen-visual-validation-v1'
PHASE='frozen_validation'
N_FAMILIES=96
FROZEN_SEED=2026091803
WORDINGS=('original','neutral_question')
PRIMARY_DEFINITIONS=[
    ('weight_support_all','all',[('image_weight',1),('image',-1)],'correct'),
    ('distance_support_all','all',[('image_distance',1),('image',-1)],'correct'),
    ('full_support_all','all',[('image_full',1),('image',-1)],'correct'),
    ('joint_support_interaction','all',[('image_full',1),('image_weight',-1),('image_distance',-1),('image',1)],'correct'),
    ('distance_equal_weight','equal_weight_distance',[('image_distance',1),('image',-1)],'correct'),
    ('full_weight_conflict','weight_conflict',[('image_full',1),('image',-1)],'correct'),
    ('full_binding_pair','binding_swap',[('image_full',1),('image',-1)],'both_correct'),
]
# These four secondary comparisons address the observed wording trade-off.
WORDING_DEFINITIONS=[
    ('image_accuracy_all','all','image','correct'),
    ('image_true_balance','balanced','image','correct'),
    ('image_false_balance','nonbalanced','image','predicted_balance'),
    ('full_binding_pair','binding_swap','image_full','both_correct'),
]


def validation_prompt(scene,condition,semantics,wording):
    assert wording in WORDINGS
    prompt=base.prompt_for(scene,condition,semantics)
    if wording=='neutral_question':
        label=LABELS[semantics.index('balanced')]
        prompt=prompt.replace(f'{label}. equal',f'{label}. balanced (neither side tips)')
        prompt=prompt.replace('Question: Which way will the balance scale tip?',
                              'Question: What happens to the balance scale?')
    return prompt


def freeze_sources(run_root,source_paths,expected_versions,extra_files=()):
    """Read four named ZIPs once; resume verifies their whole-file hashes only."""
    run_root=Path(run_root)
    lock=run_root/'historical_exclusion.json'
    resolved={kind:{key:str(Path(p).resolve()) for key,p in paths.items()} for kind,paths in source_paths.items()}
    extra=[str(Path(p).resolve()) for p in extra_files]
    if lock.is_file():
        old=read_json(lock)
        assert old['source_paths']==resolved and old['extra_files']==extra,'Source list changed; do not mix validation protocols.'
        assert old['expected_versions']==expected_versions
        for path,h in old['file_hashes'].items():
            assert file_hash(path)==h,'Frozen source file changed: '+path
        return old,base.geometries_in(old)
    excluded=set();evidence={};hashes={};shared_identities={}
    for kind,paths in resolved.items():
        phase='development' if kind=='development' else 'diagnostic'
        evidence[kind]={}
        for key,path in paths.items():
            assert Path(path).is_file(),f'Missing {kind} output for {key}: {path}. Set SOURCE_ZIPS to its exact path.'
            h=file_hash(path);hashes[path]=h
            with zipfile.ZipFile(path) as z:
                manifest=json.loads(z.read('export_manifest.json'))
                assert manifest['status']=='complete' and manifest['model']==key and manifest['phase']==phase
                assert set(z.namelist())==set(manifest['files'])|{'export_manifest.json'}
                for name,value in manifest['files'].items():
                    assert hashlib.sha256(z.read(name)).hexdigest()==value,'ZIP manifest mismatch: '+name
                prefix=f'models/{key}/{phase}/'
                complete=json.loads(z.read(prefix+'completion.json'))
                runtime=json.loads(z.read(prefix+'runtime_identity.json'))
                assert complete['status']=='complete'
                assert runtime['implementation']==complete['implementation']==expected_versions[kind]
                assert runtime['model_metadata']['resolved_revision']==MODEL_SPECS[key]['revision']
                assert runtime['dtype']=='bfloat16' and runtime['thinking'] is False
                assert complete['runtime_sha256']==hashlib.sha256(z.read(prefix+'runtime_identity.json')).hexdigest()
                support_name=f'shared/{phase}/support.json'
                inputs_name=f'shared/{phase}/inputs.json'
                support=json.loads(z.read(support_name))
                support_hash=hashlib.sha256(z.read(support_name)).hexdigest()
                inputs_hash=hashlib.sha256(z.read(inputs_name)).hexdigest()
                assert runtime['support_sha256']==complete['support_sha256']==support_hash
                assert runtime['inputs_sha256']==inputs_hash
                identity=(support_hash,inputs_hash)
                if kind in shared_identities:
                    assert identity==shared_identities[kind],'The models did not share prior inputs.'
                else:
                    shared_identities[kind]=identity
                excluded.update(base.geometries_in(support))
                if kind=='development':
                    excluded.update(base.geometries_in(json.loads(z.read('historical_exclusion.json'))))
                    assert complete['n_scored']==1440
                    lineage=None
                else:
                    assert complete['n_scored']==600
                    snapshots=json.loads(z.read('shared/diagnostic/source_snapshots.json'))
                    lineage={k:v['zip_sha256'] for k,v in snapshots.items()}
                evidence[kind][key]=dict(zip_name=Path(path).name,zip_sha256=h,
                    support_sha256=support_hash,inputs_sha256=inputs_hash,runtime=runtime,
                    development_lineage=lineage)
    assert set(evidence)=={'development','diagnostic'}
    for d in evidence['diagnostic'].values():
        assert d['development_lineage']=={key:x['zip_sha256'] for key,x in evidence['development'].items()}
    for name in extra:
        path=Path(name)
        assert path.is_file(),'Extra exclusions must name a JSON or JSONL manifest: '+name
        hashes[name]=file_hash(path)
        if path.suffix=='.jsonl':
            with path.open(encoding='utf-8-sig') as stream:
                for line in stream:
                    if line.strip():
                        excluded.update(base.geometries_in(json.loads(line)))
        else:
            excluded.update(base.geometries_in(json.loads(path.read_text(encoding='utf-8-sig'))))
    snapshot=dict(schema=SCHEMA,expected_versions=expected_versions,source_paths=resolved,extra_files=extra,
        file_hashes=hashes,evidence=evidence,signatures=[[list(t) for t in s] for s in sorted(excluded)],
        scope='Known development history, development and diagnostic geometries, and explicit extra manifests only; mirrors canonicalized.')
    freeze(lock,snapshot)
    return snapshot,excluded


def make_inputs(families):
    generation_families={}
    for group in STRATA:
        fs=sorted((f for f in families if f['stratum']==group),key=lambda f:digest(['generation',f['family_id']]))
        generation_families[group]={f['family_id'] for f in fs[:2]}
    inputs=[]
    scenes={s['scene_id']:s for f in families for s in f['scenes']}
    for spec in base.trial_specs(families):
        scene=scenes[spec['scene_id']]
        audit_map=int(digest(['audit-map',spec['family_id']])[:8],16)%3
        audited=spec['family_id'] in generation_families[spec['stratum']] and spec['role']=='base' and spec['mapping_index']==audit_map
        for wording in WORDINGS:
            prompt=validation_prompt(scene,spec['condition'],spec['semantics'],wording)
            name=None if spec['condition']=='text_full' else spec['scene_id']+('__plain.png' if spec['condition']=='legacy_image' else '__ids.png')
            inputs.append(dict(spec,trial_id=spec['trial_id']+'__'+wording,parent_trial_id=spec['trial_id'],
                wording=wording,prompt=prompt,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                image_filename=name,generation_audit=audited))
    assert len(inputs)==len({x['trial_id'] for x in inputs})
    return inputs


def prepare_validation(shared_root,source_snapshot,excluded):
    shared_root=Path(shared_root)
    path=shared_root/'support.json'
    phase_hash=file_hash(shared_root/'phase_protocol.json')
    source_hash=digest(source_snapshot)
    if path.exists():
        support=read_json(path)
        assert support['source_snapshot_sha256']==source_hash and support['phase_protocol_sha256']==phase_hash
        assert support['n_families']==N_FAMILIES and support['seed']==FROZEN_SEED
        families=support['families']
    else:
        families=base.generate_families(N_FAMILIES,FROZEN_SEED,excluded,PHASE)
        support=dict(schema=SCHEMA,phase=PHASE,n_families=N_FAMILIES,seed=FROZEN_SEED,families=families,
            source_snapshot_sha256=source_hash,phase_protocol_sha256=phase_hash)
        freeze(path,support)
    assert len(families)==N_FAMILIES
    assert {s:sum(f['stratum']==s for f in families) for s in STRATA}==base.allocation(N_FAMILIES)
    base.validate_families(families,excluded)
    images=base.render_images(families,shared_root/'images')
    inputs=make_inputs(families)
    for inp in inputs:
        inp['image_sha256']=images[inp['image_filename']]['sha256'] if inp['image_filename'] else None
    assert len(inputs)==8640 and sum(x['generation_audit'] for x in inputs)==240
    freeze(shared_root/'inputs.json',inputs)
    freeze(shared_root/'generation_audit_plan.json',[x['trial_id'] for x in inputs if x['generation_audit']])
    freeze(shared_root/'geometry_checks.json',dict(n_families=N_FAMILIES,n_scene_presentations=240,
        n_mirror_canonical_geometries=len(base.geometries_in(families)),
        excluded_geometries=len(excluded),overlap_with_recorded_history=0,
        strata=base.allocation(N_FAMILIES),scope=source_snapshot['scope']))
    return families,inputs


def analyze_validation(rows,inputs,families,outdir,model_key,replicates=10000):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows)
    expected={x['trial_id']:x for x in inputs}
    assert df.trial_id.is_unique and set(df.trial_id)==set(expected),'Complete all planned inputs before final analysis.'
    for r in rows:
        inp=expected[r['trial_id']]
        for k in ('family_id','scene_id','condition','wording','gold_label','gold_semantic','mapping_index','mirror','prompt_sha256'):
            assert r[k]==inp[k]
        vals=[r['candidate_logprobs'][l] for l in LABELS]
        assert np.isfinite(vals).all()
        label=LABELS[int(np.argmax(vals))]
        assert r['predicted_label']==label and r['predicted_semantic']==r['semantics'][LABELS.index(label)]
        assert r['correct']==int(label==r['gold_label'])
        assert r['generation_audited']==inp['generation_audit']
    df.to_json(outdir/'trial_results.jsonl',orient='records',lines=True,force_ascii=False)
    for side in ('left','right','balanced'):
        df['answer_'+side]=(df.predicted_semantic==side).astype(int)
    for letter in LABELS:
        df['label_'+letter]=(df.predicted_label==letter).astype(int)
    metrics=['correct','gold_margin','predicted_balance','wrong_follows_weight','top_score_tie',
             'answer_left','answer_right','answer_balanced','label_A','label_B','label_C']
    fam=df.groupby(['family_id','stratum','condition','wording'])[metrics].mean().reset_index()
    fam.to_csv(outdir/'family_metrics.csv',index=False)
    summaries=[]
    for scope in ('all',*STRATA):
        subset=fam if scope=='all' else fam[fam.stratum==scope]
        for (condition,wording),g in subset.groupby(['condition','wording']):
            stats=base.paired_stats(g.correct,[model_key,PHASE,scope,condition,wording],replicates)
            stats.pop('p_value')
            summaries.append(dict(model=model_key,scope=scope,condition=condition,wording=wording,
                **stats,**{k:float(g[k].mean()) for k in metrics if k!='correct'}))
    summary=pd.DataFrame(summaries)
    summary.to_csv(outdir/'accuracy_and_answer_bias.csv',index=False)
    pairs=[]
    for k,g in df[df.stratum=='binding_swap'].groupby(['family_id','condition','wording','mirror','mapping_index']):
        assert len(g)==2 and set(g.role)=={'base','swapped'}
        a=g[g.role=='base'].iloc[0];b=g[g.role=='swapped'].iloc[0]
        pairs.append(dict(zip(['family_id','condition','wording','mirror','mapping_index'],k),
            both_correct=int(a.correct and b.correct),answer_changed=int(a.predicted_semantic!=b.predicted_semantic)))
    pair_df=pd.DataFrame(pairs);pair_df.to_csv(outdir/'binding_pair_trials.csv',index=False)
    pair_fam=pair_df.groupby(['family_id','condition','wording'])[['both_correct','answer_changed']].mean().reset_index()
    pair_fam.to_csv(outdir/'binding_pair_families.csv',index=False)
    effects=[]
    for wording in WORDINGS:
        for name,scope,coefficients,metric in PRIMARY_DEFINITIONS:
            source=pair_fam[pair_fam.wording==wording] if metric=='both_correct' else fam[fam.wording==wording]
            if metric!='both_correct' and scope!='all':
                source=source[source.stratum==scope]
            pivot=source.pivot(index='family_id',columns='condition',values=metric)
            delta=sum(coef*pivot[condition] for condition,coef in coefficients)
            stats=base.paired_stats(delta.to_numpy(),[model_key,PHASE,name,wording],replicates)
            row=dict(model=model_key,wording=wording,comparison=name,scope=scope,metric=metric,**stats)
            if wording!='original':
                row.pop('p_value')
            effects.append(row)
    effect_df=pd.DataFrame(effects)
    primary=effect_df[effect_df.wording=='original'].copy()
    primary['holm_within_model_7']=base.holm(primary.p_value)
    primary['inference_status']='pending_two_model_primary_14_correction'
    primary.to_csv(outdir/'primary_original_comparisons.csv',index=False)
    neutral=effect_df[effect_df.wording!='original'].drop(columns='p_value').copy()
    neutral['inference_status']='descriptive_paired_intervals_no_significance_claim'
    neutral.to_csv(outdir/'neutral_information_effects_descriptive.csv',index=False)
    secondary=[]
    for name,scope,condition,metric in WORDING_DEFINITIONS:
        source=pair_fam if metric=='both_correct' else fam
        if metric!='both_correct' and scope!='all':
            source=source[source.stratum!='balanced'] if scope=='nonbalanced' else source[source.stratum==scope]
        pivot=source[source.condition==condition].pivot(index='family_id',columns='wording',values=metric)
        stats=base.paired_stats((pivot.neutral_question-pivot.original).to_numpy(),[model_key,PHASE,'wording',name],replicates)
        secondary.append(dict(model=model_key,comparison=name,scope=scope,condition=condition,metric=metric,
                              contrast='neutral_question minus original',**stats))
    secondary=pd.DataFrame(secondary)
    secondary['holm_within_model_4']=base.holm(secondary.p_value)
    secondary['inference_status']='pending_two_model_secondary_8_correction'
    secondary.to_csv(outdir/'secondary_wording_comparisons.csv',index=False)
    # Every condition's paired repairs and harms, including denominator-zero cases.
    events=[]
    index=['family_id','scene_id','role','mirror','mapping_index']
    def event_summary(a,b,metadata):
        joined=a[index+['stratum','correct']].rename(columns={'correct':'before'}).merge(
            b[index+['correct']],on=index,validate='one_to_one').rename(columns={'correct':'after'})
        for scope in ('all',*STRATA):
            g=joined if scope=='all' else joined[joined.stratum==scope]
            errors=int((g.before==0).sum());correct=int((g.before==1).sum())
            repairs=int(((g.before==0)&(g.after==1)).sum());harms=int(((g.before==1)&(g.after==0)).sum())
            events.append(dict(**metadata,scope=scope,n_trials=len(g),repairs=repairs,harms=harms,
                original_errors=errors,original_correct=correct,
                repairs_per_original_error=repairs/errors if errors else None,
                harms_per_original_correct=harms/correct if correct else None))
    for wording in WORDINGS:
        source=df[df.wording==wording]
        for condition in CONDITIONS:
            event_summary(source[source.condition=='image'],source[source.condition==condition],
                          dict(comparison_type='information_support',condition=condition,wording=wording))
    for condition in CONDITIONS:
        source=df[df.condition==condition]
        event_summary(source[source.wording=='original'],source[source.wording=='neutral_question'],
                      dict(comparison_type='wording',condition=condition,wording='neutral_minus_original'))
    pd.DataFrame(events).to_csv(outdir/'repairs_and_harms.csv',index=False)
    balance=[]
    for (condition,wording),g in df.groupby(['condition','wording']):
        for truth,sub in [('true_balance',g[g.gold_semantic=='balanced']),('should_tip',g[g.gold_semantic!='balanced'])]:
            family=sub.groupby('family_id').predicted_balance.mean()
            balance.append(dict(condition=condition,wording=wording,truth=truth,n_families=len(family),
                family_mean_balance_response=float(family.mean()),n_trials=len(sub),balanced_responses=int(sub.predicted_balance.sum())))
    pd.DataFrame(balance).to_csv(outdir/'true_and_false_balance.csv',index=False)
    mirror=[];flip={'left':'right','right':'left','balanced':'balanced'}
    for k,g in df.groupby(['family_id','stratum','role','condition','wording','mapping_index']):
        assert len(g)==2 and set(g.mirror)=={False,True}
        a=g[g.mirror==False].iloc[0];b=g[g.mirror==True].iloc[0]
        mirror.append(dict(zip(['family_id','stratum','role','condition','wording','mapping_index'],k),
            both_correct=int(a.correct and b.correct),semantic_equivariant=int(flip[a.predicted_semantic]==b.predicted_semantic)))
    pd.DataFrame(mirror).to_csv(outdir/'mirror_pairs.csv',index=False)
    maps=[]
    for k,g in df.groupby(['family_id','stratum','scene_id','condition','wording']):
        assert len(g)==3 and g.mapping_index.nunique()==3
        maps.append(dict(zip(['family_id','stratum','scene_id','condition','wording'],k),
            semantic_invariant=int(g.predicted_semantic.nunique()==1),all_correct=int(g.correct.min())))
    pd.DataFrame(maps).to_csv(outdir/'mapping_invariance.csv',index=False)
    audits=[]
    for (condition,wording),g in df[df.generation_audited].groupby(['condition','wording']):
        valid=g.generated_label.isin(LABELS);agree=valid&(g.generated_label==g.predicted_label)
        audits.append(dict(condition=condition,wording=wording,n=len(g),valid=int(valid.sum()),agrees=int(agree.sum()),
            agreement_all=float(agree.mean()),agreement_valid=float(agree.sum()/valid.sum()) if valid.sum() else None))
    pd.DataFrame(audits).to_csv(outdir/'generation_agreement.csv',index=False)
    # Constant-side baseline under the pre-fixed strata mixture, not uniform chance.
    gold=df[(df.condition=='image')&(df.wording=='original')]
    save_json(outdir/'simple_policy_baselines.json',{side:float(gold.groupby('family_id').gold_semantic.apply(lambda x:float((x==side).mean())).mean()) for side in ('left','right','balanced')})
    return summary,primary,secondary
