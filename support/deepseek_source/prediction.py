"""Sections 4.2–4.3: original generators, estimators and sealed validation lifecycle."""
from pathlib import Path
import shutil
from core import load_kernels, load_model, dataset, verify_scoring_path


def setup(n, phase, source_support=None):
    root=n['ROOT']/'prediction'; model=root/'models/deepseek16b'; out=model/phase
    out.mkdir(parents=True,exist_ok=True)
    n.update(RUN_ROOT=root,MODEL_ROOT=model,PHASE_ROOT=out,PHASE=phase,
             SUPPORT_PATH=root/'support.json',IMAGE_DIR=root/'images',CAPTURE_LAYERS=(20,26))
    protocol=dict(schema='deepseek-prospective-v1',model=n['MODEL_ID'],revision=n['MODEL_REVISION'],
        implementation=n['IMPLEMENTATION_SHA256'],n_calibration=32,n_validation=96,
        layers=[20,26],dose=1.,random_fraction=.25,random_seeds=[0,1,2,3],norm_tolerance=.02,
        primary_comparisons=18, multiplicity='Holm within new DeepSeek study',
        source_support=str(source_support) if source_support else 'automatic_original_or_new',
        seed=n['SEED'],site='decoder block output, final scored prompt token',
        predictors='original ridge=1 / nonnegative scalar; five family folds on calibration only')
    n['PROTOCOL']=protocol; n['freeze'](root/'protocol.json',protocol)
    if n['SUPPORT_PATH'].exists():
        support=n['read_json'](n['SUPPORT_PATH'])
        assert support['protocol_sha256']==n['digest'](protocol)
    else:
        assert phase=='calibration','Run calibration first.'
        src=Path(source_support) if source_support else n['PROJECT_ROOT']/'results/intervention_prediction_v1/main/support.json'
        if source_support:assert src.is_file(),src
        if src.is_file():
            original=n['read_json'](src); families=original['families']
            provenance=dict(mode='reuse_original_families',path=str(src),sha256=n['file_hash'](src),
                            upstream_provenance=original.get('provenance'))
            n['IMAGE_DIR'].mkdir(parents=True,exist_ok=True)
            expected=n['read_json'](src.parent/'image_manifest.json')
            for f in families:
                for r in f['roles'].values():
                    name=r['image_filename']; p=src.parent/'images'/name
                    assert p.is_file() and n['file_hash'](p)==expected[name],p
                    shutil.copy2(p,n['IMAGE_DIR']/name)
        else:
            ds=dataset(n)
            excluded,snapshot=n['inventory_geometries'](n['PROJECT_ROOT'],root,[ds])
            families=n['generate_support'](32,96,n['SEED'],excluded)
            provenance=dict(mode='new_families_same_protocol_not_paired_with_original_models',
                            exclusions_sha256=n['file_hash'](root/'historical_geometry_snapshot.json'))
        support=dict(protocol_sha256=n['digest'](protocol),families=families,provenance=provenance)
        n['save_json'](n['SUPPORT_PATH'],support)
    fs=support['families']
    assert len({f['family_id'] for f in fs})==256
    for module in ('content','physical'):
        for split,count in [('calibration',32),('validation',96)]:
            assert len([f for f in fs if f['module']==module and f['split']==split])==count
    assert not (n['geometries_in']([f for f in fs if f['split']=='calibration']) &
                n['geometries_in']([f for f in fs if f['split']=='validation']))
    n['FAMILIES']=[f for f in fs if f['split']==phase]
    return support


def prepare(n, phase, source_support=None):
    support=setup(n,phase,source_support)
    load_model(n)
    n['render_support'](support['families'])
    n['freeze'](n['RUN_ROOT']/'image_manifest.json',
        {p.name:n['file_hash'](p) for p in sorted(n['IMAGE_DIR'].glob('*.png'))})
    n['RUNTIME_IDENTITY']=dict(n['RUNTIME_IDENTITY'],support_sha256=n['file_hash'](n['SUPPORT_PATH']),
                             image_manifest_sha256=n['file_hash'](n['RUN_ROOT']/'image_manifest.json'))
    n['freeze'](n['MODEL_ROOT']/'runtime_identity.json',n['RUNTIME_IDENTITY'])
    f=next(f for f in support['families'] if f['split']=='calibration')
    role=next(iter(f['roles']))
    ex=n['make_predictive_example'](f,role,'outcome',n['mapping_reference'](f['family_id']))
    verify_scoring_path(n,ex,n['PHASE_ROOT']/'technical_check.json')
    return support


def run(n,phase,source_support=None):
    prepare(n,phase,source_support)
    pre=n['pre_intervention_rows'](n['FAMILIES'])
    frozen_path=n['MODEL_ROOT']/'frozen_predictors.json'
    if phase=='calibration':
        rows=n['execute_targets'](n['FAMILIES'])
        fitted=n['fit_all'](rows)
        identity=dict(runtime_sha256=n['file_hash'](n['MODEL_ROOT']/'runtime_identity.json'),
            support_sha256=n['file_hash'](n['SUPPORT_PATH']),
            calibration_rows_sha256=n['file_hash'](n['PHASE_ROOT']/'target_rows.json'),
            model_key=n['MODEL_KEY'],protocol_sha256=n['digest'](n['PROTOCOL']))
        n['freeze'](frozen_path,dict(identity=identity,operations=fitted))
    else:
        frozen=n['read_json'](frozen_path); fitted=frozen['operations']
        assert frozen['identity']['runtime_sha256']==n['file_hash'](n['MODEL_ROOT']/'runtime_identity.json')
        assert frozen['identity']['support_sha256']==n['file_hash'](n['SUPPORT_PATH'])
        identity=dict(runtime_sha256=n['file_hash'](n['MODEL_ROOT']/'runtime_identity.json'),
            frozen_models_sha256=n['file_hash'](frozen_path),
            pre_manifest_sha256=n['file_hash'](n['PHASE_ROOT']/'pre_intervention_manifest.json'),
            support_sha256=n['file_hash'](n['SUPPORT_PATH']))
        sha=n['seal_predictions'](n['PHASE_ROOT']/'predictions_before_intervention.json',pre,fitted,identity)
        n['freeze'](n['PHASE_ROOT']/'prediction_seal.json',dict(identity,prediction_sha256=sha))
        rows=n['execute_targets'](n['FAMILIES'])
        sealed=n['read_json'](n['PHASE_ROOT']/'predictions_before_intervention.json')
        n['evaluate_predictions'](rows,fitted,sealed,n['PHASE_ROOT']/'analysis')
        event_metrics=n['pd'].read_csv(n['PHASE_ROOT']/'analysis/trial_prediction_metrics.csv')
        event_summary=[]
        for (operation,predictor),g in event_metrics.groupby(['operation','predictor']):
            changed=g[g.actual_change.eq(1)]
            event_summary.append(dict(operation=operation,predictor=predictor,trials=len(g),
                actual_changed=len(changed),correct_when_changed=int(changed.post_answer_match.sum()),
                accuracy_when_changed=float(changed.post_answer_match.mean()) if len(changed) else None))
        n['pd'].DataFrame(event_summary).to_csv(n['PHASE_ROOT']/'analysis/changed_choice_summary.csv',index=False)
        mapping_gains(n,rows,n['PHASE_ROOT']/'analysis')
    controls=n['analyze_controls_and_counterfactuals']()
    marker=dict(status='calibration_complete_not_heldout_evidence' if phase=='calibration' else 'heldout_validation_complete',
        n_families=len(n['FAMILIES']),n_trials=len(rows),control_status=controls)
    if phase=='validation':marker['prediction_sha256']=sha
    n['save_json'](n['PHASE_ROOT']/'completion.json',marker)
    print(marker)


def mapping_gains(n, rows, out):
    np,pd=n['np'],n['pd']; units=[]
    for r in rows:
        if r['module']!='content':continue
        before=int(np.argmax(r['s0'])); after=int(np.argmax(np.array(r['s0'])+r['actual_delta']))
        gold=n['LABELS'].index(r['gold'])
        units.append(dict(family_id=r['family_id'],layer=r['layer'],codebook=r.get('codebook','ABC'),
            mapping='matching' if r['mapping_name']=='identity' else 'conflicting',
            baseline_correct=int(before==gold),post_correct=int(after==gold),
            gain=float(after==gold)-float(before==gold),repaired=int(before!=gold and after==gold),
            harmed=int(before==gold and after!=gold)))
    df=pd.DataFrame(units); df.to_csv(out/'mapping_trial_gains.csv',index=False)
    families=df.groupby(['codebook','layer','mapping','family_id'])[['gain','repaired','harmed']].mean().reset_index()
    families.to_csv(out/'mapping_family_gains.csv',index=False)
    summaries=[]; tests=[]
    for (cb,layer,mapping),g in df.groupby(['codebook','layer','mapping']):
        x=g.groupby('family_id').gain.mean().values
        summaries.append(dict(codebook=cb,layer=int(layer),mapping=mapping,trials=len(g),
            baseline_correct=int(g.baseline_correct.sum()),post_correct=int(g.post_correct.sum()),
            baseline_accuracy=float(g.baseline_correct.mean()),post_accuracy=float(g.post_correct.mean()),
            repaired=int(g.repaired.sum()),harmed=int(g.harmed.sum()),
            **n['paired_interval'](x,['mapping',cb,int(layer),mapping])))
    for (cb,layer),g in families.groupby(['codebook','layer']):
        pivot=g.pivot(index='family_id',columns='mapping',values='gain')
        # Two-sided sign-flip p-value; gain direction is not chosen after seeing results.
        tests.append(dict(codebook=cb,layer=int(layer),contrast='matching_minus_conflicting',
            **n['paired_summary']((pivot.matching-pivot.conflicting).values,['mapping',cb,int(layer)],two_sided=True)))
    for t,p in zip(tests,n['holm']([t['p_value'] for t in tests])):t['holm_p']=float(p)
    pd.DataFrame(summaries).to_csv(out/'mapping_gain_summary.csv',index=False)
    pd.DataFrame(tests).to_csv(out/'mapping_contrasts.csv',index=False)


def formats(n,source_support=None):
    support=prepare(n,'validation',source_support)
    model=n['MODEL_ROOT']; out=model/'formats'; out.mkdir(exist_ok=True)
    assert n['read_json'](model/'validation/completion.json')['status']=='heldout_validation_complete'
    n['PHASE_ROOT']=out
    n['FAMILIES']=[f for f in support['families'] if f['split']=='validation' and f['module']=='content']
    n['FITTED']={k:v for k,v in n['read_json'](model/'frozen_predictors.json')['operations'].items() if k.startswith('content:')}
    n['freeze'](out/'runtime_identity.json',n['RUNTIME_IDENTITY'])
    n['freeze'](model/'source_identity.json',dict(frozen=n['file_hash'](model/'frozen_predictors.json'),
        support=n['file_hash'](n['SUPPORT_PATH'])))
    load_kernels(n,['frozen_format_transfer_core.py','frozen_format_transfer_gpu.py'])
    for cb in n['FORMAT_SYMBOLS']:
        ex=n['make_format_example'](n['FAMILIES'][0],'base','outcome',
                                   n['mapping_reference'](n['FAMILIES'][0]['family_id']),cb)
        n['adapter']._candidate_plan(ex)
    assert len({p[0] for p in n['adapter'].plans.values()})==1,'Format-specific scoring prefixes are not comparable.'
    from types import SimpleNamespace
    n['TAIL']=SimpleNamespace(scores=lambda hidden,cb:n['adapter'].tail_scores(hidden,n['FORMAT_SYMBOLS'][cb]))
    (out/'analysis').mkdir(exist_ok=True)
    rows=n['capture_all_formats']()
    n['freeze_format_predictions'](rows)
    rows=n['execute_format_targets']()
    sealed=n['read_json'](out/'predictions_before_intervention.json')
    n['analyze_transfer'](rows,sealed,n['FITTED'],out/'analysis',n['MODEL_KEY'])
    mapping_gains(n,rows,out/'analysis')
    n['save_json'](out/'completion.json',dict(status='heldout_format_validation_complete',trials=len(rows),
        no_refit=True,families=96,formats=list(n['FORMAT_SYMBOLS'])))
