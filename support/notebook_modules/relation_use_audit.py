"""CPU recomputation of complete or partial relation-use exports."""
from __future__ import annotations
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from relation_use_core import *


def audit_one(path,outdir,replicates=10000,expected_implementation=None):
    a=Archive(path)
    try:
        model=a.manifest['model'];assert model in MODEL_SPECS
        # Check every exported member, including model records and the nested image archive.
        for name in a.manifest['files']: a.raw(name)
        config=a.json('shared/protocol.json');support=a.json('shared/support.json')
        if expected_implementation is not None:
            assert config['implementation']==expected_implementation, 'Use the CPU notebook delivered with this GPU protocol'
        snapshot=a.json('shared/historical_exclusion.json')
        assert config==protocol(config['implementation'])
        assert support['implementation']==config['implementation'] and support['history_sha256']==digest(snapshot)
        validate_families(support['families'],geometries_in(snapshot))
        assert Counter(f['phase'] for f in support['families'])==COUNTS
        runtime_name=f'models/{model}/runtime_identity.json'
        runtime=a.json(runtime_name) if runtime_name in a.manifest['files'] else None
        images=a.json('shared/image_manifest.json') if 'shared/image_manifest.json' in a.manifest['files'] else {}
        if runtime:
            assert runtime['adapter']['resolved_revision']==MODEL_SPECS[model]['revision']
            assert runtime['support_sha256']==a.consumed['shared/support.json']
            assert runtime['protocol_sha256']==a.consumed['shared/protocol.json']
            assert runtime['implementation']==config['implementation']
        reports=[];tests=[]
        for phase in COUNTS:
            prefix=f'models/{model}/{phase}/'
            names=sorted(n for n in a.manifest['files'] if n.startswith(prefix+'blocks/') and n.endswith('.json'))
            fs=[f for f in support['families'] if f['phase']==phase]
            expected={t['trial_id']:t for f in fs for t in trials(f)}
            by_scene={s['scene_id']:s for f in fs for s in f['scenes']}
            records=[];seen=set()
            for name in names:
                payload=a.json(name);rs=payload['records']
                planned=[expected[r['trial_id']] for r in rs]
                assert payload['identity']==digest([digest(runtime),phase,planned]), 'Block identity mismatch'
                for r,t in zip(rs,planned):
                    assert r['trial_id'] not in seen;seen.add(r['trial_id'])
                    assert r['runtime_sha256']==digest(runtime)
                    assert r['prompt']==prompt_for(by_scene[t['scene_id']],t)
                    assert r['completed']==bool(r['token_ids'] and r['token_ids'][-1] in r['eos_token_ids'])
                    assert len(r['token_ids'])<=r['max_new_tokens']
                    if images:
                        assert r['image_sha256']==(None if t['text_only'] else images[t['scene_id']+'__ids.png']['sha256'])
                    records.append(r)
            done_name=prefix+'completion.json'
            complete=done_name in a.manifest['files']
            if complete:
                done=a.json(done_name)
                assert done['n_trials']==len(records)==len(expected) and seen==set(expected)
                assert done['runtime_sha256']==digest(runtime)
                assert done['record_blocks']=={n.rsplit('/',1)[1]:a.consumed[n] for n in names}
                summary,ts=analyze(records,fs,model,Path(outdir)/model/phase,replicates)
                tests.append(ts)
            reports.append(dict(model=model,phase=phase,complete=complete,n_saved=len(records),n_planned=len(expected)))
        return dict(model=model,protocol_sha256=digest(config),support_sha256=digest(support),
                    reports=reports,verified_files=len(a.manifest['files'])),tests
    finally:
        a.close()


def audit_outputs(paths,outdir,replicates=10000,expected_implementation=None):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    assert paths, 'Upload at least one output ZIP or set OUTPUT_ZIPS'
    reports=[];tests=[]
    for path in paths:
        r,t=audit_one(path,outdir,replicates,expected_implementation);reports.append(r);tests.extend(t)
    assert len({r['model'] for r in reports})==len(reports), 'Duplicate model exports; choose one run per model'
    assert len({r['protocol_sha256'] for r in reports})==1 and len({r['support_sha256'] for r in reports})==1, 'Models must share the frozen experiment and scenes'
    if tests:
        test=pd.concat(tests,ignore_index=True)
        primary=test[test.phase=='validation'].copy()
        if len(primary):
            assert len(primary)<=10
            primary['holm_p']=holm(list(primary.p_value)+[1.]*(10-len(primary)))[:len(primary)]
            primary['status']='Both models complete' if len(primary)==10 else 'Partial models; missing tests p=1 in fixed 10-test scope'
            primary.to_csv(outdir/'cross_model_primary.csv',index=False)
    status=pd.DataFrame([x for r in reports for x in r['reports']]);status.to_csv(outdir/'completion_status.csv',index=False)
    save_json(outdir/'audit_report.json',dict(schema=SCHEMA,reports=reports,
        interpretation='Recomputed from raw actual generations. Partial exports are never called complete. Development results are descriptive.'))
    make_figure(outdir,reports)
    target=outdir.parent/'VisualTorque_Relation_Use_Audit_Output.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(outdir.rglob('*')):
            if p.is_file(): z.write(p,p.relative_to(outdir).as_posix())
    return status,target


def make_figure(outdir,reports):
    import matplotlib.pyplot as plt
    models=[r['model'] for r in reports if (Path(outdir)/r['model']/'validation/summary.csv').exists()]
    if not models:return
    fig,axes=plt.subplots(1,len(models),figsize=(6*len(models),4.3),squeeze=False)
    arms=['image__semantic','facts__semantic','products__semantic','totals__semantic']
    for ax,model in zip(axes.flat,models):
        df=pd.read_csv(Path(outdir)/model/'validation/summary.csv')
        for scope,color in [('swap_flip','#275F8C'),('swap_stay','#B66A33')]:
            sub=df[(df.metric=='pair_accuracy')&(df.scope==scope)].set_index('arm').loc[arms]
            y=sub['mean'].to_numpy(float)*100
            ax.errorbar(range(4),y,yerr=np.maximum(0,np.stack([sub['mean']-sub.lower,sub.upper-sub['mean']]).astype(float)*100),
                        marker='o',capsize=3,label='Outcome changes' if scope=='swap_flip' else 'Outcome stays',color=color)
        ax.set_xticks(range(4),['Image','Weights &\ndistances','Local\ntorques','Side\ntotals'])
        ax.set_ylim(0,100);ax.set_ylabel('Both paired scenes correct (%)');ax.set_title(model)
        ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(Path(outdir)/'relation_support_summary.png',dpi=180);plt.close(fig)
