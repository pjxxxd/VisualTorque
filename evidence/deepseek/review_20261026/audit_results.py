"""Audit the supplied Notebook 01 archive without executing any code from it."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
PROJECT=OUT.parents[2]
B='prediction/models/deepseek16b/'

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,allow_nan=False).encode()).hexdigest()

def holm(values):
    p=np.asarray(values,float);order=np.argsort(p);q=np.empty_like(p);previous=0.
    for rank,i in enumerate(order):
        previous=max(previous,min(1.,p[i]*(len(p)-rank)));q[i]=previous
    return q

def canonical(stacks):
    direct=tuple(sorted((x['side'],x['distance'],x['units']) for x in stacks))
    mirror=tuple(sorted(('right' if s=='left' else 'left',d,u) for s,d,u in direct))
    return min(direct,mirror)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('archive',type=Path)
    args=parser.parse_args();result={}
    with zipfile.ZipFile(args.archive) as z:
        names=set(z.namelist())
        def obj(name):return json.loads(z.read(name))
        def sha(name):return hashlib.sha256(z.read(name)).hexdigest()
        def table(name):return pd.read_csv(io.BytesIO(z.read(name)),keep_default_na=False,low_memory=False)
        assert z.testzip() is None
        assert not [n for n in names if Path(n).name.startswith('failure_')]
        result['archive']=dict(name=args.archive.name,sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest(),
                               files=len(names),crc_passed=True)
        result['completion']=obj('summary.json')
        result['runtime']=obj('runtime_identity.json')
        result['gpu_check']=obj('logs/gpu_check.json')
        support=obj('prediction/support.json');families=support['families']
        counts=Counter((f['module'],f['split']) for f in families)
        assert counts=={('content','calibration'):32,('physical','calibration'):32,
                       ('content','validation'):96,('physical','validation'):96}
        geometries={split:{canonical(r['stacks']) for f in families if f['split']==split
                           for r in f['roles'].values()} for split in ('calibration','validation')}
        assert not geometries['calibration'] & geometries['validation']
        result['family_counts']={f'{k[0]}:{k[1]}':v for k,v in counts.items()}
        result['calibration_validation_geometry_overlap']=0
        result['support_provenance']=support['provenance']
        images=obj('prediction/image_manifest.json')
        for name,expected in images.items():assert sha('prediction/images/'+name)==expected
        result['image_hashes_verified']=len(images)
        protocol=obj('prediction/protocol.json');assert digest(protocol)==support['protocol_sha256']
        frozen=obj(B+'frozen_predictors.json')
        assert frozen['identity']['calibration_rows_sha256']==sha(B+'calibration/target_rows.json')
        assert frozen['identity']['protocol_sha256']==digest(protocol)
        assert frozen['identity']['support_sha256']==sha('prediction/support.json')
        assert frozen['identity']['runtime_sha256']==sha(B+'runtime_identity.json')
        result['calibration_selected']={k:v['selected_baseline'] for k,v in frozen['operations'].items()}
        cal=obj(B+'calibration/target_rows.json');assert len(cal)==1152
        for phase in ('calibration','validation','formats'):
            prefix=B+phase+'/'
            manifest=obj(prefix+'pre_intervention_manifest.json')
            for name,expected in manifest.items():
                path=prefix+'pre_intervention/'+name
                assert sha(path)==expected
                assert sha(path[:-5]+'.npz')==obj(path)['arrays_sha256']
            result[phase+'_pre_files_verified']=len(manifest)
        all_computed={}
        for phase in ('validation','formats'):
            prefix=B+phase+'/';seal=obj(prefix+'prediction_seal.json')
            paths=dict(prediction_sha256=prefix+'predictions_before_intervention.json',
                       pre_manifest_sha256=prefix+'pre_intervention_manifest.json',
                       frozen_models_sha256=B+'frozen_predictors.json',
                       runtime_sha256=(prefix if phase=='formats' else B)+'runtime_identity.json',
                       support_sha256='prediction/support.json',protocol_sha256='prediction/protocol.json',
                       source_identity_sha256=B+'source_identity.json')
            for key,expected in seal.items():assert sha(paths[key])==expected,(phase,key)
            sealed=obj(prefix+'predictions_before_intervention.json')
            assert sealed['payload']['identity']=={k:v for k,v in seal.items() if k!='prediction_sha256'}
            stamp=datetime.fromisoformat(sealed['created_at_utc']);sealed_identity=digest(seal)
            rows=obj(prefix+'target_rows.json');assert len(rows)==3456
            preds={r['trial_id']:r for r in sealed['payload']['records']}
            assert len(preds)==len(rows) and set(preds)=={r['trial_id'] for r in rows}
            metrics=[];mapping=[];times=[];native_errors=[];ties=Counter()
            expected_families={f['family_id'] for f in families if f['split']=='validation'
                               and (phase=='validation' or f['module']=='content')}
            assert {r['family_id'] for r in rows}==expected_families
            for r in rows:
                tid=r['trial_id'];p=preds[tid]
                pre={k:v for k,v in r.items() if k not in ('actual_delta','post_record','gold','cf_gold')}
                assert digest(pre)==p['input_sha256'],(phase,tid,'input changed')
                saved=obj(prefix+'interventions/'+tid+'.json')
                assert saved['sealed_prediction_identity']==sealed_identity
                date=datetime.fromisoformat(saved['created_at_utc']);assert date>=stamp
                times.append(date)
                assert saved['record']==r['post_record']
                for a in r['post_record']['write_audits'].values():
                    assert a['matched'] and a['relative_error']<=.02;native_errors.append(a['relative_error'])
                s0=np.asarray(r['s0']);actual=np.asarray(r['actual_delta'])
                scores=np.asarray(r['post_record']['candidate_logprobs'])
                np.testing.assert_allclose(scores-scores.mean()-s0,actual,atol=1e-12,rtol=0)
                before=int(np.argmax(s0));after=int(np.argmax(s0+actual));gold='ABC'.index(r['gold'])
                assert (after==gold)==r['post_record']['correct']
                cb=r.get('codebook','ABC');op=f"{r['module']}:L{r['layer']}"
                ties[(cb,op)]+=int(np.sum(s0==s0.max())>1)
                for name,predicted in p['predicted_deltas'].items():
                    predicted=np.asarray(predicted);assert np.isfinite(predicted).all()
                    guess=int(np.argmax(s0+predicted))
                    metrics.append(dict(trial_id=tid,family_id=r['family_id'],codebook=cb,operation=op,predictor=name,
                                        mse=float(np.mean((predicted-actual)**2)),post_answer_match=int(guess==after),
                                        actual_change=int(before!=after),predicted_change=int(before!=guess),
                                        baseline_correct=int(before==gold),post_correct=int(after==gold),
                                        predicted_correct=int(guess==gold),mapping=r['mapping_name']))
                if r['module']=='content':
                    mapping.append(dict(codebook=cb,layer=r['layer'],family_id=r['family_id'],
                        mapping='matching' if r['mapping_name']=='identity' else 'conflicting',
                        baseline_correct=int(before==gold),post_correct=int(after==gold),
                        repaired=int(before!=gold and after==gold),harmed=int(before==gold and after!=gold)))
            df=pd.DataFrame(metrics);all_computed[phase]=df
            reported=table(prefix+'analysis/trial_prediction_metrics.csv')
            keys=['trial_id','predictor'];vals=['mse','post_answer_match','actual_change','predicted_change']
            assert len(df)==len(reported)
            a=df.set_index(keys)[vals].sort_index();b=reported.set_index(keys)[vals].sort_index()
            assert a.index.equals(b.index)
            np.testing.assert_allclose(a.to_numpy(),b.to_numpy(),atol=1e-12,rtol=1e-12)
            mp=pd.DataFrame(mapping)
            got=table(prefix+'analysis/mapping_gain_summary.csv').set_index(['codebook','layer','mapping'])
            for key,g in mp.groupby(['codebook','layer','mapping']):
                rec=got.loc[key]
                assert len(g)==rec.trials and g.family_id.nunique()==rec.n_families
                for col in ('baseline_correct','post_correct','repaired','harmed'):assert int(g[col].sum())==rec[col]
                assert abs((g.post_correct-g.baseline_correct).mean()-rec['mean'])<1e-12
            changes=[]
            for key,g in df.groupby(['codebook','operation','predictor']):
                changed=g[g.actual_change.eq(1)]
                changes.append(dict(codebook=key[0],operation=key[1],predictor=key[2],
                    n_trials=len(g),n_families=g.family_id.nunique(),mse=g.mse.mean(),
                    answer_match=g.post_answer_match.mean(),actual_changed=len(changed),
                    correct_when_changed=int(changed.post_answer_match.sum()),
                    changed_answer_match=changed.post_answer_match.mean()))
            pd.DataFrame(changes).to_csv(OUT/(phase+'_audited_predictions.csv'),index=False)
            for name in ('mapping_gain_summary.csv','mapping_contrasts.csv','primary_comparisons.csv'):
                data=table(prefix+'analysis/'+name)
                qcol=next(c for c in data.columns if c.startswith('holm')) if name!='mapping_gain_summary.csv' else None
                if qcol:np.testing.assert_allclose(holm(data.p_value),data[qcol],atol=1e-12,rtol=1e-12)
                data.to_csv(OUT/(phase+'_'+name),index=False)
            result[phase+'_audit']=dict(trials=len(rows),families=len(expected_families),
                sealed_at=stamp.isoformat(),first_intervention=min(times).isoformat(),last_intervention=max(times).isoformat(),
                sealed_hashes_verified=True,all_predictions_precede_interventions=True,
                metrics_recomputed=len(df),max_target_write_relative_error=max(native_errors),
                baseline_top_ties={str(k):v for k,v in ties.items()})
        # ABC repeats are a reproducibility check, not an independent sample.
        v=all_computed['validation'];f=all_computed['formats']
        left=v[v.operation.str.startswith('content')].set_index(['trial_id','predictor']).sort_index()
        right=f[f.codebook.eq('ABC')].copy();right['trial_id']=right.trial_id.str.removesuffix('__ABC')
        right=right.set_index(['trial_id','predictor']).sort_index()
        np.testing.assert_allclose(left[['mse','post_answer_match']],right[['mse','post_answer_match']],atol=0,rtol=0)
        result['abc_repeat_exact_match']=True
        result['random_control_norm_checks']={}
        for phase in ('calibration','validation'):
            a=table(B+phase+'/analysis/random_native_norm_audit.csv')
            assert a.matched.all() and a.relative_error.max()<=.02
            result['random_control_norm_checks'][phase]=dict(n=len(a),max_relative_error=a.relative_error.max(),unmatched=0)
        tail=table(B+'formats/analysis/current_tail_replay.csv')
        result['tail_replay']=dict(trials=len(tail),max_abs_error=tail.max_abs_error.max(),choice_match=tail.post_answer_match.mean())
        exceptions=[]
        for row in tail[tail.post_answer_match.eq(0)].to_dict('records'):
            record=obj(B+'formats/interventions/'+row['trial_id']+'.json')
            exceptions.append(dict(**row,head_scores=record['tail_scores'],
                actual_candidate_logprobs=record['record']['candidate_logprobs'],
                actual_label=record['record']['predicted_label']))
        result['tail_replay_exceptions']=exceptions
        # Sensitivity descriptions only: preserve all trials in the primary results.
        sensitivity=[]
        for phase in ('formats',):
            rows=obj(B+phase+'/target_rows.json')
            raw_by_id={r['trial_id']:r for r in rows}
            selected=all_computed[phase].query("predictor == 'source_letter'").copy()
            def no_tie(tid):
                row=raw_by_id[tid];before=np.asarray(row['s0']);after=np.asarray(row['post_record']['candidate_logprobs'])
                return np.sum(before==before.max())==1 and np.sum(after==after.max())==1
            selected['no_top_tie']=selected.trial_id.map(no_tie)
            for (cb,op),g in selected.groupby(['codebook','operation']):
                nt=g[g.no_top_tie];changes=nt[nt.actual_change.eq(1)]
                sensitivity.append(dict(codebook=cb,operation=op,all_trials=len(g),non_tied_trials=len(nt),
                    changed_trials=len(changes),changed_prediction_matches=int(changes.post_answer_match.sum()),
                    changed_match_fraction=changes.post_answer_match.mean()))
        pd.DataFrame(sensitivity).to_csv(OUT/'non_tied_descriptive_sensitivity.csv',index=False)
        natural=table(B+'validation/analysis/natural_counterfactual_family_metrics.csv')
        cf=[]
        for op,g in natural.groupby('operation'):
            gain=(g.patched_cf_correct-g.baseline_cf_correct).to_numpy();rng=np.random.default_rng(2601)
            draws=rng.choice(gain,(10000,len(gain)),replace=True).mean(1)
            low,high=np.quantile(draws,[.025,.975])
            cf.append(dict(operation=op,n_families=len(gain),mean=gain.mean(),lower=low,upper=high,
                           scope='additional descriptive family bootstrap, seed 2601'))
        pd.DataFrame(cf).to_csv(OUT/'physical_counterfactual_gain_audit.csv',index=False)
        selected=[]
        for (cb,op),g in f.groupby(['codebook','operation']):
            source=g[g.predictor.eq('source_letter')];null=g[g.predictor.eq('null')]
            changed=source[source.actual_change.eq(1)]
            selected.append(dict(codebook=cb,operation=op,mse=source.mse.mean(),
                reduction_percent=100*(1-source.mse.mean()/null.mse.mean()),
                all_answer_match_percent=100*source.post_answer_match.mean(),
                changed_count=len(changed),changed_correct=int(changed.post_answer_match.sum()),
                changed_match_percent=100*changed.post_answer_match.mean()))
        pd.DataFrame(selected).to_csv(OUT/'selected_content_predictors.csv',index=False)
        result['selected_content_predictors']=selected
        # Reconcile original 12 contrasts with the new 6, without rewriting prior reports.
        old=pd.read_csv(PROJECT/'output/analysis/Frozen_Format_Transfer/mapping_gap_audit_supplement.csv')
        new=table(B+'formats/analysis/mapping_contrasts.csv').assign(model='deepseek16b')
        new['operation']='content:L'+new.layer.astype(str)
        cols=['model','codebook','operation','mean','lower','upper','p_value']
        combined=pd.concat([old[cols],new[cols]],ignore_index=True)
        combined['holm_p_all_18']=holm(combined.p_value)
        combined.to_csv(OUT/'three_model_mapping_contrasts_18.csv',index=False)
        result['deepseek_holm_all18']=combined[combined.model.eq('deepseek16b')].to_dict('records')
        for name in ('natural_counterfactual_summary.csv','random_specificity_secondary.csv'):
            table(B+'validation/analysis/'+name).to_csv(OUT/name,index=False)
        # Inspect log failure markers; ordinary warnings are not counted as failed execution.
        result['log_failure_markers']={}
        for stage in ('install','preflight','calibration','validation','formats'):
            text=z.read('logs/'+stage+'.log').decode(errors='replace')
            result['log_failure_markers'][stage]=[word for word in ('Traceback (most recent call last)',
                'CUDA error:','no kernel image is available') if word in text]
        result['software_sources_match']={}
        suite=obj('suite_identity.json')
        for name,expected in suite['source_files'].items():
            path=PROJECT/'VisualTorque/deepseek_vl2'/name
            result['software_sources_match'][name]=path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==expected
        assert all(result['software_sources_match'].values())
    (OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('software_sources_match','runtime','completion','gpu_check','support_provenance')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
