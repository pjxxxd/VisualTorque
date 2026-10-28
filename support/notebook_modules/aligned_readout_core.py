"""CPU helpers for matched-capture readout verification; no model execution."""
from collections.abc import Mapping
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import sklearn
import scipy


def fixed_variant(scene_id, seed=20260820):
    key=hashlib.sha256(f'{seed}:variant:{scene_id}'.encode()).hexdigest()
    return 'permutation_'+str(int(key[:8],16)%6)


def probe_parameters(probe):
    mean,scale,coef=[np.asarray(probe[k],np.float32) for k in ('mean','scale','coef')]
    assert mean.ndim==1 and mean.shape==scale.shape==coef.shape
    assert (scale>0).all() and all(np.isfinite(a).all() for a in (mean,scale,coef))
    assert float(np.asarray(coef,np.float64)@coef)>0 and np.isfinite(probe['intercept'])
    return mean,scale,coef,float(probe['intercept'])


def decode_with(hidden,probe):
    return readout_value(hidden,*probe_parameters(probe))


def readout_class(value,threshold=.5):
    return 'left' if value>threshold else 'right' if value < -threshold else 'balanced'


def six_mapping_mean(native):
    # Preserve the original per-mapping FP16 capture and FP16 scene-mean rule.
    a=np.asarray(native,np.float32)
    assert a.ndim==2 and a.shape[0]==6 and np.isfinite(a).all()
    return a.astype(np.float16).astype(np.float32).mean(axis=0).astype(np.float16).astype(np.float32)


def fit_fixed_probe(states,targets,scene_ids,layer,alpha,provenance):
    x=np.asarray(states,np.float32);y=np.asarray(targets,float)
    assert len(x)==len(y)==len(scene_ids)==len(set(scene_ids)) and np.isfinite(x).all() and np.isfinite(y).all()
    scaler=StandardScaler()
    transformed=scaler.fit_transform(x)
    estimator=Ridge(alpha=float(alpha),solver='lsqr',max_iter=5000).fit(transformed,y)
    result=dict(layer=int(layer),alpha=float(alpha),mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),coef=estimator.coef_.astype(np.float32),intercept=float(estimator.intercept_),
        fit_scene_ids=list(scene_ids),fit_provenance=provenance)
    probe_parameters(result)
    return json.loads(json.dumps(result,default=plain,allow_nan=False))


def aligned_cache(root,stage,key,identity,compute):
    folder=Path(root)/'cache'/stage/digest(key)[:24]
    folder.mkdir(parents=True,exist_ok=True)
    meta_path=folder/'record.json';array_path=folder/'states.npz'
    if meta_path.is_file():
        saved=read_json(meta_path)
        assert saved['identity']==identity and saved['key']==key, 'Cache identity mismatch; use a new RUN_ID.'
        assert array_path.is_file() and saved['states_sha256']==file_hash(array_path)
        with np.load(array_path,allow_pickle=False) as z:
            arrays={k:z[k].copy() for k in z.files}
        return saved['records'],arrays,True
    records,arrays=compute()
    assert all(np.isfinite(a).all() for a in arrays.values())
    temp=array_path.with_suffix('.npz.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,**arrays)
    temp.replace(array_path)
    save_json(meta_path,dict(identity=identity,key=key,records=records,states_sha256=file_hash(array_path)))
    return read_json(meta_path)['records'],arrays,False


def save_arrays(packet,name,**arrays):
    path=packet.root/name;temp=path.with_suffix('.npz.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,**arrays)
    temp.replace(path);packet.outputs.add(path)
    return file_hash(path)


def require_same_pre(before,reference,input_hash,reference_hash):
    assert input_hash==reference_hash, 'Prepared inputs changed within a paired comparison.'
    assert np.array_equal(before,reference), 'Pre-intervention hidden state changed within a paired comparison.'


def candidate_margin(record):
    scores=record['semantic_logprobs'];gold=record['gold_semantic']
    return float(scores[gold]-max(v for k,v in scores.items() if k!=gold))


def random_validity(actual_norm,target_norm,tolerance=.02):
    assert np.isfinite(actual_norm) and np.isfinite(target_norm) and actual_norm>=0 and target_norm>=0
    # A nonzero edit cannot match a zero reference; its relative error is undefined.
    if target_norm==0:return bool(actual_norm==0),0. if actual_norm==0 else None
    relative=abs(actual_norm-target_norm)/target_norm
    return bool(relative<=tolerance),float(relative)


def summarize_test(records,expected=1007):
    frame=pd.DataFrame(records)
    assert len(frame)==expected and frame.scene_id.is_unique
    rows=[]
    for group,sub in [('all',frame),*[(str(k),v) for k,v in frame.groupby('problem_type')]]:
        for probe in ('aligned','historical'):
            correct=sub[probe+'_readout_correct'].astype(bool)
            rows.append(dict(group=group,probe=probe,scenes=len(sub),readout_correct=int(correct.sum()),
                readout_accuracy=float(correct.mean()),model_correct=int(sub.correct.sum()),model_accuracy=float(sub.correct.mean()),
                readout_correct_model_wrong=int((correct&~sub.correct.astype(bool)).sum()),
                readout_mae=float((sub[probe+'_readout']-sub.net_torque).abs().mean()),
                interpretation='fixed_single_mapping_on_reused_test_scenes'))
    return pd.DataFrame(rows)


def summarize_interventions(records,selected_ids,control_ids,replicates=10000):
    frame=pd.DataFrame(records)
    assert not frame.duplicated(['scene_id','condition','random_seed']).any()
    assert set(frame.condition)=={'zero','oracle','random'}
    selected_ids=set(selected_ids);control_ids=set(control_ids)
    assert control_ids<=selected_ids
    zero=frame[frame.condition.eq('zero')].set_index('scene_id').sort_index()
    oracle=frame[frame.condition.eq('oracle')].set_index('scene_id').sort_index()
    random=frame[frame.condition.eq('random')]
    assert set(zero.index)==set(oracle.index)==selected_ids and zero.index.is_unique and oracle.index.is_unique
    assert set(random.scene_id)==control_ids
    assert all(set(g.random_seed)=={0,1,2,3} and len(g)==4 for _,g in random.groupby('scene_id'))
    ci=lambda values:scene_bootstrap(np.asarray(values,float),replicates=replicates)
    summaries=[]
    for name,ids in [('all_304',sorted(selected_ids)),('tilt_263',list(zero.index[zero.net_torque.ne(0)])),
                     ('balance_41',list(zero.index[zero.net_torque.eq(0)]))]:
        if not ids:continue
        a=zero.loc[ids];b=oracle.loc[ids]
        repaired=(~a.correct.astype(bool)&b.correct.astype(bool));harmed=(a.correct.astype(bool)&~b.correct.astype(bool))
        change=b.correct.astype(float)-a.correct.astype(float)
        interval=ci(change)
        summaries.append(dict(group=name,scenes=len(ids),before_correct=int(a.correct.sum()),after_correct=int(b.correct.sum()),
            original_errors=int((~a.correct.astype(bool)).sum()),repaired=int(repaired.sum()),harmed=int(harmed.sum()),
            answer_changed=int((a.predicted_semantic!=b.predicted_semantic).sum()),
            readout_correct_before=int(a.readout_correct.sum()),readout_correct_after=int(b.readout_correct.sum()),
            readout_mae_before=float((a.decoded_post-a.net_torque).abs().mean()),
            readout_mae_after=float((b.decoded_post-b.net_torque).abs().mean()),
            oracle_achieved=int(b.achieved.eq(True).sum()),oracle_failed=int(b.achieved.ne(True).sum()),
            accuracy_change=interval['mean'],lower=interval['lower'],upper=interval['upper'],
            excludes_5pp_gain_for_all_attempts=bool(interval['upper']<.05)))
    contrasts=[]
    def add_contrast(scope,ids,kind):
        ids=sorted(ids)
        if not ids:return
        a=zero.loc[ids];o=oracle.loc[ids]
        if kind=='oracle_minus_zero':
            acc=o.correct.astype(float)-a.correct.astype(float)
            margin=o.correct_answer_margin-a.correct_answer_margin
        else:
            r=random[random.scene_id.isin(ids)].groupby('scene_id')[['correct','correct_answer_margin']].mean().loc[ids]
            acc=o.correct.astype(float)-r.correct
            margin=o.correct_answer_margin-r.correct_answer_margin
        for metric,x in [('accuracy',acc),('correct_answer_margin',margin)]:
            contrasts.append(dict(scope=scope,comparison=kind,metric=metric,**ci(x),
                interpretation='paired_protocol_robustness_not_new_independent_confirmation'))
    add_contrast('all_304_attempted',selected_ids,'oracle_minus_zero')
    succeeded=set(oracle.index[oracle.achieved.eq(True)])
    add_contrast('oracle_achieved_subset',succeeded,'oracle_minus_zero')
    add_contrast('all_72_attempted',control_ids,'oracle_minus_random')
    valid=set(random.groupby('scene_id').valid_control.all().loc[lambda x:x].index)&succeeded
    add_contrast('oracle_achieved_all_four_random_valid',valid,'oracle_minus_random')
    technical=dict(oracle_attempted=len(oracle),oracle_achieved=len(succeeded),oracle_failed=len(oracle)-len(succeeded),
        zero_or_tiny_edits=int((oracle.actual_delta_norm<=1e-10).sum()),random_attempted=len(random),
        random_invalid=int((~random.valid_control.astype(bool)).sum()),common_valid_control_scenes=len(valid),
        all_attempts_retained=True)
    return pd.DataFrame(summaries),pd.DataFrame(contrasts),technical
