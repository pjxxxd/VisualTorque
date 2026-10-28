"""Section 4.4: new-model development selection followed by frozen numerical correction."""
from functools import partial
from core import load_kernels, load_model, fit_examples, verify_scoring_path


def forward(n, example, edit=None, reference=None, reference_input_hash=None):
    torch,np=n['torch'],n['np']
    prepared=n['prepare_examples']([example]); fingerprint=n['prepared_fingerprint'](prepared)
    if reference_input_hash is not None:assert reference_input_hash==fingerprint
    captured={}; audit={}
    def hook(_module,_args,output):
        hidden=n['output_hidden_tensor'](output)
        assert hidden.dtype==torch.bfloat16 and hidden.shape[0]==1
        pre=hidden[0,-1].detach().float().cpu().numpy().copy(); captured['pre']=pre
        if reference is not None:assert np.array_equal(pre,reference),'Untreated hidden state replay changed.'
        if edit is None:
            captured['post']=pre.copy(); return None
        cast=lambda x:torch.as_tensor(np.asarray(x).copy(),dtype=torch.bfloat16).float().numpy()
        proposed,details=edit(pre,cast)
        new=hidden.clone(); new[0,-1]=torch.as_tensor(proposed,dtype=hidden.dtype,device=hidden.device)
        post=new[0,-1].detach().float().cpu().numpy().copy(); captured['post']=post
        audit.update(details,actual_delta_norm=float(np.linalg.norm(post-pre)),
            standardized_delta_norm=float(np.linalg.norm((post-pre)/n['PROBE_SCALE'])),
            decoded_pre=n['decode'](pre),decoded_post=n['decode'](post),hidden_norm=float(np.linalg.norm(pre)))
        return n['replace_output_hidden'](output,new)
    handle=n['DECODER_LAYERS'][n['SELECTED_LAYER']].register_forward_hook(hook)
    try:
        with torch.inference_mode():out=n['adapter'].model(**prepared['inputs'],use_cache=False)
        r=n['score_records'](prepared,out.logits)[0]
    finally:handle.remove()
    r['candidate_logprobs']=dict(zip(example.labels,r['candidate_logprobs']))
    r.update(example_id=example.example_id,scene_id=str(example.scene_id),variant=example.variant,
        split=example.split,problem_type=example.problem_type,net_torque=int(example.net_torque),
        input_fingerprint=fingerprint,correct_answer_margin=n['correct_margin'](r),**audit)
    return r,captured


def capture_fit(n, root, groups, layers, key):
    """Native BF16 states; each scene is one independent unit across its six mappings."""
    np=n['np']; means={}; records={}
    identity=dict(runtime=n['RUNTIME_IDENTITY'],stage=key,layers=list(layers),
        inputs={s:[n['example_stamp'](e) for e in groups[s]] for s in groups})
    n['freeze'](root/(key+'_identity.json'),identity)
    for sid,examples in n['tqdm'](list(groups.items()),desc=key):
        def compute(examples=examples):
            rows=[]; states={str(l):[] for l in layers}
            for ex in examples:
                r,h=n['final_forward'](ex,layers); rows.append(r)
                for l in layers:states[str(l)].append(h[l].ravel())
            return rows,{k:np.stack(v).astype(np.float32) for k,v in states.items()}
        rows,arrays,_=n['aligned_cache'](root,key,sid,n['digest']([identity,sid]),compute)
        assert len(rows)==6 and all(a.shape[0]==6 for a in arrays.values())
        means[sid]={k:(a.mean(axis=0) if key=='repair_training' else n['six_mapping_mean'](a)) for k,a in arrays.items()}
        records[sid]=rows
    return means,records,identity


def select_probe(n, means, targets, train_ids, dev_ids, layers):
    assert set(train_ids).isdisjoint(dev_ids) and set(train_ids)|set(dev_ids)==set(means)
    np=n['np']; candidates=[]
    yt=np.array([targets[s] for s in train_ids],float); yd=np.array([targets[s] for s in dev_ids],float)
    assert float(((yd-yd.mean())**2).sum())>0
    for layer in layers:
        xt=np.array([means[s][str(layer)] for s in train_ids],np.float32)
        xd=np.array([means[s][str(layer)] for s in dev_ids],np.float32)
        for alpha in (.01,.1,1.,10.,100.,1000.):
            p=n['fit_fixed_probe'](xt,yt,train_ids,layer,alpha,dict(stage='train_only'))
            pred=np.array([n['decode_with'](x,p) for x in xd])
            r2=1.-float(((yd-pred)**2).sum()/((yd-yd.mean())**2).sum())
            candidates.append(dict(layer=layer,alpha=alpha,dev_r2=r2))
    best=max(candidates,key=lambda x:(x['dev_r2'],-x['alpha'],-x['layer']))
    return best,candidates


def fit(n):
    fit_examples(n); load_model(n)
    root=n['ROOT']/'readout'; root.mkdir(exist_ok=True)
    groups=n['FIT_EXAMPLES']
    verify_scoring_path(n,groups[n['TRAIN_IDS'][0]][0],root/'technical_check.json')
    means,records,identity=capture_fit(n,root,groups,tuple(range(27)),'training')
    targets={s:groups[s][0].net_torque for s in groups}
    best,rows=select_probe(n,means,targets,n['TRAIN_IDS'],n['DEV_IDS'],tuple(range(27)))
    n['pd'].DataFrame(rows).to_csv(root/'development_grid.csv',index=False)
    n['freeze'](root/'selection.json',dict(selected=best,train_scenes=n['TRAIN_IDS'],dev_scenes=n['DEV_IDS'],
        selection_metric='maximum development R2; ties prefer smaller alpha then shallower layer',test_scenes_used=0))
    ids=sorted(groups); layer=best['layer']
    provenance=dict(training_identity=n['digest'](identity),train=235,development=19,
        refit_scenes=254,test_scenes_used=0,storage='per-mapping FP16; six-map FP32 mean then FP16')
    p=n['fit_fixed_probe']([means[s][str(layer)] for s in ids],[targets[s] for s in ids],
                           ids,layer,best['alpha'],provenance)
    n['freeze'](root/'frozen_probe.json',p)
    n['save_json'](root/'fit_complete.json',dict(status='development_complete_not_test_evidence',selected=best))
    print('Readout selected on development only:',best)


def install_readout(n, probe):
    load_kernels(n,['aligned_readout_gpu.py'])
    mean,scale,coef,intercept=n['probe_parameters'](probe)
    n.update(SELECTED_LAYER=probe['layer'],PROBE_MEAN=mean,PROBE_SCALE=scale,PROBE_COEF=coef,
        PROBE_INTERCEPT=intercept,READOUT_TOLERANCE=.05,NORM_TOLERANCE=.02,BALANCE_THRESHOLD=.5)
    n['aligned_forward']=partial(forward,n)


def validate(n):
    fit_examples(n)
    root=n['ROOT']/'readout'; probe=n['read_json'](root/'frozen_probe.json')
    assert set(probe['fit_scene_ids'])==set(n['FIT_EXAMPLES'])
    shared=n['PROJECT_ROOT']/'results/cross_model_readout_correction_v2'
    test_ids=list(map(str,n['read_json'](shared/'shared_1007_scene_ids.json')))
    selected=n['read_json'](shared/'shared_304_manifest.json')
    assert len(test_ids)==len(set(test_ids))==1007 and set(test_ids).isdisjoint(probe['fit_scene_ids'])
    assert len(selected)==304 and len({str(r['scene_id']) for r in selected})==304
    examples={s:next(e for e in n['ALL_EXAMPLES'][s] if e.variant==n['fixed_variant'](s)) for s in test_ids}
    assert all(ex.split=='probe_test' for ex in examples.values())
    controls={str(r['scene_id']) for r in selected if r['orthogonal_control_subset']}; assert len(controls)==72
    for row in selected:
        ex=examples[str(row['scene_id'])]
        assert row['example_id']==ex.example_id and row['variant']==ex.variant
        assert row['gold_outcome']==ex.gold_semantic and row['net_torque']==ex.net_torque
    load_model(n); install_readout(n,probe)
    n['PROBE_SHA']=n['file_hash'](root/'frozen_probe.json')
    identity=dict(runtime=n['RUNTIME_IDENTITY'],probe_sha256=n['PROBE_SHA'],
        test_manifest=n['file_hash'](shared/'shared_1007_scene_ids.json'),
        diagnostic_manifest=n['file_hash'](shared/'shared_304_manifest.json'),
        examples={s:n['example_stamp'](e) for s,e in examples.items()})
    n['freeze'](root/'validation_identity.json',identity)
    # One critical, paired no-op check before costly validation.
    dev=n['FIT_EXAMPLES'][n['DEV_IDS'][0]][0]
    a,h=forward(n,dev); b,hb=forward(n,dev,lambda pre,cast:(pre.copy(),{}),h['pre'],a['input_fingerprint'])
    assert n['np'].array_equal(h['pre'],hb['post']) and a['candidate_logprobs']==b['candidate_logprobs']
    rows=[]; by_id={}
    for sid,ex in n['tqdm'](list(examples.items()),desc='Readout: 1007 fixed test scenes'):
        rs,arrays,_=n['aligned_cache'](root,'test',sid,n['digest']([identity,sid]),lambda ex=ex:n['capture_test_example'](ex))
        r=rs[0]; value=n['decode'](arrays['native'])
        r.update(readout=value,readout_correct=n['readout_class'](value)==ex.gold_semantic)
        rows.append(r); by_id[sid]=(r,arrays['native'])
    n['save_json'](root/'test_records.json',rows)
    frame=n['pd'].DataFrame(rows); frame.to_csv(root/'test_records.csv',index=False)
    summaries=[]
    for name,g in [('all',frame),*[(str(k),v) for k,v in frame.groupby('problem_type')]]:
        summaries.append(dict(group=name,scenes=len(g),readout_accuracy=float(g.readout_correct.mean()),
            model_accuracy=float(g.correct.mean()),readout_mae=float((g.readout-g.net_torque).abs().mean()),
            readout_correct_model_wrong=int((g.readout_correct & ~g.correct).sum())))
    n['pd'].DataFrame(summaries).to_csv(root/'test_summary.csv',index=False)
    records=[]
    for item in n['tqdm'](selected,desc='Numerical correction: 304 fixed scenes'):
        sid=str(item['scene_id']); ex=examples[sid]; rec,h=by_id[sid]
        rs,_,_=n['aligned_cache'](root,'interventions',sid,n['digest']([identity,sid]),
            lambda ex=ex,rec=rec,h=h,sid=sid:n['run_aligned_scene'](ex,rec,h,sid in controls))
        records.extend(rs)
    n['save_json'](root/'intervention_records.json',records)
    n['pd'].DataFrame(records).to_csv(root/'intervention_records.csv',index=False)
    summary,contrasts,technical=n['summarize_interventions'](records,[str(r['scene_id']) for r in selected],controls)
    contrasts['interpretation']='new_model_same_fixed_test_scenes'
    summary.to_csv(root/'correction_summary.csv',index=False);contrasts.to_csv(root/'correction_contrasts.csv',index=False)
    n['save_json'](root/'completion.json',dict(status='heldout_readout_validation_complete',
        test_scenes=1007,intervention_scenes=304,random_scenes=72,technical=technical,selected_layer=probe['layer']))
    print(summary.to_string(index=False))
