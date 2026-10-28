"""One forward convention for training, test, zero replay and all interventions."""


def aligned_forward(example,edit=None,reference=None,reference_input_hash=None):
    global NEW_FORWARD_COUNT
    plan=adapter._candidate_plan(example)
    assert plan is not None
    prefix,tokens=plan
    inputs=adapter._append_text_prefix(adapter._to_device(adapter._template(
        [example.messages()],add_generation_prompt=True)),prefix)
    assert inputs['input_ids'].shape[0]==1 and bool(inputs['attention_mask'][0,-1])
    fingerprint=digest(tensor_signature(inputs))
    if reference_input_hash is not None:assert fingerprint==reference_input_hash
    captured={};audit={};handle=None
    def hook(_module,_inputs,output):
        hidden=output_hidden(output)
        assert hidden.dtype==torch.bfloat16 and hidden.shape[0]==1
        pre=hidden[0,-1].detach().float().cpu().numpy().copy()
        if reference is not None:require_same_pre(pre,reference,fingerprint,reference_input_hash)
        captured['pre']=pre
        if edit is None:
            captured['post']=pre.copy()
            return None
        cast=lambda x:torch.as_tensor(np.asarray(x).copy(),dtype=hidden.dtype).float().cpu().numpy()
        proposed,details=edit(pre,cast)
        modified=hidden.clone()
        modified[0,-1]=torch.as_tensor(proposed,dtype=hidden.dtype,device=hidden.device)
        post=modified[0,-1].detach().float().cpu().numpy().copy()
        captured['post']=post
        audit.update(details)
        audit.update(actual_delta_norm=float(np.linalg.norm(post-pre)),
            standardized_delta_norm=float(np.linalg.norm((post-pre)/PROBE_SCALE)),
            decoded_pre=decode(pre),decoded_post=decode(post),hidden_norm=float(np.linalg.norm(pre)))
        return replace_hidden(output,modified)
    handle=DECODER_LAYERS[SELECTED_LAYER].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            NEW_FORWARD_COUNT+=1
            result=adapter.model(**inputs,use_cache=False)
            scores=torch.log_softmax(result.logits[0,-1].float(),dim=-1)
            values={str(label):float(scores[tokens[label]].cpu()) for label in example.labels}
    finally:
        handle.remove()
    assert set(captured)=={'pre','post'} and all(np.isfinite(v).all() for v in captured.values())
    label=max(example.labels,key=lambda k:values[str(k)])
    predicted=str(example.label_to_semantic[label])
    record=dict(example_id=str(example.example_id),scene_id=str(example.scene_id),variant=str(example.variant),
        split=str(example.split),problem_type=str(example.problem_type),net_torque=int(example.net_torque),
        gold_semantic=str(example.gold_semantic),predicted_semantic=predicted,predicted_label=str(label),
        correct=predicted==str(example.gold_semantic),input_fingerprint=fingerprint,
        label_to_semantic={str(k):str(v) for k,v in example.label_to_semantic.items()},
        candidate_logprobs=values,semantic_logprobs={str(example.label_to_semantic[k]):values[str(k)] for k in example.labels},**audit)
    record['correct_answer_margin']=candidate_margin(record)
    return record,captured


def decode(hidden):
    return readout_value(hidden,PROBE_MEAN,PROBE_SCALE,PROBE_COEF,PROBE_INTERCEPT)


def aligned_oracle(target):
    def edit(pre,cast):
        post,audit=numerical_correction(pre,float(target),PROBE_MEAN,PROBE_SCALE,PROBE_COEF,PROBE_INTERCEPT,
            cast,tolerance=READOUT_TOLERANCE,steps=32)
        audit.update(condition='oracle',random_seed=-1,target_torque=float(target))
        return post,audit
    return edit


def aligned_random(scene_id,seed,target_norm):
    def edit(pre,cast):
        direction=deterministic_direction(pre.shape,f'20260820:{scene_id}:random:{seed}',
            np.ones_like(PROBE_SCALE),PROBE_COEF,orthogonal=False)
        post,audit=match_native_norm(pre,direction,target_norm,cast,tolerance=NORM_TOLERANCE)
        audit.update(condition='random',random_seed=int(seed),oracle_actual_norm=float(target_norm))
        return post,audit
    return edit


def capture_training_scene(examples):
    records=[];states=[]
    for example in examples:
        r,a=aligned_forward(example)
        records.append(r);states.append(a['pre'])
    native=np.stack(states).astype(np.float32)
    return records,dict(native=native,fit_mean=six_mapping_mean(native))


def capture_test_example(example):
    r,a=aligned_forward(example)
    return [r],dict(native=a['pre'])


def run_aligned_scene(example,expected_record,expected_hidden,with_controls):
    zero,a=aligned_forward(example,reference=expected_hidden,reference_input_hash=expected_record['input_fingerprint'])
    assert zero['predicted_semantic']==expected_record['predicted_semantic']
    assert max(abs(zero['candidate_logprobs'][k]-expected_record['candidate_logprobs'][k]) for k in zero['candidate_logprobs'])<=1e-6
    pre=a['pre'];fingerprint=zero['input_fingerprint']
    zero.update(condition='zero',random_seed=-1,actual_delta_norm=0.,decoded_pre=decode(pre),decoded_post=decode(pre))
    oracle,b=aligned_forward(example,aligned_oracle(example.net_torque),pre,fingerprint)
    oracle['residual']=abs(oracle['decoded_post']-float(example.net_torque))
    oracle['achieved']=oracle['residual']<=READOUT_TOLERANCE
    records=[zero,oracle];arrays=dict(pre=pre,zero_post=a['post'],oracle_post=b['post'])
    if with_controls:
        for seed in RANDOM_SEEDS:
            row,c=aligned_forward(example,aligned_random(example.scene_id,seed,oracle['actual_delta_norm']),pre,fingerprint)
            valid,error=random_validity(row['actual_delta_norm'],oracle['actual_delta_norm'],NORM_TOLERANCE)
            row.update(valid_control=valid,norm_matched=valid,norm_error_relative=error)
            records.append(row);arrays['random_'+str(seed)+'_post']=c['post']
    for r in records:
        r['readout_correct']=readout_class(r['decoded_post'],BALANCE_THRESHOLD)==r['gold_semantic']
        r['original_model_correct']=zero['correct']
        r['repaired']=not zero['correct'] and r['correct']
        r['harmed']=zero['correct'] and not r['correct']
        r['readout_version']='aligned_current_capture'
        r['probe_sha256']=PROBE_SHA
    return records,arrays
