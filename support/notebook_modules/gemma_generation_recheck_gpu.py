"""Generation capture for one frozen Gemma protocol; embedded in the notebook."""
import gc
import shutil
import time
from datetime import datetime, timezone
from tqdm.auto import tqdm
from visual_torque_toolkit.behavior.data import BehaviorExample
from visual_torque_toolkit.behavior.schema import PromptCondition


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def tensor_identity(inputs):
    result={}
    for key,value in inputs.items():
        if torch.is_tensor(value):
            x=value.detach().cpu().contiguous()
            result[key]=dict(shape=list(x.shape),dtype=str(x.dtype),
                sha256=hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest())
        else:
            result[key]=str(value)
    return result


def example_for(inp):
    scene=SCENES[inp['scene_id']]
    semantics=tuple(inp['semantics'])
    words={'left':'left side','right':'right side','balanced':'equal'}
    if inp['wording']=='neutral_question':words['balanced']='balanced (neither side tips)'
    return BehaviorExample(example_id=inp['trial_id'],question_id=inp['trial_id'],scene_id=inp['scene_id'],
        condition=PromptCondition.IMAGE_RULE,question_kind='outcome',variant='image__'+inp['wording'],
        split='frozen_validation',problem_type=inp['stratum'],outcome=inp['gold_semantic'],
        net_torque=scene['features']['net_torque'],prompt=inp['prompt'],
        image_path=LOCAL_IMAGES/inp['image_filename'],labels=LABELS,
        option_text=tuple(words[s] for s in semantics),option_semantics=semantics,
        gold_label=inp['gold_label'],gold_semantic=inp['gold_semantic'])


def initialize_runtime():
    global RUNTIME_IDENTITY,RUNTIME_SHA
    import PIL,transformers
    assert adapter.metadata['resolved_revision']==REVISION
    assert adapter.metadata['dtype']=='bfloat16'
    assert not getattr(adapter.model,'is_quantized',False),'Use the native BF16 checkpoint.'
    assert not any(str(v) in ('cpu','disk') for v in getattr(adapter.model,'hf_device_map',{}).values()),'Full model must fit on GPU; no CPU/disk offload.'
    adapter.model.eval();torch.manual_seed(20260920);np.random.seed(20260920)
    torch.backends.cuda.matmul.allow_tf32=False
    processor=adapter.processor;tokenizer=processor.tokenizer
    serial=lambda obj:json.loads(json.dumps(obj,default=str))
    prior=SOURCE_SNAPSHOT['source_runtime']
    values=dict(model_metadata=adapter.metadata,gpu=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        torch=torch.__version__,transformers=transformers.__version__,pillow=PIL.__version__,cuda=torch.version.cuda,
        numpy=np.__version__,pandas=pd.__version__,dtype='bfloat16',batch_size=1,thinking=False,
        toolkit={str(p.relative_to(TOOLKIT_ROOT)):file_hash(p) for p in sorted((TOOLKIT_ROOT/'src').rglob('*.py'))},
        processor_config=serial(processor.to_dict()),image_processor_config=serial(processor.image_processor.to_dict()),
        chat_template=processor.chat_template,tokenizer_chat_template=tokenizer.chat_template)
    differences={k:dict(identical=v==prior.get(k)) for k,v in values.items()}
    critical=('transformers','pillow','dtype','batch_size','thinking','toolkit','processor_config','image_processor_config','chat_template','tokenizer_chat_template')
    changed=[k for k in critical if not differences[k]['identical']]
    assert not changed,'Restore the archived toolkit/preprocessing identity before this recheck: '+str(changed)
    token_contract={}
    for text in ('<|turn>','<turn|>','<|channel>','<channel|>','A','B','C'):
        ids=tokenizer(text,add_special_tokens=False)['input_ids']
        assert len(ids)==1,('Expected pinned single-token contract',text,ids)
        decoded=processor.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        assert decoded==text,(text,decoded)
        token_contract[text]=dict(ids=ids,special=ids[0] in tokenizer.all_special_ids,decoded=decoded)
    assert token_contract['<turn|>']['special']
    eos=adapter.model.generation_config.eos_token_id
    eos=[eos] if isinstance(eos,int) else list(eos or [])
    assert token_contract['<turn|>']['ids'][0] in eos,'The archived turn-end token is not an active generation stop.'
    plans={}
    for wording in WORDINGS:
        ex=example_for(next(i for i in INPUTS if i['wording']==wording))
        plan=adapter._candidate_plan(ex)
        assert plan is not None and not plan[0]
        plans[wording]=dict(prefix=list(plan[0]),token_by_label=plan[1])
        assert plans[wording]==prior['candidate_plans']['image']
    values.update(schema=SCHEMA,implementation=IMPLEMENTATION_SHA256,parser_version=PARSER_VERSION,
        source_zip_sha256=SOURCE_SHA,protocol_sha256=file_hash(RUN_ROOT/'protocol.json'),
        inputs_sha256=file_hash(SHARED_ROOT/'inputs.json'),generation_config=serial(adapter.model.generation_config.to_dict()),
        generation_overrides=dict(do_sample=False,max_new_tokens=16,pad_token_id=tokenizer.pad_token_id),
        attention_implementation=serial(getattr(adapter.model.config,'_attn_implementation',None)),
        token_contract=token_contract,candidate_plans=plans)
    RUNTIME_IDENTITY=values;RUNTIME_SHA=freeze(PHASE_ROOT/'runtime_identity.json',values)
    freeze(PHASE_ROOT/'source_environment_comparison.json',differences)
    print('Changed recorded environment fields:',[k for k,v in differences.items() if not v['identical']] or 'none')
    print('Response grammar frozen before generation. Channel wrappers remain separately labeled.')
    display(pd.DataFrame([dict(text=k,**v) for k,v in token_contract.items()]))


def capture(inp, *, preflight=False):
    ex=example_for(inp)
    inputs=adapter._to_device(adapter._template([ex.messages()],add_generation_prompt=True))
    identity=tensor_identity(inputs);old=ARCHIVED_ROWS[inp['trial_id']]
    assert identity==old['prepared_inputs'],'Input tensors differ from archive: '+inp['trial_id']
    assert inputs['input_ids'].shape[0]==1
    prompt_ids=inputs['input_ids'][0].detach().cpu().tolist()
    assert prompt_ids==old['input_token_ids']
    plan=adapter._candidate_plan(ex);assert plan is not None and not plan[0]
    tokens=plan[1];assert tokens==old['candidate_tokens']
    replay_scores=None
    if preflight:
        with torch.inference_mode():
            logits=adapter.model(**dict(inputs),use_cache=False).logits[0,-1].float()
            lp=torch.log_softmax(logits,dim=-1)
            replay_scores={l:float(lp[tokens[l]].item()) for l in LABELS}
        assert np.isfinite(list(replay_scores.values())).all()
        del logits,lp
    started=time.monotonic();width=inputs['input_ids'].shape[1]
    # Same decoding arguments as the archived adapter; return scores only adds instrumentation.
    with torch.inference_mode():
        output=adapter.model.generate(**dict(inputs),do_sample=False,max_new_tokens=16,
            pad_token_id=adapter.processor.tokenizer.pad_token_id,return_dict_in_generate=True,output_scores=True)
    ids=output.sequences[0,width:].detach().cpu().tolist()
    assert ids and len(ids)==len(output.scores)
    raw=adapter.processor.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False).strip()
    parsed=parse_response(raw)
    steps=[]
    finite=lambda value:float(value) if np.isfinite(float(value)) else None
    for index,(token,score) in enumerate(zip(ids,output.scores)):
        lp=torch.log_softmax(score[0].float(),dim=-1)
        candidate={l:finite(lp[tokens[l]].item()) for l in LABELS}
        steps.append(dict(index=index,token_id=int(token),token_logprob=finite(lp[token].item()),
            argmax_token_id=int(score[0].argmax().item()),candidate_logprobs=candidate,
            abc_probability_mass=float(torch.exp(lp[[tokens[l] for l in LABELS]]).sum().item())))
    inp_meta={k:inp[k] for k in ('trial_id','family_id','scene_id','stratum','role','mirror','mapping_index','wording','gold_label','gold_semantic','semantics','prompt_sha256')}
    row=dict(inp_meta,runtime_sha256=RUNTIME_SHA,input_definition_sha256=digest(inp),
        prepared_inputs=identity,input_sha256=digest(identity),input_token_ids=prompt_ids,
        prompt_tail_decoded=adapter.processor.decode(prompt_ids[-64:],skip_special_tokens=False,clean_up_tokenization_spaces=False),
        generated_ids=ids,generated_token_pieces=[adapter.processor.tokenizer.convert_ids_to_tokens(i) for i in ids],
        generation_steps=steps,raw_response=raw,parsed=parsed,
        archived_adapter_parsed=adapter._parse_generated_content(raw,inputs['input_ids'][0],output.sequences[0,width:]),
        hit_token_limit_without_turn_end=len(ids)>=16 and not parsed['completed_turn'],
        replayed_candidate_logprobs=replay_scores,preflight=preflight,
        generation_seconds=time.monotonic()-started,recorded_at_utc=now_utc())
    validate_row(row,inp,RUNTIME_SHA)
    del output,inputs,lp
    return row


def run_preflight():
    rows=load_rows(FAMILY_ROOT,INPUT_BY_ID,RUNTIME_SHA)
    try:
        for tid in tqdm(PREFLIGHT_IDS,desc='40 paired boundary checks',unit='input'):
            if tid not in rows:
                rows[tid]=capture(INPUT_BY_ID[tid],preflight=True)
                checkpoint(FAMILY_ROOT,rows[tid]['family_id'],rows,RUNTIME_SHA)
            assert rows[tid]['replayed_candidate_logprobs'] is not None
        report,table=preflight_report(rows,PREFLIGHT_IDS,ARCHIVED_ROWS,ANALYSIS_DIR)
        display(table[['wording','stratum','raw_response','response_form','normalized_label','old_candidate_label','raw_exact_replay']])
        print(json.dumps(report,ensure_ascii=False,indent=2))
        print('Example exact prompt ending:',repr(rows[PREFLIGHT_IDS[0]]['prompt_tail_decoded']))
        trace=pd.read_csv(ANALYSIS_DIR/'preflight_token_trace.csv')
        example_ids=[next(t for t in PREFLIGHT_IDS if INPUT_BY_ID[t]['wording']==w) for w in WORDINGS]
        display(trace[trace.trial_id.isin(example_ids)])
        assert report['ready'],'An unrecognized/incomplete response or input mismatch needs inspection. Full generation is stopped; partial output has been saved. Do not relax parsing or select cases by accuracy.'
        freeze(PHASE_ROOT/'preflight_pass.json',dict(runtime_sha256=RUNTIME_SHA,ids=PREFLIGHT_IDS,
            report_sha256=file_hash(ANALYSIS_DIR/'preflight_report.json'),passed_at_rows_sha256=digest([rows[t] for t in PREFLIGHT_IDS])))
    except BaseException:
        path=export_run(RUN_ROOT,'preflight_incomplete_or_needs_review')
        print('Partial evidence ZIP:',path)
        raise
    return rows


def run_full():
    gate=read_json(PHASE_ROOT/'preflight_pass.json');assert gate['runtime_sha256']==RUNTIME_SHA
    rows=load_rows(FAMILY_ROOT,INPUT_BY_ID,RUNTIME_SHA)
    assert digest([rows[t] for t in PREFLIGHT_IDS])==gate['passed_at_rows_sha256']
    current_family=None
    try:
        for inp in tqdm(INPUTS,desc='1440 paired generations (cached rows reused)',unit='input'):
            if inp['trial_id'] in rows:continue
            if current_family is not None and current_family!=inp['family_id']:
                checkpoint(FAMILY_ROOT,current_family,rows,RUNTIME_SHA)
            current_family=inp['family_id']
            rows[inp['trial_id']]=capture(inp)
        if current_family is not None:checkpoint(FAMILY_ROOT,current_family,rows,RUNTIME_SHA)
    except BaseException:
        if current_family is not None:checkpoint(FAMILY_ROOT,current_family,rows,RUNTIME_SHA)
        print('Partial evidence ZIP:',export_run(RUN_ROOT,'partial'))
        raise
    assert len(rows)==1440
    return rows


def finish(rows):
    summary,comparisons=analyze(rows,INPUTS,ARCHIVED_ROWS,ANALYSIS_DIR)
    checks=read_json(ANALYSIS_DIR/'measurement_status.json')
    status='complete' if checks['normalized_invalid']==0 else 'complete_with_unresolved_responses'
    save_json(PHASE_ROOT/'completion.json',dict(status=status,n_generated=len(rows),n_families=96,
        runtime_sha256=RUNTIME_SHA,implementation=IMPLEMENTATION_SHA256,source_zip_sha256=SOURCE_SHA,
        completed_at_utc=now_utc(),family_files={p.name:file_hash(p) for p in sorted(FAMILY_ROOT.glob('*.json'))},
        analysis_files={p.name:file_hash(p) for p in sorted(ANALYSIS_DIR.glob('*')) if p.is_file()},
        meaning='Generation acquisition and analysis complete. Normalized wrappers are not ordinary direct letters; consult measurement_status and both metrics.'))
    display(summary[(summary.scope=='all') & summary.metric.isin(['normalized_correct','strict_correct','normalized_valid','strict_valid','wrapper'])].round(4))
    display(comparisons.round(4));display(pd.read_csv(ANALYSIS_DIR/'response_forms.csv'))
    print(json.dumps(checks,ensure_ascii=False,indent=2))
    return status
