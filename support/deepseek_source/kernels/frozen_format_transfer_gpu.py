"""GPU execution for a fixed codebook transfer protocol; never fits a predictor."""
from dataclasses import replace
import inspect
import shutil

_original_example = make_predictive_example
_original_trial_specs = trial_specs


def prepare_transfer_images():
    """Prefer original PNGs: identical pixels need not have identical PNG compression across platforms."""
    IMAGE_DIR.mkdir(parents=True,exist_ok=True)
    candidates=[]
    if SOURCE_IMAGES_OVERRIDE: candidates.append(Path(SOURCE_IMAGES_OVERRIDE))
    candidates += [prior_root/'images']
    if PRIOR.is_zip: candidates.append(PRIOR.path.parent/'images')
    provenance=[]
    for family in tqdm(FAMILIES,desc='Restore original scene images'):
        for role in family['roles'].values():
            name=role['image_filename']; target=IMAGE_DIR/name; expected=PRIOR.image_manifest[name]
            if target.exists():
                assert file_hash(target)==expected, 'Existing run image differs: '+str(target)
                provenance.append(dict(name=name,method='verified_existing')); continue
            original=next((root/name for root in candidates if (root/name).is_file() and file_hash(root/name)==expected),None)
            if original:
                shutil.copy2(original,target); method='verified_original_copy'
            else:
                scene=Scene(role['scene_id'],tuple(Stack(str(i),Side(s['side']),int(s['distance']),int(s['units']))
                                                 for i,s in enumerate(role['stacks'])))
                # Verify a temporary candidate before making it part of this run.
                temp=target.with_suffix('.tmp.png'); render_scene(scene,temp,RenderStyle(ticks_per_side=6))
                if file_hash(temp)!=expected:
                    temp.unlink()
                    raise RuntimeError('Rendered PNG bytes differ from the previous run. Keep the original images and set SOURCE_IMAGES_OVERRIDE to the old run images folder; do not loosen the checksum. Different Pillow/zlib versions can change PNG bytes.')
                temp.replace(target); method='verified_rerender'
            provenance.append(dict(name=name,method=method))
    save_json(PHASE_ROOT/'image_loading_provenance.json',provenance)


def make_format_example(family, role, task, mapping, codebook):
    ex=_original_example(family,role,task,mapping)
    if codebook=='ABC':
        return ex
    symbols=FORMAT_SYMBOLS[codebook]
    actual=canonical_mapping(mapping,codebook)
    options='\n'.join(f"{x['label']}. {SEMANTIC_TEXT[x['semantic']]}" for x in actual)
    noun='number' if codebook=='DIGITS' else 'letter'
    prompt='\n\n'.join(['Answer this question using only the supplied balance-scale scene.',RULE_TEXT,
        'Question: '+QUESTION_TEXT[task], 'Options:\n'+options,
        f'Do not explain your reasoning. Reply with exactly one option {noun} and nothing else. Valid replies are: '+', '.join(symbols)+'.'])
    return replace(ex,example_id=ex.example_id+'__'+codebook,question_id=ex.question_id+'__'+codebook,
                   prompt=prompt,labels=symbols,gold_label=symbols[LABELS.index(ex.gold_label)])


def format_trial_specs(family, codebook):
    for spec in _original_trial_specs(family):
        assert family['module']=='content'
        yield dict(spec,base_trial_id=spec['trial_id'],trial_id=spec['trial_id']+'__'+codebook,codebook=codebook)


class TailReadout:
    """Actual final norm and vocabulary head; no decoder blocks and no learned surrogate.

    Transformers Qwen3.5/Gemma4 implementations use norm -> lm_head, with Gemma
    applying config-defined final logit softcapping. Save installed source and
    report numerical replay error; do not silently call it an exact BF16 replay.
    """
    def __init__(self):
        parent=resolve_module_path(adapter.model,LAYER_PATH.rsplit('.',1)[0])
        self.norm=parent.norm
        self.head=adapter.model.get_output_embeddings()
        assert self.head is adapter.model.lm_head and isinstance(self.head,torch.nn.Linear)
        self.config=adapter.model.config.get_text_config()
        self.cap=getattr(self.config,'final_logit_softcapping',None)
        self.device=next(self.norm.parameters()).device
        self.head_device=self.head.weight.device
        self.dtype=self.head.weight.dtype
        assert self.dtype==torch.bfloat16
        self.plans={}
        for cb in FORMAT_SYMBOLS:
            ex=make_format_example(FAMILIES[0],'base','outcome',mapping_reference(FAMILIES[0]['family_id']),cb)
            plan=adapter._candidate_plan(ex)
            assert plan is not None, f'{cb} must have three single-token continuations after one shared prefix.'
            prefix,tokens=plan
            assert set(tokens)==set(FORMAT_SYMBOLS[cb]) and len(set(tokens.values()))==3
            self.plans[cb]=dict(prefix=list(prefix),tokens=tokens)
        # A changed scoring position would be an extra manipulation beyond the codebook.
        assert len({tuple(p['prefix']) for p in self.plans.values()})==1, 'Codebooks have different scoring prefixes; inspect the tokenizer protocol before continuing.'
        self.identity=dict(norm_class=type(self.norm).__name__,head_class=type(self.head).__name__,
            norm_path=LAYER_PATH.rsplit('.',1)[0]+'.norm',final_logit_softcapping=self.cap,
            plans=self.plans,model_type=self.config.model_type,head_batch_size=1,
            interpretation='output-path reconstruction, not a learned forecast or physical-mechanism test')
        source='\n\n'.join(inspect.getsource(type(m).forward) for m in (parent,adapter.model,self.norm))
        freeze(PHASE_ROOT/'tail_readout_identity.json',dict(self.identity,installed_source_sha256=hashlib.sha256(source.encode()).hexdigest()))
        (PHASE_ROOT/'installed_tail_source.txt').write_text(source,encoding='utf-8')
        # Export only the small reproducibility subset, never full model weights.
        arrays={'norm_weight':self.norm.weight.detach().float().cpu().numpy()}
        for cb,plan in self.plans.items():
            ids=[plan['tokens'][s] for s in FORMAT_SYMBOLS[cb]]
            arrays[cb+'_head_rows']=self.head.weight[ids].detach().float().cpu().numpy()
            if self.head.bias is not None: arrays[cb+'_head_bias']=self.head.bias[ids].detach().float().cpu().numpy()
        p=PHASE_ROOT/'tail_parameters.npz'; tmp=p.with_suffix('.tmp.npz'); np.savez_compressed(tmp,**arrays); tmp.replace(p)
        freeze(PHASE_ROOT/'tail_parameters_identity.json',dict(sha256=file_hash(p),norm_eps=float(self.norm.eps)))

    def scores(self, hidden, codebook):
        h=torch.as_tensor(np.asarray(hidden).copy(),dtype=self.dtype,device=self.device).reshape(1,1,-1)
        with torch.inference_mode():
            logits=self.head(self.norm(h).to(self.head_device))
            if self.cap is not None:
                logits=torch.tanh(logits/self.cap)*self.cap
            ids=[self.plans[codebook]['tokens'][s] for s in FORMAT_SYMBOLS[codebook]]
            scores=logits[0,-1,ids].float().cpu().numpy()
        # Centered log probabilities equal centered logits; no full-vocabulary softmax needed.
        return center(scores)


def run_archive_tail_audit():
    folder=PHASE_ROOT/'archive_tail_audit'; folder.mkdir(parents=True,exist_ok=True)
    all_rows=[]; last=EXPECTED_LAYER_COUNT-1
    for family in tqdm(FAMILIES,desc='Replay archived final states through output modules'):
        fid=family['family_id']; pre=read_json(PRIOR_CACHE/(fid+'.json'))
        saved=folder/(fid+'.json')
        identity=digest([RUNTIME_IDENTITY,pre['arrays_sha256'],TAIL.identity,
                         SOURCE_IDENTITY['prior_files_sha256'],IMPLEMENTATION_SHA256])
        if saved.exists():
            obj=read_json(saved); assert obj['identity']==identity
            all_rows.extend(obj['rows']); continue
        records=[]; cache={}
        with np.load(PRIOR_CACHE/(fid+'.npz'),allow_pickle=False) as arrays:
            for row in OLD_ROWS_BY_FAMILY[fid]:
                if row['layer']!=last: continue
                h0=arrays[row['pre_key']]; delta=arrays[row['payload_key']]
                assert state_hash(h0)==row['pre_sha256'] and state_hash(delta)==row['payload_sha256']
                if row['pre_key'] not in cache: cache[row['pre_key']]=TAIL.scores(h0,'ABC')
                before=cache[row['pre_key']]; after=TAIL.scores(bf16(h0+delta),'ABC')
                actual_before=np.asarray(row['s0']); actual_after=actual_before+np.asarray(row['actual_delta'])
                records.append(dict(trial_id=row['trial_id'],family_id=fid,
                    before_max_abs_error=float(np.max(np.abs(before-actual_before))),
                    after_max_abs_error=float(np.max(np.abs(after-actual_after))),
                    delta_mse=float(np.mean(((after-before)-row['actual_delta'])**2)),
                    post_answer_match=int(np.argmax(after)==np.argmax(actual_after)),
                    actual_change=int(np.argmax(actual_before)!=np.argmax(actual_after)),
                    reconstructed_before=before.tolist(),reconstructed_after=after.tolist()))
        save_json(saved,dict(identity=identity,rows=records)); all_rows.extend(records)
    assert len(all_rows)==576
    df=pd.DataFrame(all_rows); df.to_csv(PHASE_ROOT/'analysis/archive_tail_replay.csv',index=False)
    print('Archived tail reconstruction; numerical errors are reported, never auto-corrected:')
    display(df[['before_max_abs_error','after_max_abs_error','delta_mse','post_answer_match']].describe())
    save_json(PHASE_ROOT/'archive_tail_summary.json',dict(n_trials=len(df),
        max_before_error=float(df.before_max_abs_error.max()),max_after_error=float(df.after_max_abs_error.max()),
        mean_delta_mse=float(df.delta_mse.mean()),post_answer_match=float(df.post_answer_match.mean()),
        status='reconstruction_reported_not_a_physical_mechanism_test',
        caveat='Replay uses actual modules but token-only GEMM shapes and potentially a different GPU; compare errors, do not assume bitwise equality.'))


def capture_format_family(family, codebook):
    folder=PHASE_ROOT/'pre_intervention'; folder.mkdir(parents=True,exist_ok=True)
    stem=family['family_id']+'__'+codebook
    path=folder/(stem+'.json'); archive=folder/(stem+'.npz')
    identity=digest([RUNTIME_IDENTITY,family,codebook,IMPLEMENTATION_SHA256])
    if path.exists():
        obj=read_json(path)
        assert obj['identity']==identity and file_hash(archive)==obj['arrays_sha256']
        return obj
    arrays={}; records={}; examples={}; rows=[]; tail_clean=[]
    def capture(role,task,mapping):
        ex=make_format_example(family,role,task,mapping,codebook)
        key=digest(ex.example_id)[:20]
        if key not in records:
            rec,hidden=predictive_forward(ex)
            records[key]=rec
            examples[key]=dict(role=role,task=task,canonical_mapping=mapping,
                actual_mapping=canonical_mapping(mapping,codebook),stamp=example_stamp(ex))
            for layer,h in hidden.items(): arrays[key+f'__L{layer}']=h
            tail=TAIL.scores(hidden[EXPECTED_LAYER_COUNT-1],codebook)
            tail_clean.append(dict(example_key=key,max_abs_error=float(np.max(np.abs(tail-center(rec['candidate_logprobs'])))),
                                   scores=tail.tolist(),actual=center(rec['candidate_logprobs']).tolist()))
        return key
    for spec in format_trial_specs(family,codebook):
        sb=capture(spec['source_before'],spec['task'],spec['source_mapping'])
        sa=capture(spec['source_after'],spec['task'],spec['source_mapping'])
        rb=capture(spec['recipient'],'outcome',spec['recipient_mapping']); layer=spec['layer']
        h0=arrays[rb+f'__L{layer}']; delta=bf16(arrays[sa+f'__L{layer}']-arrays[sb+f'__L{layer}'])
        norm=float(np.linalg.norm(bf16(h0+delta)-h0)); payload_key=spec['trial_id']+'__delta'
        arrays[payload_key]=delta
        gold=_original_example(family,spec['recipient'],'outcome',spec['recipient_mapping']).gold_label
        rows.append(dict(spec,family_id=family['family_id'],module='content',split='validation',stratum=family['stratum'],
            gold=gold,source_key_before=sb,source_key_after=sa,recipient_key=rb,payload_key=payload_key,pre_key=rb+f'__L{layer}',
            payload_sha256=state_hash(delta),pre_sha256=state_hash(h0),source_delta=center(np.asarray(records[sa]['candidate_logprobs'])-records[sb]['candidate_logprobs']).tolist(),
            source_before_argmax=int(np.argmax(records[sb]['candidate_logprobs'])),source_after_argmax=int(np.argmax(records[sa]['candidate_logprobs'])),
            s0=center(records[rb]['candidate_logprobs']).tolist(),planned_norm=norm,relative_norm=norm/max(float(np.linalg.norm(h0)),1e-12)))
    temp=archive.with_suffix('.tmp.npz'); np.savez_compressed(temp,**arrays); temp.replace(archive)
    obj=dict(identity=identity,arrays_sha256=file_hash(archive),family_id=family['family_id'],codebook=codebook,
             examples=examples,baseline_records=records,rows=rows,tail_clean=tail_clean,created_at_utc=utc_now())
    save_json(path,obj); return obj


def capture_all_formats():
    rows=[]; manifest={}
    for codebook in FORMAT_SYMBOLS:
        for family in tqdm(FAMILIES,desc=f'{codebook}: source and untreated inputs'):
            obj=capture_format_family(family,codebook); rows.extend(obj['rows'])
            name=family['family_id']+'__'+codebook+'.json'
            manifest[name]=file_hash(PHASE_ROOT/'pre_intervention'/name)
    assert len(rows)==3456 and len({r['trial_id'] for r in rows})==3456
    freeze(PHASE_ROOT/'pre_intervention_manifest.json',manifest)
    return rows


def freeze_format_predictions(rows):
    identity=dict(runtime_sha256=file_hash(PHASE_ROOT/'runtime_identity.json'),
        frozen_models_sha256=file_hash(MODEL_ROOT/'frozen_predictors.json'),
        pre_manifest_sha256=file_hash(PHASE_ROOT/'pre_intervention_manifest.json'),
        protocol_sha256=file_hash(RUN_ROOT/'protocol.json'),source_identity_sha256=file_hash(MODEL_ROOT/'source_identity.json'))
    sha=seal_predictions(PHASE_ROOT/'predictions_before_intervention.json',rows,FITTED,identity)
    freeze(PHASE_ROOT/'prediction_seal.json',dict(identity,prediction_sha256=sha))
    print('All ABC/XYZ/DIGITS forecasts sealed:',sha)


def verify_format_seal():
    seal=read_json(PHASE_ROOT/'prediction_seal.json')
    for key,path in [('prediction_sha256',PHASE_ROOT/'predictions_before_intervention.json'),
                     ('runtime_sha256',PHASE_ROOT/'runtime_identity.json'),
                     ('frozen_models_sha256',MODEL_ROOT/'frozen_predictors.json'),
                     ('pre_manifest_sha256',PHASE_ROOT/'pre_intervention_manifest.json'),
                     ('protocol_sha256',RUN_ROOT/'protocol.json'),
                     ('source_identity_sha256',MODEL_ROOT/'source_identity.json')]:
        assert file_hash(path)==seal[key], 'Sealed input changed: '+str(path)
    return seal


def execute_format_targets():
    seal=verify_format_seal(); sealed_hash=digest(seal)
    manifest=read_json(PHASE_ROOT/'pre_intervention_manifest.json')
    folder=PHASE_ROOT/'interventions'; folder.mkdir(parents=True,exist_ok=True)
    all_rows=[]; tail=[]
    for cb in FORMAT_SYMBOLS:
        for family in tqdm(FAMILIES,desc=f'{cb}: fixed interventions'):
            stem=family['family_id']+'__'+cb
            path=PHASE_ROOT/'pre_intervention'/(stem+'.json'); obj=read_json(path)
            assert file_hash(path)==manifest[path.name] and file_hash(path.with_suffix('.npz'))==obj['arrays_sha256']
            with np.load(path.with_suffix('.npz'),allow_pickle=False) as arrays:
                for row in obj['rows']:
                    dest=folder/(row['trial_id']+'.json')
                    identity=digest([RUNTIME_IDENTITY,row,sealed_hash,obj['arrays_sha256']])
                    if dest.exists():
                        saved=read_json(dest); assert saved['identity']==identity
                    else:
                        ex=make_format_example(family,row['recipient'],'outcome',row['recipient_mapping'],cb)
                        baseline=obj['baseline_records'][row['recipient_key']]
                        payload=arrays[row['payload_key']]; pre=arrays[row['pre_key']]
                        assert state_hash(payload)==row['payload_sha256'] and state_hash(pre)==row['pre_sha256']
                        # Source-free terminal replay is computed before the real patch, but is
                        # a deterministic output-path reconstruction, not a competing fitted predictor.
                        head_scores=TAIL.scores(bf16(pre+payload),cb).tolist() if row['layer']==EXPECTED_LAYER_COUNT-1 else None
                        record,_=predictive_forward(ex,layer=row['layer'],payload=payload,
                            expected_pre=pre,expected_input=baseline['input_sha256'])
                        saved=dict(identity=identity,record=record,tail_scores=head_scores,
                            sealed_prediction_identity=sealed_hash,created_at_utc=utc_now())
                        save_json(dest,saved)
                    actual=center(saved['record']['candidate_logprobs'])-row['s0']
                    all_rows.append(dict(row,actual_delta=actual.tolist(),post_record=saved['record']))
                    if saved['tail_scores'] is not None:
                        tail.append(dict(trial_id=row['trial_id'],family_id=row['family_id'],codebook=cb,
                            max_abs_error=float(np.max(np.abs(np.asarray(saved['tail_scores'])-center(saved['record']['candidate_logprobs'])))),
                            post_answer_match=int(np.argmax(saved['tail_scores'])==np.argmax(saved['record']['candidate_logprobs']))))
    assert len(all_rows)==3456 and len({r['trial_id'] for r in all_rows})==3456
    save_json(PHASE_ROOT/'target_rows.json',all_rows)
    pd.DataFrame(tail).to_csv(PHASE_ROOT/'analysis/current_tail_replay.csv',index=False)
    return all_rows


def compare_current_abc_to_archive(rows):
    old={r['trial_id']:r for r in PRIOR.old_rows}; reports=[]
    for r in rows:
        if r['codebook']!='ABC': continue
        prior=old[r['base_trial_id']]
        reports.append(dict(trial_id=r['base_trial_id'],family_id=r['family_id'],layer=r['layer'],
            baseline_max_abs_error=float(np.max(np.abs(np.asarray(r['s0'])-prior['s0']))),
            effect_max_abs_error=float(np.max(np.abs(np.asarray(r['actual_delta'])-prior['actual_delta']))),
            before_state_identical=r['pre_sha256']==prior['pre_sha256'],source_delta_identical=r['payload_sha256']==prior['payload_sha256']))
    pd.DataFrame(reports).to_csv(PHASE_ROOT/'analysis/archived_vs_current_ABC.csv',index=False)
    print('Cross-format comparisons use the fresh ABC anchor. Archive/runtime differences are reported separately.')


def export_transfer():
    marker=PHASE_ROOT/'completion.json'
    status=read_json(marker)['status'] if marker.exists() else 'partial_not_complete'
    # Include the old evidence subset actually used, without duplicating unrelated physical trials.
    snapshot=MODEL_ROOT/'source_snapshot'; snapshot.mkdir(parents=True,exist_ok=True)
    for p in PRIOR_CACHE.glob('*'):
        if p.is_file() and p.suffix in ('.json','.npz'):
            dest=snapshot/p.name
            if not dest.exists() or file_hash(dest)!=file_hash(p): shutil.copy2(p,dest)
    selected=[RUN_ROOT/'protocol.json',RUN_ROOT/'support.json']
    selected += [p for p in MODEL_ROOT.rglob('*') if p.is_file() and '.tmp' not in p.name and p.suffix!='.zip']
    manifest={str(p.relative_to(RUN_ROOT)):file_hash(p) for p in sorted(set(selected))}
    archive=RUN_ROOT/f'VisualTorque_Frozen_Format_Transfer_{MODEL_KEY}_Output.zip'
    temp=archive.with_suffix('.tmp.zip')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for name in manifest: z.write(RUN_ROOT/name,name)
        z.writestr('export_manifest.json',json.dumps(dict(status=status,files=manifest,created_at_utc=utc_now()),indent=2))
    temp.replace(archive)
    print('Saved:',archive,'|',status)
    try:
        from google.colab import files
        files.download(str(archive))
    except Exception:
        print('Browser download unavailable. The ZIP remains at the Drive path above.')
    return archive
