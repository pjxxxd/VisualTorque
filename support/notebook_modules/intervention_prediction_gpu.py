"""GPU execution, pre-intervention sealing and exports for the prospective study."""
import shutil
import zipfile
from visual_torque_toolkit.behavior import PromptCondition
from visual_torque_toolkit.behavior.data import BehaviorExample
from visual_torque_toolkit.render import RenderStyle, render_scene
from visual_torque_toolkit.world import Scene, Side, Stack

RULE_TEXT = ('Each square is one unit of weight. Distance is measured in tick '
             'intervals from the center pivot. Torque equals weight times distance. '
             'Total torque on a side is the sum over stacks on that side.')
QUESTION_TEXT = {'outcome':'Which way will the balance scale tip?',
                 'weight_more':'Which side has more total weight?',
                 'weight_less':'Which side has less total weight?'}
SEMANTIC_TEXT = {'left':'left side','balanced':'equal','right':'right side'}


def question_gold(role, task):
    f = role['features']
    value = f['net_torque'] if task == 'outcome' else f['weight_score']*(1 if task == 'weight_more' else -1)
    return 'left' if value>0 else 'right' if value<0 else 'balanced'


def make_predictive_example(family, role_name, task, mapping):
    role = family['roles'][role_name]
    sem = [next(x['semantic'] for x in mapping if x['label']==label) for label in LABELS]
    options = '\n'.join(f"{x['label']}. {SEMANTIC_TEXT[x['semantic']]}" for x in mapping)
    prompt = '\n\n'.join(['Answer this question using only the supplied balance-scale scene.',
        RULE_TEXT, 'Question: '+QUESTION_TEXT[task], 'Options:\n'+options,
        'Do not explain your reasoning. Reply with exactly one option letter and nothing else. Valid replies are: A, B, C.'])
    gold = question_gold(role,task)
    eid = family['family_id']+'__'+role_name+'__'+task+'__'+digest(mapping)[:12]
    return BehaviorExample(example_id=eid,question_id=eid,scene_id=role['scene_id'],
        condition=PromptCondition.IMAGE_RULE,question_kind=task,variant=SCHEMA,
        split=family['split'],problem_type='controlled_four_stack_family',outcome=gold,
        net_torque=int(role['features']['net_torque']),prompt=prompt,
        image_path=IMAGE_DIR/role['image_filename'],labels=LABELS,
        option_text=tuple(SEMANTIC_TEXT[s] for s in sem),option_semantics=tuple(sem),
        gold_label=LABELS[sem.index(gold)],gold_semantic=gold)


def prepared_fingerprint(prepared):
    tensors = {}
    for key, value in prepared['inputs'].items():
        if torch.is_tensor(value):
            x = value.detach().cpu().contiguous()
            raw = x.view(torch.uint8).numpy().tobytes()
            tensors[key] = dict(shape=list(x.shape),dtype=str(x.dtype),sha256=hashlib.sha256(raw).hexdigest())
        else:
            tensors[key] = str(value)
    return digest(dict(inputs=tensors,candidate_tokens=prepared['token_by_label']))


def predictive_forward(example, layer=None, payload=None, random_key=None, expected_pre=None, expected_input=None):
    prepared = prepare_examples([example])
    fingerprint = prepared_fingerprint(prepared)
    if expected_input is not None:
        assert fingerprint == expected_input, 'Prepared prompt/image tensors changed.'
    mask = final_mask(prepared)
    captured, audits, handles = {}, {}, []
    active = CAPTURE_LAYERS if layer is None else (int(layer),)
    for index in active:
        def hook(_module,_inputs,output,index=index):
            hidden = output_hidden_tensor(output)
            m = mask.to(hidden.device)
            assert hidden.dtype == torch.bfloat16 and int(m.sum()) == 1
            pre = hidden[0,m]
            before = pre.detach().float().cpu().numpy().copy()
            if payload is not None:
                assert index == layer
                # Check the current untreated state against the state used to plan this write.
                assert expected_pre is not None
                max_error = float(np.max(np.abs(before-np.asarray(expected_pre))))
                assert max_error == 0., (
                    f'Untreated-state replay differs ({max_error}). Do not loosen the check; '
                    'inspect runtime/input identity and restart this run consistently.')
                delta = torch.as_tensor(np.asarray(payload).copy(),device=pre.device,dtype=pre.dtype).reshape(pre.shape)
                desired = float(((pre+delta).float()-pre.float()).norm())
                if random_key is None:
                    post = pre+delta
                    audit = dict(matched=True,relative_error=0.,target_norm=desired)
                else:
                    cast = lambda x:torch.as_tensor(np.asarray(x).copy(),dtype=torch.bfloat16).float().numpy()
                    post_np,audit = native_norm_edit(before,random_unit(before.shape,random_key),desired,cast)
                    post = torch.as_tensor(post_np,device=pre.device,dtype=pre.dtype)
                audit.update(actual_norm=float((post.float()-pre.float()).norm()),
                             pre_sha256=state_hash(before), planned_payload_sha256=state_hash(payload))
                audits[str(index)] = audit
                modified = hidden.clone(); modified[0,m] = post
                captured[index] = post.detach().float().cpu().numpy().copy()
                return replace_output_hidden(output,modified)
            captured[index] = before
            return None
        handles.append(DECODER_LAYERS[index].register_forward_hook(hook))
    try:
        with torch.inference_mode():
            out = adapter.model(**prepared['inputs'],use_cache=False)
        record = score_records(prepared,out.logits)[0]
    finally:
        for handle in handles:
            handle.remove()
    record.update(input_sha256=fingerprint,write_audits=audits)
    return record,captured


def trial_specs(family):
    source_mapping = mapping_reference(family['family_id'])
    if family['module'] == 'content':
        routes = [dict(source_before='base',source_after='aligned_source',recipient='base',
                       task=task,tag=task) for task in ('weight_more','weight_less')]
    else:
        routes = [dict(source_before=f'source_{ds}_before',source_after=f'source_{ds}_after',
                       recipient=f'recipient_{dr}_before',cf_role=f'recipient_{dr}_after',
                       task='outcome',tag=f'{ds}_{dr}',source_distance=family[ds],
                       recipient_distance=family[dr])
                  for ds in ('near','far') for dr in ('near','far')]
    for route in routes:
        if family['module'] == 'content':
            focus = question_gold(family['roles'][route['source_after']],route['task'])
        else:
            side_sign = 1 if family['target_side']=='left' else -1
            focus = 'left' if family['dw']*side_sign>0 else 'right'
        mappings = [('identity',source_mapping),
                    ('conflict_forward',mapping_conflict(source_mapping,focus)),
                    ('conflict_reverse',mapping_conflict(source_mapping,focus,True))]
        for mapping_name,mapping in mappings:
            for layer in SPEC[family['module']+'_layers']:
                tid = family['family_id']+f"__{route['tag']}__{mapping_name}__L{layer}"
                yield dict(route,trial_id=tid,layer=layer,source_mapping=source_mapping,
                           recipient_mapping=mapping,mapping_name=mapping_name)


def render_support(families):
    IMAGE_DIR.mkdir(parents=True,exist_ok=True)
    for f in tqdm(families,desc='Render frozen support'):
        for r in f['roles'].values():
            path = IMAGE_DIR/r['image_filename']
            if path.exists():
                continue
            scene = Scene(r['scene_id'],tuple(Stack(str(i),Side(s['side']),int(s['distance']),int(s['units']))
                                              for i,s in enumerate(r['stacks'])))
            render_scene(scene,path,RenderStyle(ticks_per_side=6))


def capture_family(family):
    """Donors, untreated recipients and a separately labelled extra-input benchmark.

    Natural CF images are queried before sealing, but enter ONLY natural_response.
    They are not hidden-state interventions or inputs to the other predictors.
    """
    folder = PHASE_ROOT/'pre_intervention'
    folder.mkdir(parents=True,exist_ok=True)
    path = folder/(family['family_id']+'.json')
    archive = folder/(family['family_id']+'.npz')
    identity = digest([RUNTIME_IDENTITY,family])
    if path.exists():
        obj = read_json(path)
        assert obj['identity'] == identity and file_hash(archive) == obj['arrays_sha256']
        return obj
    arrays, examples, records, rows = {}, {}, {}, []

    def capture(role,task,mapping):
        ex = make_predictive_example(family,role,task,mapping)
        key = digest(ex.example_id)[:20]
        if key not in examples:
            rec, hidden = predictive_forward(ex)
            examples[key] = dict(role=role,task=task,mapping=mapping,stamp=example_stamp(ex))
            records[key] = rec
            for layer,h in hidden.items():
                arrays[key+f'__L{layer}'] = h
        return key

    for spec in trial_specs(family):
        sb = capture(spec['source_before'],spec['task'],spec['source_mapping'])
        sa = capture(spec['source_after'],spec['task'],spec['source_mapping'])
        rb = capture(spec['recipient'],'outcome',spec['recipient_mapping'])
        layer = spec['layer']
        h0 = arrays[rb+f'__L{layer}']
        # Exact raw native-BF16 subtraction and unit dose; no layer-content label
        # is automatically inherited by this physical-difference operation.
        delta = bf16(arrays[sa+f'__L{layer}']-arrays[sb+f'__L{layer}'])
        realized = bf16(h0+delta)-h0
        norm = float(np.linalg.norm(realized))
        array_key = spec['trial_id']+'__delta'
        arrays[array_key] = delta
        gold = make_predictive_example(family,spec['recipient'],'outcome',spec['recipient_mapping']).gold_label
        row = dict(spec,family_id=family['family_id'],module=family['module'],split=family['split'],
            stratum=family['stratum'],gold=gold,source_key_before=sb,source_key_after=sa,
            recipient_key=rb,payload_key=array_key,pre_key=rb+f'__L{layer}',
            payload_sha256=state_hash(delta),pre_sha256=state_hash(h0),
            source_delta=center(np.array(records[sa]['candidate_logprobs'])-records[sb]['candidate_logprobs']).tolist(),
            source_before_argmax=int(np.argmax(records[sb]['candidate_logprobs'])),
            source_after_argmax=int(np.argmax(records[sa]['candidate_logprobs'])),
            s0=center(records[rb]['candidate_logprobs']).tolist(),
            planned_norm=norm,relative_norm=norm/max(float(np.linalg.norm(h0)),1e-12),
            random_control_family=family['random_control_family'])
        if family['module']=='physical':
            f = family['roles'][spec['recipient']]['features']
            row.update(dw=family['dw'],target_side=family['target_side'],
                       baseline_physics=[f['net_torque'],f['normalized_margin'],f['weight_score'],f['distance_score']])
            row['cf_gold'] = make_predictive_example(family,spec['cf_role'],'outcome',spec['recipient_mapping']).gold_label
            cf_key=capture(spec['cf_role'],'outcome',spec['recipient_mapping'])
            row['natural_input_key']=cf_key
            row['natural_input_delta']=(center(records[cf_key]['candidate_logprobs'])-np.asarray(row['s0'])).tolist()
        rows.append(row)
    # Numpy arrays are data, not pickled Python. Marker written only after the archive.
    temp = archive.with_suffix('.tmp.npz')
    np.savez_compressed(temp,**arrays); temp.replace(archive)
    obj = dict(identity=identity,arrays_sha256=file_hash(archive),family=family,examples=examples,
               baseline_records=records,rows=rows,created_at_utc=utc_now())
    save_json(path,obj)
    return obj


def pre_intervention_rows(families):
    rows, manifest = [], {}
    for family in tqdm(families,desc='Capture donor and untreated recipient states'):
        obj = capture_family(family)
        rows.extend(obj['rows'])
        p = PHASE_ROOT/'pre_intervention'/(family['family_id']+'.json')
        manifest[p.name] = file_hash(p)
    freeze(PHASE_ROOT/'pre_intervention_manifest.json',manifest)
    return rows


def verify_prediction_seal():
    if PHASE != 'validation':
        return None
    path = PHASE_ROOT/'predictions_before_intervention.json'
    assert path.is_file(), 'Save all predictions before running any validation interventions.'
    marker = read_json(PHASE_ROOT/'prediction_seal.json')
    assert file_hash(path) == marker['prediction_sha256']
    assert marker['frozen_models_sha256'] == file_hash(MODEL_ROOT/'frozen_predictors.json')
    assert marker['pre_manifest_sha256'] == file_hash(PHASE_ROOT/'pre_intervention_manifest.json')
    return marker


def execute_targets(families):
    seal = verify_prediction_seal()
    expected_seal = digest(seal) if seal is not None else 'calibration-only'
    folder = PHASE_ROOT/'interventions'; folder.mkdir(parents=True,exist_ok=True)
    all_rows, control_rows, natural_rows = [], [], []
    for family in tqdm(families,desc='Validate fixed interventions'):
        pre_path = PHASE_ROOT/'pre_intervention'/(family['family_id']+'.json')
        obj = read_json(pre_path)
        archive = pre_path.with_suffix('.npz')
        assert file_hash(archive)==obj['arrays_sha256']
        assert file_hash(pre_path)==read_json(PHASE_ROOT/'pre_intervention_manifest.json')[pre_path.name]
        with np.load(archive,allow_pickle=False) as arrays:
            for row in tqdm(obj['rows'],desc=family['family_id'],leave=False):
                path = folder/(row['trial_id']+'.json')
                identity = digest([RUNTIME_IDENTITY,row,expected_seal,obj['arrays_sha256']])
                if path.exists():
                    saved = read_json(path)
                    assert saved['identity']==identity, 'Intervention cache identity changed.'
                else:
                    ex = make_predictive_example(family,row['recipient'],'outcome',row['recipient_mapping'])
                    baseline = obj['baseline_records'][row['recipient_key']]
                    kwargs = dict(layer=row['layer'],payload=arrays[row['payload_key']],
                                  expected_pre=arrays[row['pre_key']],expected_input=baseline['input_sha256'])
                    assert state_hash(kwargs['payload'])==row['payload_sha256']
                    record,_ = predictive_forward(ex,**kwargs)
                    control = []
                    if family['random_control_family']:
                        for seed in RANDOM_SEEDS:
                            rec,_ = predictive_forward(ex,random_key=[SEED,row['trial_id'],seed],**kwargs)
                            control.append(dict(seed=seed,record=rec))
                    natural = None
                    if family['module']=='physical':
                        natural = obj['baseline_records'][row['natural_input_key']]
                    saved = dict(identity=identity,record=record,controls=control,natural_cf=natural,
                                 sealed_prediction_identity=expected_seal,created_at_utc=utc_now())
                    save_json(path,saved)
                record = saved['record']
                actual = center(record['candidate_logprobs'])-np.asarray(row['s0'])
                all_rows.append(dict(row,actual_delta=actual.tolist(),post_record=record))
                for control in saved['controls']:
                    rec = control['record']; audit = rec['write_audits'][str(row['layer'])]
                    control_rows.append(dict(trial_id=row['trial_id'],family_id=row['family_id'],
                        operation=operation_key(row),seed=control['seed'],**audit,
                        actual_delta=(center(rec['candidate_logprobs'])-np.asarray(row['s0'])).tolist()))
                if saved['natural_cf'] is not None:
                    cf = saved['natural_cf']
                    natural_rows.append(dict(trial_id=row['trial_id'],family_id=row['family_id'],
                        operation=operation_key(row),mapping=row['mapping_name'],
                        source_distance=row['source_distance'],recipient_distance=row['recipient_distance'],
                        baseline_gold=row['gold'],cf_gold=row['cf_gold'],
                        baseline_answer=LABELS[int(np.argmax(row['s0']))],
                        patched_answer=record['predicted_label'],natural_answer=cf['predicted_label'],
                        natural_delta=(center(cf['candidate_logprobs'])-np.asarray(row['s0'])).tolist(),
                        patch_delta=actual.tolist()))
    expected = sum(len(list(trial_specs(f))) for f in families)
    assert len(all_rows)==expected and len({r['trial_id'] for r in all_rows})==expected
    save_json(PHASE_ROOT/'target_rows.json',all_rows)
    save_json(PHASE_ROOT/'random_control_rows.json',control_rows)
    save_json(PHASE_ROOT/'natural_counterfactual_rows.json',natural_rows)
    return all_rows


def analyze_controls_and_counterfactuals():
    out = PHASE_ROOT/'analysis'; out.mkdir(parents=True,exist_ok=True)
    target = {r['trial_id']:r for r in read_json(PHASE_ROOT/'target_rows.json')}
    controls = read_json(PHASE_ROOT/'random_control_rows.json')
    pd.DataFrame(controls).to_csv(out/'random_native_norm_audit.csv',index=False)
    comparisons = []
    for c in controls:
        row = target[c['trial_id']]
        if row['module']=='physical':
            axis = np.asarray(predictor_features(row)['recipient_weight'])
        else:
            axis = np.asarray(predictor_features(row)['source_semantic'])
        norm = np.linalg.norm(axis)
        if norm:
            value = float(np.dot(np.asarray(row['actual_delta'])-c['actual_delta'],axis/norm))
        else:
            value = None
        comparisons.append(dict(operation=c['operation'],family_id=c['family_id'],seed=c['seed'],
                                matched=c['matched'],target_minus_random_projection=value))
    frame = pd.DataFrame(comparisons)
    frame.to_csv(out/'random_specificity_trials.csv',index=False)
    # Show all attempts; unmatched controls never silently count as valid equal-norm evidence.
    control_tests = []
    for op,g in frame.groupby('operation'):
        for scope in ('all_attempts_descriptive','all_controls_matched_families'):
            if scope.endswith('families'):
                valid = g.groupby('family_id').matched.all()
                sub = g[g.family_id.isin(valid[valid].index)]
            else: sub=g
            means = sub.groupby('family_id').target_minus_random_projection.mean().dropna()
            if len(means):
                control_tests.append(dict(operation=op,scope=scope,
                    **paired_interval(means.values,[op,scope,'random'])))
    pd.DataFrame(control_tests).to_csv(out/'random_specificity_secondary.csv',index=False)
    natural = read_json(PHASE_ROOT/'natural_counterfactual_rows.json')
    for r in natural:
        r['natural_cf_correct'] = int(r['natural_answer']==r['cf_gold'])
        r['patched_cf_correct'] = int(r['patched_answer']==r['cf_gold'])
        r['baseline_cf_correct'] = int(r['baseline_answer']==r['cf_gold'])
        r['baseline_original_correct'] = int(r['baseline_answer']==r['baseline_gold'])
        r['both_natural_endpoints_correct'] = r['natural_cf_correct']*r['baseline_original_correct']
        r['patch_matches_natural_answer'] = int(r['patched_answer']==r['natural_answer'])
        r['patch_natural_delta_mse'] = float(np.mean((np.asarray(r['patch_delta'])-r['natural_delta'])**2))
        r['cf_flip_by_physics'] = r['baseline_gold'] != r['cf_gold']
    df = pd.DataFrame(natural)
    df.to_csv(out/'natural_counterfactual_trials.csv',index=False)
    columns = ['natural_cf_correct','patched_cf_correct','baseline_cf_correct',
               'baseline_original_correct','both_natural_endpoints_correct',
               'patch_matches_natural_answer','patch_natural_delta_mse']
    if len(df):
        fam = df.groupby(['operation','family_id'])[columns].mean().reset_index()
        fam.to_csv(out/'natural_counterfactual_family_metrics.csv',index=False)
        fam.groupby('operation')[columns].mean().to_csv(out/'natural_counterfactual_summary.csv')
        df.groupby(['operation','source_distance','recipient_distance','family_id'])[columns].mean().groupby(
            ['operation','source_distance','recipient_distance']).mean().to_csv(out/'distance_factorial_secondary.csv')
    return dict(random_writes=len(controls),unmatched_random_writes=sum(not c['matched'] for c in controls))


def export_phase(status):
    """Include support, frozen models, sealed predictions, compact states and raw outputs."""
    completion=PHASE_ROOT/'completion.json'
    status=read_json(completion)['status'] if completion.is_file() else 'partial_capture_not_complete'
    manifest = {}
    candidates = [SUPPORT_PATH,RUN_ROOT/'protocol.json',RUN_ROOT/'historical_geometry_snapshot.json',
                  RUN_ROOT/'image_manifest.json',MODEL_ROOT/'runtime_identity.json',
                  MODEL_ROOT/'frozen_predictors.json']
    candidates += [p for p in PHASE_ROOT.rglob('*') if p.is_file() and not p.name.endswith('.tmp')]
    for p in sorted(set(candidates)):
        if p.is_file():
            manifest[str(p.relative_to(RUN_ROOT))] = file_hash(p)
    archive = RUN_ROOT/f'VisualTorque_Intervention_Prediction_{MODEL_KEY}_{PHASE}_Output.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in manifest:
            z.write(RUN_ROOT/name,name)
        z.writestr('export_manifest.json',json.dumps(dict(status=status,files=manifest,created_at_utc=utc_now()),indent=2))
    print('Output ZIP:',archive)
    print('Status:',status)
    try:
        from google.colab import files
        files.download(str(archive))
    except Exception:
        print('Automatic browser download did not start; the ZIP is saved at the Drive path above.')
    return archive
