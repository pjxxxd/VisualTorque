# %% Select model and preserve all previous experiments
MODEL_KEY='gemma12b'  # @param ["gemma12b", "qwen9b"]
RUN_ID='main'
AUTO_DOWNLOAD=True
SPECS={
 'gemma12b':dict(model_id='google/gemma-4-12B-it',slug='google__gemma-4-12B-it',layers=48,layer=7,alpha=100.,
     revision='707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7'),
 'qwen9b':dict(model_id='Qwen/Qwen3.5-9B',slug='Qwen__Qwen3.5-9B',layers=32,layer=13,alpha=.01,
     revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a')}
assert MODEL_KEY in SPECS, 'This notebook is only for Gemma-12B and Qwen-9B.'
SPEC=SPECS[MODEL_KEY];MODEL_ID=SPEC['model_id'];MODEL_SLUG=SPEC['slug'];MODEL_REVISION=SPEC['revision']
EXPECTED_LAYER_COUNT=SPEC['layers'];SELECTED_LAYER=SPEC['layer'];FROZEN_ALPHA=SPEC['alpha']
CAPTURE_LAYERS=(SELECTED_LAYER,);MODEL_DTYPE='bfloat16';DEVICE_MAP='auto'
READOUT_TOLERANCE=.05;NORM_TOLERANCE=.02;BALANCE_THRESHOLD=.5;RANDOM_SEEDS=(0,1,2,3)
MODEL_ROOT=PROJECT_ROOT/'results/cross_model_aligned_readout_v1'/RUN_ID/MODEL_SLUG
PROTOCOL=dict(schema='aligned-readout-validation-1.0',implementation=IMPLEMENTATION_SHA256,model=MODEL_KEY,
    model_id=MODEL_ID,revision=MODEL_REVISION,layer=SELECTED_LAYER,alpha=FROZEN_ALPHA,batch_size=1,dtype=MODEL_DTYPE,
    fit_scenes=254,fit_mapping_count=6,test_scenes=1007,test_mapping='historically_fixed_single_mapping',
    intervention_scenes=304,random_control_scenes=72,random_seeds=list(RANDOM_SEEDS),
    storage='native states preserved; fit uses per-mapping FP16 then FP32 six-map mean then FP16',
    fit='StandardScaler and ridge lsqr max_iter=5000; refit original 254 with fixed layer and alpha',
    readout_tolerance=READOUT_TOLERANCE,norm_tolerance=NORM_TOLERANCE,balance_threshold=BALANCE_THRESHOLD,
    accuracy_sesoi=.05,bootstrap_unit='scene',bootstrap_replicates=10000,oracle_uses_true_torque=True,
    primary_comparison='oracle minus fresh zero accuracy, all 304 attempts',
    control_comparison='oracle minus mean of four raw-isotropic equal-actual-norm directions, original 72 scenes',
    new_independent_confirmation=False,historical_activation_reuse=False,historical_test_cache_reuse=False)
packet=DiagnosticPacket(MODEL_ROOT/'setup',PROTOCOL)
ready=True;issues=[];stages=[];NEW_FORWARD_COUNT=0;CACHE_HITS=0
training_records=[];training_arrays=[];training_meta=[];test_records=[];test_arrays=[]
intervention_records=[];intervention_arrays=[];intervention_keys=[];dev_checks=[]
eval_summary=pd.DataFrame();effect_summary=pd.DataFrame();contrasts=pd.DataFrame();technical={}
def record_issue(stage,error):
    global ready
    ready=False
    item=dict(stage=stage,error_type=type(error).__name__,reason=str(error))
    issues.append(item)
    print('此阶段未完成；请继续运行到最后打包。',item)
print('模型：',MODEL_ID,'；固定层：',SELECTED_LAYER,'；训练、测试和干预全部逐张处理。')
print('原实验目录不改动；新结果保存于：',MODEL_ROOT)

# %% Validate original fit scenes, fixed test mappings and diagnostic cohorts
try:
    from visual_torque_toolkit.behavior import BehaviorRunConfig,PromptCondition,load_behavior_examples
    old_root=PROJECT_ROOT/'results/cross_model_readout_correction_v2'
    old_model=old_root/'models'/MODEL_SLUG
    historical_probe=read_json(packet.track(old_model/'frozen_probe.json'))
    old_protocol=read_json(packet.track(old_model/'protocol.json'))
    discovery_config=read_json(packet.track(PROJECT_ROOT/'results'/MODEL_SLUG/'activation_discovery_v1/capture_config.json'))['identity']
    assert old_protocol['model_id']==MODEL_ID and old_protocol['revision']==MODEL_REVISION
    assert discovery_config['model_id']==MODEL_ID and discovery_config['resolved_revision']==MODEL_REVISION
    assert historical_probe['layer']==SELECTED_LAYER and historical_probe['alpha']==FROZEN_ALPHA
    assert old_protocol['dtype']==discovery_config['dtype']==MODEL_DTYPE
    assert old_protocol['batch_size']==1 and old_protocol['balance_threshold']==BALANCE_THRESHOLD
    assert old_protocol['readout_tolerance']==READOUT_TOLERANCE and old_protocol['norm_tolerance']==NORM_TOLERANCE
    probe_parameters(historical_probe)
    fit_manifest_path=packet.track(PROJECT_ROOT/'results/behavior_comparison_v3/activation_pilot_manifest.jsonl')
    fit_manifest=read_jsonl(fit_manifest_path)
    assert len(fit_manifest)==254 and len({str(r['scene_id']) for r in fit_manifest})==254
    assert Counter(r['split'] for r in fit_manifest)=={'probe_train':235,'probe_dev':19}
    FIT_IDS=list(map(str,historical_probe['fit_scene_ids']))
    assert len(FIT_IDS)==len(set(FIT_IDS))==254 and set(FIT_IDS)=={str(r['scene_id']) for r in fit_manifest}
    FIT_SPLITS={str(r['scene_id']):r['split'] for r in fit_manifest}
    selection=read_json(packet.track(old_root/'shared_304_manifest.json'))
    TEST_IDS=read_json(packet.track(old_root/'shared_1007_scene_ids.json'))
    assert len(TEST_IDS)==len(set(TEST_IDS))==1007 and set(TEST_IDS).isdisjoint(FIT_IDS)
    assert len(selection)==304 and len({r['scene_id'] for r in selection})==304
    selected_ids={str(r['scene_id']) for r in selection}
    CONTROL_IDS={str(r['scene_id']) for r in selection if r['orthogonal_control_subset']}
    assert len(CONTROL_IDS)==72 and selected_ids<=set(TEST_IDS)
    assert Counter(r['gold_outcome'] for r in selection)=={'left':132,'right':131,'balanced':41}
    def example_config(splits):
        return BehaviorRunConfig(dataset_dir=DATASET_DIR,output_dir=MODEL_ROOT/'unused',model_id=MODEL_ID,
            model_revision=MODEL_REVISION,conditions=(PromptCondition.IMAGE_RULE,),question_kinds=('outcome',),
            splits=tuple(splits),batch_size=1,generation_every=0,seed=20260820)
    config=example_config(('probe_train','probe_dev','probe_test'))
    assert config.dataset_fingerprint()==discovery_config['dataset_fingerprint']
    packet.track(DATASET_DIR/'scenes.jsonl');packet.track(DATASET_DIR/'splits.json')
    examples=load_behavior_examples(config)
    by_scene=defaultdict(list)
    for ex in examples:
        if str(ex.scene_id) in set(FIT_IDS)|set(TEST_IDS):by_scene[str(ex.scene_id)].append(ex)
    assert set(by_scene)==set(FIT_IDS)|set(TEST_IDS)
    for sid,group in by_scene.items():
        assert len(group)==6 and {str(e.variant) for e in group}=={'permutation_'+str(i) for i in range(6)}
        assert len({tuple(e.label_to_semantic[k] for k in e.labels) for e in group})==6
        if sid in FIT_SPLITS:assert {e.split for e in group}=={FIT_SPLITS[sid]}
        else:assert {e.split for e in group}=={'probe_test'}
    FIT_EXAMPLES={sid:sorted(by_scene[sid],key=lambda e:e.variant) for sid in FIT_IDS}
    TEST_EXAMPLES={sid:next(e for e in by_scene[sid] if e.variant==fixed_variant(sid)) for sid in sorted(TEST_IDS)}
    SELECTED_EXAMPLES=[]
    for row in selection:
        ex=TEST_EXAMPLES[str(row['scene_id'])]
        assert row['example_id']==ex.example_id and row['variant']==ex.variant
        assert row['problem_type']==ex.problem_type and row['gold_outcome']==ex.gold_semantic and int(row['net_torque'])==int(ex.net_torque)
        SELECTED_EXAMPLES.append(ex)
    used=[e for sid in FIT_IDS for e in FIT_EXAMPLES[sid]]+list(TEST_EXAMPLES.values())
    image_hashes={}
    for path in tqdm(sorted({str(e.image_path) for e in used}),desc='Checking existing images'):
        tracked=packet.track(Path(path));image_hashes[path]=packet.inputs[str(tracked)]
    INPUTS={}
    for ex in used:
        INPUTS[str(ex.example_id)]=dict(example_id=str(ex.example_id),scene_id=str(ex.scene_id),variant=str(ex.variant),
            split=str(ex.split),prompt=str(ex.prompt),labels=list(ex.labels),
            label_to_semantic={str(k):str(v) for k,v in ex.label_to_semantic.items()},
            image_sha256=image_hashes[str(ex.image_path)],net_torque=int(ex.net_torque),gold_semantic=str(ex.gold_semantic))
    assert len(INPUTS)==1524+1007
    packet.json('historical_probe_reference.json',historical_probe)
    packet.json('fit_manifest.json',fit_manifest);packet.json('diagnostic_manifest.json',selection)
    packet.json('test_fixed_mapping.json',[INPUTS[str(e.example_id)] for e in TEST_EXAMPLES.values()])
    stages.append('inputs_validated')
    print('已固定：254 个拟合场景 × 6 映射；1007 个测试场景 × 1 映射；304 个诊断场景；72 个随机对照场景。')
except Exception as error:
    record_issue('input_validation',error)

# %% Freeze the current runtime and isolate its resumable captures
if ready:
    try:
        import platform,importlib.metadata,transformers
        adapter.model.eval()
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        assert adapter.metadata['resolved_revision']==MODEL_REVISION and LAYER_PATH==discovery_config['layer_path']
        package_versions={}
        for name in ('accelerate','torchvision','flash-attn','flash-linear-attention','causal-conv1d','safetensors'):
            try:package_versions[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:package_versions[name]=None
        runtime=dict(torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,
            scipy=scipy.__version__,transformers=transformers.__version__,python=platform.python_version(),cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0),capability=list(torch.cuda.get_device_capability(0)),
            checkpoint=MODEL_REVISION,dtype=MODEL_DTYPE,batch_size=1,layer=SELECTED_LAYER,layer_path=LAYER_PATH,
            attention_implementation=str(getattr(adapter.model.config,'_attn_implementation',None)),
            tf32_matmul=False,cudnn_benchmark=False,packages=package_versions,
            toolkit={str(p.relative_to(TOOLKIT_ROOT)):file_hash(p) for p in sorted((TOOLKIT_ROOT/'src/visual_torque_toolkit').rglob('*.py'))})
        RUN_IDENTITY=dict(protocol=PROTOCOL,runtime=runtime,input_files=dict(packet.inputs),examples_digest=digest(INPUTS))
        old_packet=packet
        RUN_ROOT=MODEL_ROOT/'runs'/digest(RUN_IDENTITY)[:20]
        packet=DiagnosticPacket(RUN_ROOT,dict(protocol=PROTOCOL,runtime_identity_sha256=digest(RUN_IDENTITY)))
        packet.inputs=dict(old_packet.inputs)
        for p in sorted(old_packet.outputs):
            if p.name!='protocol.json':packet.text(p.name,p.read_text(encoding='utf-8'))
        packet.json('runtime_identity.json',RUN_IDENTITY);packet.json('example_inputs.json',INPUTS)
        CAPTURE_IDENTITY=digest(RUN_IDENTITY)
        stages.append('runtime_locked')
        print('本次结果及断点缓存：',RUN_ROOT)
        print('不同运行身份自动进入不同目录；不读取旧实验的训练或测试激活。')
    except Exception as error:
        record_issue('runtime_identity',error)

# %% Recapture all training states with singleton forwards
if ready:
    try:
        for sid in tqdm(FIT_IDS,desc='Capturing 254 fit scenes',unit='scene'):
            group=FIT_EXAMPLES[sid]
            identity=digest([CAPTURE_IDENTITY,[INPUTS[str(e.example_id)] for e in group]])
            rows,arrays,reused=aligned_cache(RUN_ROOT,'training',sid,identity,lambda group=group:capture_training_scene(group))
            assert [r['example_id'] for r in rows]==[str(e.example_id) for e in group]
            assert arrays['native'].shape==(6,len(historical_probe['coef']))
            assert np.array_equal(arrays['fit_mean'],six_mapping_mean(arrays['native']))
            training_records.extend(rows);training_arrays.append(arrays['native'])
            training_meta.append(dict(scene_id=sid,split=FIT_SPLITS[sid],net_torque=int(group[0].net_torque)))
            CACHE_HITS+=int(reused)
        assert len(training_meta)==254 and len(training_records)==1524
        TRAIN_NATIVE=np.stack(training_arrays).astype(np.float32)
        TRAIN_MEANS=np.stack([six_mapping_mean(a) for a in TRAIN_NATIVE])
        TRAIN_TARGETS=np.asarray([r['net_torque'] for r in training_meta],float)
        TRAIN_STATE_SHA=save_arrays(packet,'training_states.npz',native=TRAIN_NATIVE,fit_means=TRAIN_MEANS,
            targets=TRAIN_TARGETS,scene_ids=np.asarray(FIT_IDS),mapping_order=np.asarray([e.variant for e in FIT_EXAMPLES[FIT_IDS[0]]]))
        packet.json('training_capture_records.json',training_records)
        packet.csv('training_scene_metadata.csv',pd.DataFrame(training_meta))
        stages.append('training_captured')
    except Exception as error:
        record_issue('training_capture',error)

# %% Refit the fixed readout using only newly captured training states
if ready:
    try:
        provenance=dict(mode='new_singleton_training_capture',capture_identity=CAPTURE_IDENTITY,
            training_states_sha256=TRAIN_STATE_SHA,training_mean_sha256=hashlib.sha256(TRAIN_MEANS.tobytes()).hexdigest(),
            layer_selection='historically_fixed_no_new_search',alpha_selection='historically_fixed_no_new_search',
            mapping_mean='sorted six variants; per-variant FP16 then FP32 mean then FP16',
            original_train_scenes=235,original_dev_scenes=19,final_refit_scenes=254,test_scenes_used=0)
        probe_path=RUN_ROOT/'aligned_probe.json'
        if probe_path.is_file():
            aligned_probe=read_json(probe_path)
            assert aligned_probe['fit_provenance']==provenance and aligned_probe['fit_scene_ids']==FIT_IDS
            assert aligned_probe['layer']==SELECTED_LAYER and aligned_probe['alpha']==FROZEN_ALPHA
            probe_parameters(aligned_probe)
        else:
            aligned_probe=fit_fixed_probe(TRAIN_MEANS,TRAIN_TARGETS,FIT_IDS,SELECTED_LAYER,FROZEN_ALPHA,provenance)
            save_json(probe_path,aligned_probe)
        packet.outputs.add(probe_path)
        PROBE_SHA=file_hash(probe_path)
        PROBE_MEAN,PROBE_SCALE,PROBE_COEF,PROBE_INTERCEPT=probe_parameters(aligned_probe)
        # Parameter fingerprints make a modified probe fail resume before inference.
        freeze(RUN_ROOT/'probe_identity.json',dict(probe_sha256=PROBE_SHA,provenance=provenance))
        packet.outputs.add(RUN_ROOT/'probe_identity.json')
        stages.append('probe_fitted')
        print('新读出器已固定；未用测试结果选层、选超参数或调阈值。')
    except Exception as error:
        record_issue('probe_fit',error)

# %% Check no-op replay and numerical writes on two fixed development examples
if ready:
    try:
        dev=sorted([e for sid in FIT_IDS if FIT_SPLITS[sid]=='probe_dev' for e in FIT_EXAMPLES[sid]],
            key=lambda e:hashlib.sha256(('aligned-write:'+str(e.example_id)).encode()).hexdigest())[:2]
        def check_development():
            rows=[]
            for ex in dev:
                clean,a=aligned_forward(ex)
                noop,b=aligned_forward(ex,lambda pre,cast:(pre.copy(),dict(condition='noop')),a['pre'],clean['input_fingerprint'])
                assert noop['actual_delta_norm']==0 and np.array_equal(a['pre'],b['post'])
                assert max(abs(clean['candidate_logprobs'][k]-noop['candidate_logprobs'][k]) for k in clean['candidate_logprobs'])<=1e-6
                oracle,c=aligned_forward(ex,aligned_oracle(ex.net_torque),a['pre'],clean['input_fingerprint'])
                random,d=aligned_forward(ex,aligned_random(ex.scene_id,0,oracle['actual_delta_norm']),a['pre'],clean['input_fingerprint'])
                valid,relative=random_validity(random['actual_delta_norm'],oracle['actual_delta_norm'],NORM_TOLERANCE)
                random.update(valid_control=valid,norm_error_relative=relative)
                rows.extend([clean,noop,oracle,random])
            return rows,dict(completed=np.asarray([len(rows)],np.int32))
        dev_checks,_,reused=aligned_cache(RUN_ROOT,'development','two_fixed_examples',
            digest([CAPTURE_IDENTITY,PROBE_SHA,[str(e.example_id) for e in dev]]),check_development)
        CACHE_HITS+=int(reused)
        packet.json('development_technical_checks.json',dev_checks)
        oracle_checks=[r for r in dev_checks if r.get('condition')=='oracle']
        assert any(r['achieved'] for r in oracle_checks), 'Neither fixed development write reached numerical tolerance; inspect checks before test interventions.'
        stages.append('development_checked')
        diagnostic_show('开发样本写入检查（不作为论文效果）',pd.DataFrame([r for r in dev_checks if 'condition' in r]),8)
    except Exception as error:
        record_issue('development_check',error)

# %% Evaluate both readouts on the same 1007 current test states
if ready:
    try:
        TEST_STATE_BY_ID={};TEST_RECORD_BY_ID={}
        for sid,ex in tqdm(list(TEST_EXAMPLES.items()),desc='Evaluating 1007 fixed test mappings',unit='scene'):
            rows,arrays,reused=aligned_cache(RUN_ROOT,'test',sid,digest([CAPTURE_IDENTITY,INPUTS[str(ex.example_id)]]),
                lambda ex=ex:capture_test_example(ex))
            assert len(rows)==1 and rows[0]['example_id']==str(ex.example_id)
            r=dict(rows[0]);h=arrays['native']
            assert h.shape==PROBE_MEAN.shape
            for name,p in [('aligned',aligned_probe),('historical',historical_probe)]:
                value=decode_with(h,p)
                r[name+'_readout']=value;r[name+'_readout_correct']=readout_class(value)==r['gold_semantic']
            test_records.append(r);test_arrays.append(h)
            TEST_STATE_BY_ID[sid]=h;TEST_RECORD_BY_ID[sid]=r;CACHE_HITS+=int(reused)
        assert len(test_records)==1007
        packet.json('test_records.json',test_records);packet.csv('test_records.csv',pd.DataFrame(test_records))
        save_arrays(packet,'test_states.npz',native=np.stack(test_arrays),scene_ids=np.asarray(list(TEST_EXAMPLES)))
        eval_summary=summarize_test(test_records)
        packet.csv('readout_answer_comparison.csv',eval_summary)
        diagnostic_show('同一当前测试状态：新旧读出器与模型答案',eval_summary[eval_summary.group.eq('all')],5)
        stages.append('test_evaluated')
    except Exception as error:
        record_issue('test_evaluation',error)

# %% Correct the true torque on 304 scenes and match random edits on the original 72
if ready:
    try:
        for ex in tqdm(SELECTED_EXAMPLES,desc='Paired correction and random controls',unit='scene'):
            sid=str(ex.scene_id)
            identity=digest([CAPTURE_IDENTITY,PROBE_SHA,INPUTS[str(ex.example_id)],sid in CONTROL_IDS])
            rows,arrays,reused=aligned_cache(RUN_ROOT,'interventions',sid,identity,
                lambda ex=ex,sid=sid:run_aligned_scene(ex,TEST_RECORD_BY_ID[sid],TEST_STATE_BY_ID[sid],sid in CONTROL_IDS))
            assert len(rows)==(6 if sid in CONTROL_IDS else 2)
            assert np.array_equal(arrays['pre'],TEST_STATE_BY_ID[sid])
            for r in rows:
                assert r['example_id']==str(ex.example_id) and r['probe_sha256']==PROBE_SHA
                key='random_'+str(r['random_seed'])+'_post' if r['condition']=='random' else r['condition']+'_post'
                post=arrays[key]
                assert abs(decode(post)-r['decoded_post'])<=1e-9
                assert abs(float(np.linalg.norm(post-arrays['pre']))-r['actual_delta_norm'])<=1e-6
                if r['condition']=='oracle':assert r['achieved']==(abs(decode(post)-ex.net_torque)<=READOUT_TOLERANCE)
                intervention_arrays.append(post);intervention_keys.append([sid,r['condition'],str(r['random_seed'])])
            intervention_records.extend(rows);CACHE_HITS+=int(reused)
        assert len(intervention_records)==896
        packet.json('intervention_records.json',intervention_records)
        packet.csv('intervention_records.csv',pd.DataFrame(intervention_records))
        save_arrays(packet,'intervention_post_states.npz',post=np.stack(intervention_arrays),keys=np.asarray(intervention_keys))
        stages.append('interventions_evaluated')
    except Exception as error:
        record_issue('interventions',error)

# %% Report numerical correction, repairs, harms, matched controls and paired intervals
if ready:
    try:
        effect_summary,contrasts,technical=summarize_interventions(intervention_records,
            [str(e.scene_id) for e in SELECTED_EXAMPLES],CONTROL_IDS)
        packet.csv('correction_summary.csv',effect_summary);packet.csv('paired_effects.csv',contrasts)
        f=pd.DataFrame(intervention_records)
        packet.csv('oracle_write_failures.csv',f[f.condition.eq('oracle')&f.achieved.ne(True)])
        packet.csv('random_norm_failures.csv',f[f.condition.eq('random')&f.valid_control.ne(True)])
        packet.json('technical_summary.json',technical)
        diagnostic_show('修准是否成功，以及修好／改错多少题',effect_summary,5)
        diagnostic_show('相同场景的准确率和正确答案得分比较',contrasts,10)
        print('所有尝试保留；修准成功子集和四个随机对照均有效的共同子集另列。')
        print('区间不足以排除较大改善时，不能把未检出提升解释为没有作用。')
        stages.append('analysis_complete')
    except Exception as error:
        record_issue('analysis',error)

# %% Download the complete result packet or an explicitly incomplete packet
changed_inputs=[p for p,h in packet.inputs.items() if not Path(p).is_file() or file_hash(p)!=h]
if changed_inputs:
    record_issue('source_verification',RuntimeError('Input files changed during this run: '+repr(changed_inputs)))
expected_stages={'inputs_validated','runtime_locked','training_captured','probe_fitted','development_checked',
    'test_evaluated','interventions_evaluated','analysis_complete'}
complete=bool(ready and set(stages)==expected_stages and len(training_meta)==254 and len(test_records)==1007 and len(intervention_records)==896)
summary=dict(status='complete' if complete else 'partial',workflow_complete=complete,model=MODEL_KEY,
    fit_scenes_completed=len(training_meta),fit_mapping_records=len(training_records),test_scenes_completed=len(test_records),
    intervention_records_completed=len(intervention_records),new_forward_attempts=NEW_FORWARD_COUNT,cache_groups_reused=CACHE_HITS,
    stages=stages,issues=issues,technical=technical,historical_activation_reuse=False,historical_test_cache_reuse=False,
    fixed_layer=SELECTED_LAYER,fixed_alpha=FROZEN_ALPHA,batch_size=1,new_independent_confirmation=False,
    primary_mapping='fixed single mapping; original six-map average retained only for fit',original_scientific_decisions_unchanged=True)
packet.json('progress.json',summary)
# A failed stage still exports completed rows, not stale outputs from an older invocation.
if not complete:
    packet.json('partial_training_records.json',training_records)
    packet.json('partial_test_records.json',test_records)
    packet.json('partial_intervention_records.json',intervention_records)
    if training_arrays:
        save_arrays(packet,'partial_training_states.npz',native=np.stack(training_arrays),
            scene_ids=np.asarray([r['scene_id'] for r in training_meta]))
    if test_arrays:
        save_arrays(packet,'partial_test_states.npz',native=np.stack(test_arrays),
            scene_ids=np.asarray([r['scene_id'] for r in test_records]))
    if intervention_arrays:
        save_arrays(packet,'partial_intervention_post_states.npz',post=np.stack(intervention_arrays),
            keys=np.asarray(intervention_keys))
lines=['# 统一捕获条件的读出验证',f'模型：{MODEL_ID}；层位：{SELECTED_LAYER}；ridge alpha：{FROZEN_ALPHA}。',
    '训练、测试和干预均采用当前逐张 BF16 前向。只重新拟合外部读出器，大模型权重不变。',
    '原 254 个场景用于固定配置的最终拟合；训练保留六映射平均，主要测试和干预使用原来固定的一种映射。',
    '原测试场景已经被查看，本轮属于协议稳健性复核，不是新的独立盲测。']
if complete:
    main=effect_summary[effect_summary.group.eq('all_304')].iloc[0]
    aligned=eval_summary[eval_summary.group.eq('all')&eval_summary.probe.eq('aligned')].iloc[0]
    lines += [f'1007 个测试场景中，新读出器判断正确 {int(aligned.readout_correct)} 个，模型答对 {int(aligned.model_correct)} 个；读出正确但模型答错 {int(aligned.readout_correct_model_wrong)} 个。',
        f'304 个诊断场景中，数值修准达标 {int(main.oracle_achieved)} 个；答对数量 {int(main.before_correct)} → {int(main.after_correct)}；修好 {int(main.repaired)} 个原错误，新增 {int(main.harmed)} 个错误。',
        f'准确率变化 {main.accuracy_change*100:.2f} 个百分点，95% 场景配对区间 [{main.lower*100:.2f}, {main.upper*100:.2f}]。',
        f'随机对照实际幅度未达标 {technical["random_invalid"]}/{technical["random_attempted"]} 次；修准达标且四个随机方向均有效的共同子集有 {technical["common_valid_control_scenes"]} 个场景。',
        '完整总体、平衡题、非平衡题、有效子集和随机对照比较分别见 CSV；未通过技术检查的尝试没有被删除。',
        '同一当前状态上的历史读出器数值仅作为参照；不将旧运行的修正效果混入本轮成对比较。']
else:
    lines+=['本次未完整完成；只能使用明确完成的部分，不能给出完整验证结论。',json.dumps(issues,ensure_ascii=False,indent=2)]
lines+=['数值修准使用真实净力矩；它检验所选外部读出器方向的作用，不自动证明模型原生物理计算被修复。',
    '原 5 个百分点实质改善标准仅用于解释区间，不能把不显著当作零作用；技术失败和有效子集必须同时查看。',
    '原有 Qwen-27B、微调、内容确认及修正—阻断结果均未修改。']
packet.finish(summary,'\n\n'.join(lines)+'\n','VisualTorque_Aligned_Readout_'+MODEL_KEY+'_Output.zip',AUTO_DOWNLOAD)
