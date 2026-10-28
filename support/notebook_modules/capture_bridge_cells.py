# %% Select Gemma or Qwen-9B and preserve all previous experiments
MODEL_KEY='gemma12b'  # @param ["gemma12b", "qwen9b"]
# 换模型必须使用全新的 GPU 运行时。本份不运行 Qwen-27B。
RUN_ID='input_mapping_fix'  # 独立保存本次修正版；保留 main 中的未完成诊断包。
REPEAT_RUN_ID='main'
AUTO_DOWNLOAD=True
BATCH_SIZE_OVERRIDE=None  # 仅在掌握历史配置时填写；默认按旧模板候选（Gemma=2，Qwen=4）。
SPECS={'gemma12b':dict(model_id='google/gemma-4-12B-it',slug='google__gemma-4-12B-it',layers=48,
    revision='707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7',template_batch=2),
    'qwen9b':dict(model_id='Qwen/Qwen3.5-9B',slug='Qwen__Qwen3.5-9B',layers=32,
    revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a',template_batch=4)}
assert MODEL_KEY in SPECS
SPEC=SPECS[MODEL_KEY];MODEL_ID=SPEC['model_id'];MODEL_REVISION=SPEC['revision'];MODEL_SLUG=SPEC['slug']
MODEL_DTYPE='bfloat16';DEVICE_MAP='auto';EXPECTED_LAYER_COUNT=SPEC['layers']
OUTPUT_ROOT=PROJECT_ROOT/'results/cross_model_independent_confirmation/capture_bridge'/RUN_ID/MODEL_SLUG
packet=DiagnosticPacket(OUTPUT_ROOT,dict(schema='readout-capture-bridge-1.0',implementation=IMPLEMENTATION_SHA256,
    model=MODEL_KEY,revision=MODEL_REVISION,batch_override=BATCH_SIZE_OVERRIDE,template_batch=SPEC['template_batch'],
    cohort='original 19 development scenes; six mappings; no refitting',source_repeat_run=REPEAT_RUN_ID,
    route='current single, local discovery-template single, candidate discovery-template batch in one current runtime',
    precision='BF16 forward; native float32 copies and actual FP16 observer',new_scientific_confirmation=False))
ready=True;issues=[];coverage=[];single_results={};single_arrays={};batch_results=[];batch_arrays={}
historical_states={};historical_records={};historical_means={};archive_order=[]
archive_mean_checks=[];source_original_mean_consistent=None;MODEL_FORWARD_ATTEMPTS=0;MODEL_EXAMPLES_ATTEMPTED=0

def bridge_issue(stage,error):
    global ready
    ready=False
    row=dict(stage=stage,status='incomplete',reason=type(error).__name__+': '+str(error))
    issues.append(row);print('此部分未完成，继续运行后面的打包单元格：',row)

print('模型：',MODEL_ID,'；输出：',OUTPUT_ROOT)

# %% Load the frozen readout, exact development examples and historical provenance
try:
    from visual_torque_toolkit.behavior import BehaviorRunConfig,PromptCondition,load_behavior_examples
    READOUT_ROOT=PROJECT_ROOT/'results/cross_model_readout_correction_v2/models'/MODEL_SLUG
    DISCOVERY_ROOT=PROJECT_ROOT/'results'/MODEL_SLUG/'activation_discovery_v1'
    probe_path=packet.track(READOUT_ROOT/'frozen_probe.json')
    old_protocol=read_json(packet.track(READOUT_ROOT/'protocol.json'))
    old_runtime=read_json(packet.track(READOUT_ROOT/'runtime_identity.json'))
    capture_config=read_json(packet.track(DISCOVERY_ROOT/'capture_config.json'))
    history_identity=capture_config['identity']
    probe=read_json(probe_path)
    expected_capture_config=probe.get('fit_provenance',{}).get('files',{}).get(str(DISCOVERY_ROOT/'capture_config.json'))
    if expected_capture_config is not None:assert normalize_sha(expected_capture_config)==file_hash(DISCOVERY_ROOT/'capture_config.json')
    assert old_protocol['model_id']==MODEL_ID and old_protocol['revision']==MODEL_REVISION
    assert history_identity['model_id']==MODEL_ID and history_identity['resolved_revision']==MODEL_REVISION
    assert old_protocol['dtype']==history_identity['dtype']==MODEL_DTYPE and old_protocol['batch_size']==1
    SELECTED_LAYER=int(probe['layer']);CAPTURE_LAYERS=(SELECTED_LAYER,)
    assert SELECTED_LAYER in list(map(int,history_identity['layer_indices']))
    PROBE_MEAN=np.asarray(probe['mean'],np.float32);PROBE_SCALE=np.asarray(probe['scale'],np.float32)
    PROBE_COEF=np.asarray(probe['coef'],np.float32);PROBE_INTERCEPT=float(probe['intercept'])
    assert PROBE_MEAN.shape==PROBE_SCALE.shape==PROBE_COEF.shape and (PROBE_SCALE>0).all()
    assert all(np.isfinite(v).all() for v in (PROBE_MEAN,PROBE_SCALE,PROBE_COEF)) and np.isfinite(PROBE_INTERCEPT)
    BALANCE_THRESHOLD=float(old_protocol['balance_threshold']);READOUT_TOLERANCE=float(old_protocol['readout_tolerance'])
    manifest_path=packet.track(PROJECT_ROOT/'results/behavior_comparison_v3/activation_pilot_manifest.jsonl')
    manifest=read_jsonl(manifest_path)
    assert len(manifest)==254 and len({str(r['scene_id']) for r in manifest})==254
    assert {str(r['split']) for r in manifest}=={'probe_train','probe_dev'}
    dev_ids={str(r['scene_id']) for r in manifest if r['split']=='probe_dev'}
    assert len(dev_ids)==19 and dev_ids<=set(map(str,probe['fit_scene_ids']))
    assert normalize_sha(history_identity['activation_manifest_sha256'])==file_hash(manifest_path)
    example_config=BehaviorRunConfig(dataset_dir=DATASET_DIR,output_dir=OUTPUT_ROOT/'unused',model_id=MODEL_ID,
        model_revision=MODEL_REVISION,conditions=(PromptCondition.IMAGE_RULE,),question_kinds=('outcome',),
        splits=('probe_train','probe_dev'),batch_size=1,generation_every=0,seed=20260817)
    assert example_config.dataset_fingerprint()==history_identity['dataset_fingerprint']==old_runtime['dataset']
    all_ids={str(r['scene_id']) for r in manifest}
    all_examples=[e for e in load_behavior_examples(example_config) if str(e.scene_id) in all_ids]
    EXAMPLES={(str(e.scene_id),str(e.variant)):e for e in all_examples}
    assert len(all_examples)==len(EXAMPLES)==1524
    WANTED=sorted(k for k in EXAMPLES if k[0] in dev_ids)
    assert len(WANTED)==114 and all(sum(k[0]==sid for k in WANTED)==6 for sid in dev_ids)
    for sid in dev_ids:
        assert len({tuple(EXAMPLES[k].option_semantics) for k in WANTED if k[0]==sid})==6
    packet.json('historical_capture_configuration.json',capture_config)
    packet.json('frozen_readout.json',probe)
    packet.json('prior_E1_runtime.json',old_runtime)
    packet.json('capture_source_provenance.json',CAPTURE_SOURCE_PROVENANCE)
    logged_batch=history_identity.get('batch_size',capture_config.get('batch_size'))
    BATCH_SIZE=int(BATCH_SIZE_OVERRIDE if BATCH_SIZE_OVERRIDE is not None else logged_batch if logged_batch is not None else SPEC['template_batch'])
    assert 1<BATCH_SIZE<=8
    BATCH_SIZE_SOURCE='user_override' if BATCH_SIZE_OVERRIDE is not None else 'historical_configuration' if logged_batch is not None else 'local_template_default_not_historically_verified'
    print('冻结层：',SELECTED_LAYER,'；开发场景：19；映射：114；候选批大小：',BATCH_SIZE,'；来源：',BATCH_SIZE_SOURCE)
    print('沿用原读出器和阈值，不重新拟合。本诊断使用旧开发集，不报告新的泛化准确率。')
except Exception as error:
    bridge_issue('frozen_inputs',error)

# %% Read per-mapping archives and verify how the old scene means were formed
if ready:
    try:
        mean_path=DISCOVERY_ROOT/'scene_mean_activations.pt'
        if mean_path.is_file():
            packet.track(mean_path)
            expected=probe.get('fit_provenance',{}).get('files',{}).get(str(mean_path))
            if expected is not None:assert normalize_sha(expected)==file_hash(mean_path)
            payload=torch.load(mean_path,map_location='cpu',weights_only=True)
            pos=list(map(int,payload['layer_indices'])).index(SELECTED_LAYER)
            assert len(payload['metadata'])==len(payload['activations'])
            for i,record in enumerate(payload['metadata']):
                sid=str(record['scene_id'])
                if sid in dev_ids:
                    assert sid not in historical_means and record['split']=='probe_dev'
                    historical_means[sid]=payload['activations'][i,pos].float().numpy().copy()
            assert set(historical_means)==dev_ids
            assert all(np.isfinite(x).all() and x.shape==PROBE_MEAN.shape for x in historical_means.values())
            del payload
        else:
            coverage.append(dict(stage='historical_scene_means',status='missing',reason=str(mean_path)))
        shard_paths=sorted(DISCOVERY_ROOT.glob('shard_*.pt'))
        for path in tqdm(shard_paths,desc='Reading archived per-mapping states'):
            payload=torch.load(packet.track(path),map_location='cpu',weights_only=True)
            pos=list(map(int,payload['layer_indices'])).index(SELECTED_LAYER)
            assert len(payload['records'])==len(payload['activations'])
            for i,record in enumerate(payload['records']):
                key=capture_key(record)
                assert key in EXAMPLES and key not in historical_records
                ex=EXAMPLES[key]
                assert str(record['example_id'])==str(ex.example_id)
                assert str(record['gold_semantic'])==str(ex.gold_semantic) and int(record['net_torque'])==int(ex.net_torque)
                mapping={str(k):str(v) for k,v in record['label_to_semantic'].items()}
                assert mapping=={str(k):str(v) for k,v in ex.label_to_semantic.items()}
                archive_order.append(key);historical_records[key]=record
                if key in WANTED:
                    historical_states[key]=payload['activations'][i,pos].float().numpy().copy()
            del payload
        if shard_paths:
            assert set(historical_records)==set(EXAMPLES) and len(historical_states)==114,'Incomplete historical shard collection'
            assert all(np.isfinite(x).all() and x.shape==PROBE_MEAN.shape for x in historical_states.values())
            for sid in sorted(dev_ids):
                # Match the original shard-record summation order exactly.
                keys=[k for k in archive_order if k[0]==sid]
                rebuilt=np.stack([historical_states[k] for k in keys]).mean(0).astype('float16').astype('float32')
                if sid in historical_means:
                    difference=float(np.max(abs(rebuilt-historical_means[sid])))
                    archive_mean_checks.append(dict(scene_id=sid,hidden_max_difference=difference,
                        readout_difference=decode(rebuilt)-decode(historical_means[sid]),exact_match=difference==0))
            source_original_mean_consistent=all(r['exact_match'] for r in archive_mean_checks) if len(archive_mean_checks)==19 else None
            schedule_source='saved_shard_record_order; batch boundaries inferred, not historically logged'
        else:
            archive_order=[(str(e.scene_id),str(e.variant)) for e in all_examples]
            schedule_source='current reconstruction of original full manifest order; no historical per-mapping archive'
            coverage.append(dict(stage='historical_variant_shards',status='missing',reason='No shard_*.pt; continue with explicitly unverified candidate batch context'))
        WANTED=[k for k in archive_order if k[0] in dev_ids]
        assert len(WANTED)==len(set(WANTED))==114
        BATCHES=candidate_batch_schedule(archive_order,set(WANTED),BATCH_SIZE)
        USED=sorted({k for batch in BATCHES for k in batch}|set(WANTED))
        stamps={}
        for key in USED:
            ex=EXAMPLES[key];packet.track(ex.image_path)
            stamps[ex.example_id]=dict(scene_id=key[0],variant=key[1],split=ex.split,prompt=ex.prompt,
                image_sha256=file_hash(ex.image_path),labels=list(ex.labels),semantics=list(ex.option_semantics),gold=str(ex.gold_semantic))
        for path in sorted((TOOLKIT_ROOT/'src/visual_torque_toolkit').rglob('*.py')):packet.track(path)
        packet.json('example_inputs.json',stamps)
        packet.json('batch_schedule.json',dict(batch_size=BATCH_SIZE,batch_size_source=BATCH_SIZE_SOURCE,
            order_source=schedule_source,original_batch_membership_verified=False,
            mean_aggregation_order='within-scene record order from the stated schedule source; shared by all compared routes',
            batches=[[list(k) for k in b] for b in BATCHES],evaluated_development_mappings=114,
            historical_training_companions=len([k for k in USED if k not in WANTED])))
        packet.csv('historical_mean_rebuild.csv',pd.DataFrame(archive_mean_checks))
        diagnostic_show('历史逐映射分片能否重建原均值',pd.DataFrame(archive_mean_checks),19)
        print('批次比较只统计19个开发场景；用于恢复候选批次上下文的旧训练图片数量：',len(USED)-len(WANTED))
        print('单张最多228次前向（输入相同则114次），另加候选批次：',len(BATCHES),'；不做重复性重跑。')
    except Exception as error:
        bridge_issue('historical_archives',error)

# %% Freeze this current runtime and an isolated resumable capture cache
if ready:
    try:
        import transformers
        adapter.model.eval()
        assert adapter.metadata['resolved_revision']==MODEL_REVISION
        assert LAYER_PATH==history_identity['layer_path']
        runtime=dict(torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0),cuda_capability=list(torch.cuda.get_device_capability(0)),
            dtype=MODEL_DTYPE,layer_path=LAYER_PATH,checkpoint=MODEL_REVISION,
            attention_implementation=str(getattr(adapter.model.config,'_attn_implementation',None)),
            tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
            adapter_metadata=adapter.metadata,
            toolkit={str(p.relative_to(TOOLKIT_ROOT)):file_hash(p) for p in sorted((TOOLKIT_ROOT/'src/visual_torque_toolkit').rglob('*.py'))})
        CAPTURE_IDENTITY=json.loads(json.dumps(dict(schema='capture-bridge-cache-1.0',implementation=IMPLEMENTATION_SHA256,
            runtime=runtime,inputs=dict(packet.inputs),schedule=[[list(k) for k in b] for b in BATCHES],
            probe=file_hash(probe_path),layer=SELECTED_LAYER,batch_size=BATCH_SIZE,batch_size_source=BATCH_SIZE_SOURCE),default=plain))
        CAPTURE_CACHE=OUTPUT_ROOT/'captures'/digest(CAPTURE_IDENTITY)[:20]
        CAPTURE_CACHE.mkdir(parents=True,exist_ok=True)
        freeze(CAPTURE_CACHE/'identity.json',CAPTURE_IDENTITY)
        packet.json('current_runtime.json',runtime)
        packet.json('capture_identity.json',CAPTURE_IDENTITY)
        runtime_rows=[dict(field=k,prior_E1=json.dumps(old_runtime.get(k),default=plain),current=json.dumps(runtime.get(k),default=plain),
            equal=old_runtime.get(k)==runtime.get(k)) for k in ('torch','transformers','cuda','dtype','checkpoint','layer_path')]
        packet.csv('runtime_comparison.csv',pd.DataFrame(runtime_rows))
        diagnostic_show('当前运行与原 E1 配置；未记录字段不视为一致',pd.DataFrame(runtime_rows))
        print('本运行独立缓存：',CAPTURE_CACHE)
    except Exception as error:
        bridge_issue('runtime_identity',error)

# %% Compare both single-image preparation and capture conventions
if ready:
    try:
        for key in tqdm(WANTED,desc='Single-image capture bridge',unit='mapping'):
            result,arrays,_=cached_bridge_capture(['single',list(key)],lambda key=key:capture_single_bridge(EXAMPLES[key]))
            assert tuple(result['key'])==key
            single_results[key]=result;single_arrays[key]=arrays
        coverage.append(dict(stage='single_capture',status='complete',mappings=len(single_results)))
        packet.json('single_capture_details.json',list(single_results.values()))
        prep_rows=[dict(scene_id=k[0],variant=k[1],preparation_equal=r['preparation_equal'],shared_forward=r['shared_forward'],
            fp16_observer_gap=r['native_to_fp16_observer_gap']) for k,r in single_results.items()]
        packet.csv('single_capture_preparation.csv',pd.DataFrame(prep_rows))
        diagnostic_show('单张输入是否一致，以及同次前向的捕获对照',pd.DataFrame(prep_rows),20)
    except Exception as error:
        bridge_issue('single_capture',error)

# %% Replay candidate batch contexts without changing positions, precision or inputs
if ready:
    try:
        for keys in tqdm(BATCHES,desc='Candidate historical batch contexts',unit='batch'):
            result,arrays,_=cached_bridge_capture(['candidate_batch',[list(k) for k in keys]],
                lambda keys=keys:capture_batch_bridge([EXAMPLES[k] for k in keys]))
            assert result['keys']==[list(k) for k in keys]
            batch_results.append(result)
            for i,key in enumerate(keys):
                if key in WANTED:
                    assert key not in batch_arrays
                    batch_arrays[key]={name:value[i:i+1].copy() for name,value in arrays.items()}
        assert set(batch_arrays)==set(WANTED)
        coverage.append(dict(stage='batch_capture',status='complete',batches=len(BATCHES),mappings=len(batch_arrays)))
        packet.json('batch_capture_details.json',batch_results)
    except Exception as error:
        bridge_issue('batch_capture',error)

# %% Compare native states, historical variants and all nineteen scene means
scene_table=pd.DataFrame();gap_summary=pd.DataFrame();variant_table=pd.DataFrame();notes=[]
if ready:
    try:
        routes={name:np.concatenate([single_arrays[k][name] for k in WANTED],axis=0) for name in next(iter(single_arrays.values()))}
        routes.update({name:np.concatenate([batch_arrays[k][name] for k in WANTED],axis=0) for name in next(iter(batch_arrays.values()))})
        if set(historical_states)==set(WANTED):routes['historical_variants_fp16']=np.stack([historical_states[k] for k in WANTED])
        batch_by_key={tuple(key):(result,i) for result in batch_results for i,key in enumerate(result['keys']) if tuple(key) in WANTED}
        variant_rows=[]
        for i,key in enumerate(WANTED):
            single=single_results[key];batch,bi=batch_by_key[key]
            a=single['legacy_prepared_rows'][0];b=batch['prepared_rows'][bi]
            scores_a=single['legacy_record']['candidate_logprobs'];scores_b=batch['records'][bi]['candidate_logprobs']
            x=np.array([scores_a[k] for k in ('A','B','C')]);y=np.array([scores_b[k] for k in ('A','B','C')])
            row=dict(scene_id=key[0],variant=key[1],single_preparation_equal=single['preparation_equal'],
                single_terminal_active=a['selected_token_active'],batch_terminal_active=b['selected_token_active'],
                unpadded_tokens_equal=a['unpadded_input_ids']==b['unpadded_input_ids'],
                image_partition=a['image_partition']+' / '+b['image_partition'],
                image_fingerprint_equal=None if a['image_fingerprint'] is None or b['image_fingerprint'] is None else a['image_fingerprint']==b['image_fingerprint'],
                batch_vs_single_hidden_max=float(np.max(abs(routes['legacy_batch_native'][i]-routes['legacy_single_native'][i]))),
                batch_vs_single_readout=decode(routes['legacy_batch_native'][i])-decode(routes['legacy_single_native'][i]),
                batch_vs_single_centered_score_max=float(np.max(abs((y-y.mean())-(x-x.mean())))),
                batch_vs_single_answer_changed=single['legacy_record']['predicted']!=batch['records'][bi]['predicted'])
            for name,array in routes.items():
                row[name+'_readout']=decode(array[i])
                if key in historical_states:
                    row[name+'_historical_hidden_max']=float(np.max(abs(array[i]-historical_states[key])))
                    row[name+'_historical_readout_gap']=decode(array[i])-decode(historical_states[key])
            variant_rows.append(row)
        variant_table=pd.DataFrame(variant_rows)
        scene_table=aggregate_capture_scenes(WANTED,routes,list(single_results.values()),historical_means,BALANCE_THRESHOLD,decode)
        gap_summary=scene_gap_summary(scene_table,list(routes))
        packet.csv('all_114_mapping_comparisons.csv',variant_table)
        packet.csv('all_19_scene_comparisons.csv',scene_table)
        packet.csv('historical_gap_by_route.csv',gap_summary)
        path=OUTPUT_ROOT/'selected_capture_states.npz'
        with path.open('wb') as f:np.savez_compressed(f,**routes,keys=np.asarray(WANTED),
            historical_scene_ids=np.asarray(sorted(historical_means)),
            historical_scene_means=np.stack([historical_means[s] for s in sorted(historical_means)]) if historical_means else np.empty((0,len(PROBE_MEAN))))
        packet.outputs.add(path)
        repeat_path=PROJECT_ROOT/'results/cross_model_independent_confirmation/readout_repeatability'/REPEAT_RUN_ID/MODEL_SLUG/'analysis/historical_scene_mean_comparison.csv'
        if repeat_path.is_file():
            previous=pd.read_csv(packet.track(repeat_path))
            prior=previous.groupby('scene_id').current_native_readout.mean().rename('prior_repeat_current_readout')
            joined=scene_table[['scene_id','current_single_native_readout']].merge(prior,on='scene_id',validate='one_to_one')
            assert len(joined)==19
            joined['current_minus_prior_repeat']=joined.current_single_native_readout-joined.prior_repeat_current_readout
            packet.csv('current_vs_previous_repeatability_run.csv',joined)
        else:
            coverage.append(dict(stage='previous_repeat_comparison',status='missing',reason=str(repeat_path)))
        storage_ok=all(r['native_to_fp16_observer_gap']==0 for r in single_results.values()) and all(r['native_to_fp16_observer_gap']==0 for r in batch_results)
        notes=bridge_classification(variant_table.single_preparation_equal.all(),storage_ok,source_original_mean_consistent,
            gap_summary,float(variant_table.batch_vs_single_readout.abs().max()))
        diagnostic_show('同一19场景：哪种捕获方式更接近历史读出',gap_summary,10)
        diagnostic_show('完整19场景的读出与分类',scene_table,19)
        diagnostic_show('批处理的文本、图像和位置是否一致',variant_table[['scene_id','variant','unpadded_tokens_equal','image_fingerprint_equal','batch_terminal_active','batch_vs_single_readout','batch_vs_single_answer_changed']],20)
        for note in notes:print('•',note)
    except Exception as error:
        bridge_issue('comparison_analysis',error)

# %% Download a complete or explicitly partial diagnostic packet
packet.json('single_capture_details.json',list(single_results.values()))
packet.json('batch_capture_details.json',batch_results)
packet.csv('coverage.csv',pd.DataFrame(coverage+issues))
complete=bool(ready and len(scene_table)==19 and len(single_results)==114 and len(batch_arrays)==114)
summary=dict(status=('complete_with_missing_inputs' if any(r.get('status')=='missing' for r in coverage) else 'complete') if complete else 'partial_diagnostic_read_issues',model=MODEL_KEY,
    diagnostic_workflow_complete=complete,compatibility_conclusion='diagnostic_only_no_automatic_historical_equivalence',
    development_scenes=19,single_mappings_completed=len(single_results),batch_mappings_completed=len(batch_arrays),
    new_forward_attempts_this_execution=MODEL_FORWARD_ATTEMPTS,new_examples_attempted_this_execution=MODEL_EXAMPLES_ATTEMPTED,
    original_mean_rebuild_consistent=source_original_mean_consistent,
    historical_batch_membership_verified=False,batch_size_source=globals().get('BATCH_SIZE_SOURCE','unavailable'),
    batch_size=globals().get('BATCH_SIZE'),old_scientific_decisions_unchanged=True,probe_refit=False,
    issues=issues,coverage=coverage,interpretation=notes)
report='\n\n'.join(['# 新旧捕获与批处理对照',
    '这是在同一当前环境重放旧模板逻辑的诊断，不是新科学确认，也不声称完整恢复了历史环境。',
    '## 覆盖情况',csv_markdown(pd.DataFrame(coverage+issues)),
    '## 各路线与历史均值的距离',csv_markdown(gap_summary),
    '## 19个开发场景',csv_markdown(scene_table),
    '## 解读',*notes,
    '## 使用边界',
    '模型、冻结读出器、层位和分类阈值不变。全部模型前向为 BF16；原生状态存为精确 float32 副本，另保留旧式 FP16 捕获与均值存储。',
    '旧配置没有批大小时按本地旧模板设定候选批大小（Gemma=2，Qwen-9B=4），明确标记为推定。分片顺序不包含历史批次边界或恢复运行日志，因此批次成员没有被宣称为历史已验证。',
    '六种映射的求平均顺序沿用历史分片记录顺序（分片缺失时采用当前重建顺序），全部比较路线共用该顺序。与上一份重复性运行的差别还可能包含求和顺序的影响。',
    '单张两套准备流程比较完整张量内容；批次逐图比较未填充文本及可识别的图像分块。不支持逐图拆分的图像字段标为未知，不能把所有批次差异直接归因于数值内核。',
    '批处理路线保留旧模板的最后位置读取，包括可能出现的末尾填充；不在诊断中偷偷修正输入。若末尾是填充，这就是应报告的问题。',
    '历史训练图片只用作候选批次的上下文，不参与拟合或新的科学效果统计。开发集曾参与原读出器拟合，不能报告成新的测试集泛化成绩。',
    '若当前运行与上一份重复性诊断的环境不同，跨运行差异不能叫作同配置随机性。缓存按完整运行身份隔离，不复用不同输入、软件、模型或硬件的状态。',
    '本包包含逐映射捕获状态、输入指纹、候选批次组成、原冻结读出器和完整表格。未完成的部分列入 issues；不要把 partial 状态作为问题解决。'])+'\n'
packet.finish(summary,report,'VisualTorque_Capture_Bridge_'+MODEL_KEY+'_Output.zip',AUTO_DOWNLOAD)
