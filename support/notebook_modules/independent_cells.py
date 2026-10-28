"""Standalone notebook cells for the next, bounded experiment round."""

# %% QWEN: Read the saved three directions and original dose
SOURCE_ROOT=FOLLOWUP_ROOT/'equal_blocking'/SOURCE_RUN_ID
SELECTION_PATH=SOURCE_ROOT/'selected_route.json'
DIRECTIONS_PATH=SOURCE_ROOT/'frozen_directions.npz'
for path in (SELECTION_PATH,DIRECTIONS_PATH,SOURCE_ROOT/'analysis/RUN_COMPLETE.json'):
    assert path.is_file(), f'先完成上一批第 2 份 Qwen-27B 等幅度阻断 notebook：{path}'
reference=read_json(SELECTION_PATH)
assert reference['model']=='qwen27b' and reference['model_id']==MODEL_ID and reference['revision']==MODEL_REVISION
assert reference['upstream_layer']==48 and reference['downstream_layer']==55
assert file_hash(DIRECTIONS_PATH)==reference['directions_sha256']
with np.load(DIRECTIONS_PATH,allow_pickle=False) as archive:
    assert {'upstream','semantic'}|{'code_'+a+'_'+b for a in LABELS for b in LABELS if a!=b}==set(archive.files)
    arrays={k:archive[k].copy() for k in archive.files}
assert len({v.shape for v in arrays.values()})==1 and all(v.ndim==1 and np.linalg.norm(v)>0 for v in arrays.values())
vector_sha=frozen_arrays(OUTPUT_ROOT/'frozen_directions.npz',arrays)
UPSTREAM_PAYLOAD=arrays['upstream']*float(reference['dose'])
SEMANTIC_PAYLOAD=arrays['semantic']
CODE_DIRECTIONS={k[5:]:v for k,v in arrays.items() if k.startswith('code_')}
freeze(OUTPUT_ROOT/'reference.json',dict(reference=reference,source_selection=file_hash(SELECTION_PATH),
    source_vectors=file_hash(DIRECTIONS_PATH),copied_vectors=vector_sha,semantic_layer=49,semantic_dose=1.))
print('固定：L48 原剂量；L49 原方向，factor=1；L55 原字母方向。')

# %% GEMMA: Recover the fixed L31 candidate from original training caches
SOURCE_ROOT=FOLLOWUP_ROOT/'repair_development'/SOURCE_RUN_ID/MODEL_SLUG
manifest_path=PROJECT_ROOT/'results/behavior_comparison_v3/activation_pilot_manifest.jsonl'
assert (SOURCE_ROOT/'analysis/RUN_COMPLETE.json').is_file(), SOURCE_ROOT
direction,reference=recover_gemma_candidate(SOURCE_ROOT,manifest_path,DATASET_DIR/'scenes.jsonl',MODEL_REVISION,
    progress=lambda rows:tqdm(rows,desc='Reading original training caches'))
vector_sha=frozen_arrays(OUTPUT_ROOT/'frozen_directions.npz',{'upstream':direction})
freeze(OUTPUT_ROOT/'reference.json',reference)
UPSTREAM_PAYLOAD=direction*float(reference['dose'])
SEMANTIC_PAYLOAD=None;CODE_DIRECTIONS={}
print('已从 235 个旧训练场景恢复 L31、factor=1；未加载开发场景状态进行拟合。')
print('原 selected_route.json 的 eligible=False 保持不变。原选中向量未导出，本次记录重建公式和软件版本。')

# %% COMMON: Freeze the independent support before model inference
from visual_torque_toolkit.render import RenderStyle,render_scene
from visual_torque_toolkit.world import Scene,Side,Stack
support_dir=OUTPUT_ROOT/'cohort'
families,support_file=independent_support(PROJECT_ROOT,DATASET_DIR,support_dir,N_FAMILIES,COHORT_SEED,PROTOCOL)
images=support_dir/'images';images.mkdir(exist_ok=True)
for family in tqdm(families,desc='Rendering independent families'):
    for row in family['scenes']:
        path=images/row['image_filename']
        if not path.is_file():
            style=RenderStyle(ticks_per_side=6) if row['style']=='standard' else RenderStyle(ticks_per_side=6,background=(239,245,255),block_fill=(37,99,235))
            scene=Scene(row['scene_id'],tuple(Stack(str(i),Side(s['side']),int(s['distance']),int(s['units'])) for i,s in enumerate(row['stacks'])))
            render_scene(scene,path,style)
image_hashes={row['image_filename']:file_hash(images/row['image_filename']) for f in families for row in f['scenes']}
freeze(support_dir/'images.json',image_hashes)
print(f'{len(families)} 个物理家族 × 2 个镜像 × 2 种渲染 × 2 种映射。统计独立单位仍为 {len(families)}。')
print('已排除当前 Drive 中登记的旧训练/开发、固定位置和修复确认场景。')

# %% COMMON: Check runtime against the frozen reference and lock capture identity
adapter.model.eval()
runtime=current_runtime_identity()
runtime['gpu']=torch.cuda.get_device_name(0)
runtime['cuda_capability']=list(torch.cuda.get_device_capability(0))
runtime['attention_implementation']=str(getattr(adapter.model.config,'_attn_implementation',None))
runtime['tf32_matmul']=bool(torch.backends.cuda.matmul.allow_tf32)
runtime['cudnn_benchmark']=bool(torch.backends.cudnn.benchmark)
for key in ('torch','transformers','cuda','dtype','batch_size','checkpoint','layer_path','toolkit'):
    assert runtime[key]==reference['runtime'][key], (
        f'旧实验与当前环境在 {key} 不同。请恢复原环境后继续；本入口不放宽原配对协议。'
        f' old={reference["runtime"][key] if key!="toolkit" else "saved source hashes"}; new={runtime[key] if key!="toolkit" else "current source hashes"}')
capture_identity=dict(protocol=PROTOCOL,runtime=runtime,reference=file_hash(OUTPUT_ROOT/'reference.json'),
    vectors=vector_sha,support=file_hash(support_file),images=file_hash(support_dir/'images.json'),
    prompt_example=repair_prompt('outcome',('left','balanced','right')))
freeze(OUTPUT_ROOT/'capture_identity.json',capture_identity)
conditions_per_example=35 if MODEL_KEY=='qwen27b' else 6
print('需要保存的条件记录：',N_FAMILIES*8*conditions_per_example,'；每个场景/映射完成后保存，可断点继续。')

# %% COMMON: Capture all predeclared conditions with the current unedited answer
records=[]
progress=tqdm(families,desc='Independent confirmation',unit='family')
for family in progress:
    fid=family['family_id']
    for scene in family['scenes']:
        for variant,semantics in two_mappings(scene,fid).items():
            ex=new_example(scene['scene_id'],images/scene['image_filename'],scene['features'],semantics,variant,'independent_fixed_confirmation')
            def evaluate(ex=ex,scene=scene,semantics=semantics,variant=variant,family=family):
                clean,h=final_forward(ex,CAPTURE_LAYERS)
                sign=1. if ex.net_torque>0 else -1.
                assert ex.net_torque!=0
                opposite='right' if ex.gold_semantic=='left' else 'left'
                code=CODE_DIRECTIONS.get(code_key(LABELS[semantics.index(opposite)],ex.gold_label))
                conditions=intervention_conditions(MODEL_KEY,sign,UPSTREAM_PAYLOAD,SEMANTIC_PAYLOAD,code,h,
                    [COHORT_SEED,family['family_id'],scene['scene_id'],variant],RANDOM_SEEDS)
                rows=[]
                for condition,seed,operations in conditions:
                    progress.set_postfix_str(f'{scene["scene_id"][-15:]} {variant} {condition}/{seed}')
                    # The zero condition IS the just-captured untreated forward, not a historical label.
                    record=clean if condition=='zero' else final_forward(ex,operations=operations)[0]
                    rows.append(dict(family_id=family['family_id'],stratum=family['stratum'],scene_id=scene['scene_id'],
                        variant=variant,mirror=scene['mirror'],style=scene['style'],physical_outcome=ex.gold_semantic,
                        condition=condition,random_seed=seed,correct=bool(record['correct']),baseline_correct=bool(clean['correct']),
                        accuracy_change=float(record['correct'])-float(clean['correct']),margin_change=correct_margin(record)-correct_margin(clean),
                        answer_changed=record['predicted_semantic']!=clean['predicted_semantic'],
                        repaired=not clean['correct'] and record['correct'],harmed=clean['correct'] and not record['correct'],
                        valid=all(a.get('matched',True) for a in record['write_audits'].values()),
                        clean_record=clean,record=record))
                return audit_paired_writes(rows,MODEL_KEY,NORM_RELATIVE_TOLERANCE)
            identity=dict(capture=capture_identity,example=example_stamp(ex))
            records.extend(cached_json(OUTPUT_ROOT/'trial_shards',ex.example_id,identity,evaluate))
progress.close()
save_json(ANALYSIS_DIR/'intervention_records.json',records)
frame=pd.DataFrame([{k:v for k,v in r.items() if k not in ('record','clean_record')} for r in records])
validate_trial_grid(frame,families,MODEL_KEY)
frame.to_csv(ANALYSIS_DIR/'all_conditions.csv',index=False)
frame[~frame.valid].to_csv(ANALYSIS_DIR/'technical_failures.csv',index=False)
print('采集完整；接下来一次性分析全部预定家族，不依结果扩样。')

# %% COMMON: Analyse fixed paired comparisons at the family level
tests,family_means=test_family_contrasts(frame,MODEL_KEY)
tests.to_csv(ANALYSIS_DIR/'primary_tests.csv',index=False)
family_means.to_csv(ANALYSIS_DIR/'family_metrics.csv',index=False)
valid_ids=set(frame.groupby('family_id').valid.all().loc[lambda x:x].index)
if valid_ids:
    sensitivity,_=test_family_contrasts(frame[frame.family_id.isin(valid_ids)],MODEL_KEY,'all_controls_valid_sensitivity')
    sensitivity.to_csv(ANALYSIS_DIR/'valid_family_sensitivity.csv',index=False)
else:
    pd.DataFrame(columns=tests.columns).to_csv(ANALYSIS_DIR/'valid_family_sensitivity.csv',index=False)
# Secondary subgroup summaries average seeds within a scene/mapping before averaging within family.
subgroups=[]
for group_column in ('baseline_correct','physical_outcome','stratum'):
    for (group_value,condition),sub in frame.groupby([group_column,'condition']):
        metrics=['accuracy_change','margin_change','repaired','harmed','correct']
        units=sub.groupby(['family_id','scene_id','variant'])[metrics].mean().reset_index()
        means=units.groupby('family_id')[metrics].mean()
        subgroups.append(dict(group=group_column,value=str(group_value),condition=condition,families=len(means),
            scene_mapping_units=len(units),repaired_units_seed_average=float(units.repaired.sum()),
            harmed_units_seed_average=float(units.harmed.sum()),**{m:float(means[m].mean()) for m in metrics}))
pd.DataFrame(subgroups).to_csv(ANALYSIS_DIR/'subgroup_descriptive.csv',index=False)
counts=[]
for condition,sub in frame.groupby('condition'):
    units=sub.groupby(['family_id','scene_id','variant'])[['repaired','harmed','answer_changed','baseline_correct','correct']].mean()
    counts.append(dict(condition=condition,scene_mapping_units=len(units),original_wrong_units=float((1-units.baseline_correct).sum()),
        original_correct_units=float(units.baseline_correct.sum()),repaired_units_seed_average=float(units.repaired.sum()),
        harmed_units_seed_average=float(units.harmed.sum()),answer_changed_units_seed_average=float(units.answer_changed.sum())))
counts=pd.DataFrame(counts);counts.to_csv(ANALYSIS_DIR/'repair_and_harm_counts.csv',index=False)
if MODEL_KEY=='qwen27b':
    # Direct late-edit comparison is secondary and cannot establish restoration of native physics.
    secondary=[]
    for metric in ('accuracy_change','margin_change'):
        a=family_means[family_means.condition.eq('bypass')];b=family_means[family_means.condition.eq('b')]
        secondary.append(dict(contrast='bypass_minus_late_only',metric=metric,
            **paired_summary(paired(a,b,'family_id',metric),['bypass',metric],two_sided=True)))
    secondary=pd.DataFrame(secondary);secondary['q_value']=bh(secondary.p_value)
    secondary.to_csv(ANALYSIS_DIR/'secondary_bypass.csv',index=False)
decision=confirmation_decisions(tests,frame,MODEL_KEY)
result=dict(status='complete',model=MODEL_KEY,families=len(families),condition_records=len(frame),
    all_controls_valid=bool(frame.valid.all()),fully_valid_families=len(valid_ids),primary_tests=len(tests),decision=decision,
    protocol_sha256=file_hash(OUTPUT_ROOT/'protocol.json'),capture_sha256=file_hash(OUTPUT_ROOT/'capture_identity.json'),
    intervention_records_sha256=file_hash(ANALYSIS_DIR/'intervention_records.json'))
save_json(ANALYSIS_DIR/'RUN_COMPLETE.json',result)
lines=['# 独立场景固定干预结果','',f'模型：{MODEL_ID}；独立物理家族：{len(families)}。',
       '准确率和答案分数分别判定；所有预定家族进入主分析。','']
for row in tests.to_dict('records'):
    if row['metric']=='accuracy_change':
        lines.append(f'- {row["contrast"]}：{100*row["mean"]:.2f} 个百分点，95% 区间 [{100*row["lower"]:.2f}, {100*row["upper"]:.2f}]，q={row["q_value"]:.4g}。')
lines+=['','## 修好与损坏（场景×映射；随机条件先对种子平均）']
for row in counts.to_dict('records'):
    if row['condition'] in ('u','a','targeted'):
        lines.append(f'- {row["condition"]}：修好 {row["repaired_units_seed_average"]:g}/{row["original_wrong_units"]:g} 道原错误；损坏 {row["harmed_units_seed_average"]:g}/{row["original_correct_units"]:g} 道原正确。')
lines+=['','## 固定判定',json.dumps(decision,ensure_ascii=False,indent=2),
    '',f'实际幅度/复位检查失败记录：{int((~frame.valid).sum())}；全部技术检查通过的家族：{len(valid_ids)}。',
    '主比较固定为 14 项（Qwen-27B）或 4 项（Gemma），分别在各自实验内 BH 校正。',
    '区间按家族 bootstrap；p 值使用家族配对符号翻转检验，其零假设要求差值符号可交换。',
    '有效家族敏感性分析和子组描述不能替代主总体的判定。',
    '方向根据真实物理答案施加，属于监督式因果干预；不代表无需真值的推理算法或整个物理计算链被修复。']
(ANALYSIS_DIR/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
display(tests);display(counts);print('完成：',ANALYSIS_DIR);print(decision)

# %% AUDIT: Separate norm failures from readout leakage in existing controls
RUN_ID='main'
SOURCE_RUN_ID='main'
MODEL_SLUGS={'qwen9b':'Qwen__Qwen3.5-9B','gemma12b':'google__gemma-4-12B-it','qwen27b':'Qwen__Qwen3.5-27B__373612cb'}
OUTPUT_ROOT=PROJECT_ROOT/'results/cross_model_independent_confirmation/targeted_audit'/RUN_ID
OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
READOUT_ROOT=PROJECT_ROOT/'results/cross_model_readout_correction_v2/models'
inputs={};coverage=[];failures=[];control_effects=[];failure_summary=[]
for model,slug in MODEL_SLUGS.items():
    path=READOUT_ROOT/slug/'analysis/intervention_records.json'
    coverage.append(dict(stage='orthogonal',model=model,path=str(path),available=path.is_file()))
    if not path.is_file():continue
    inputs[str(path)]=file_hash(path)
    records=read_json(path);detail=orthogonal_failure_table(records);detail['model']=model
    failures.extend(detail.to_dict('records'))
    counts=detail.failure_kind.value_counts()
    failure_summary.append(dict(model=model,total=len(detail),**{k:int(counts.get(k,0)) for k in ('valid','norm_only','readout_only','both','missing_audit')}))
    frame=pd.DataFrame([{**r,'margin':correct_margin(r)} for r in records])
    zero=frame[frame.condition.eq('zero')].copy();oracle=frame[frame.condition.eq('oracle')].copy()
    assert zero.scene_id.is_unique and oracle.scene_id.is_unique and set(zero.scene_id)==set(oracle.scene_id)
    valid=set(detail.groupby('scene_id').valid.all().loc[lambda x:x].index)&set(oracle[oracle.achieved.eq(True)].scene_id)
    control=frame[frame.condition.eq('orthogonal')].groupby('scene_id')[['correct','margin']].mean().reset_index()
    attempted=set(control.scene_id)
    for scope,ids in [('all_attempted',attempted),('oracle_achieved_and_four_controls_valid',valid)]:
        if not ids:continue
        for metric in ('correct','margin'):
            values=paired(oracle[oracle.scene_id.isin(ids)],control[control.scene_id.isin(ids)],'scene_id',metric)
            control_effects.append(dict(model=model,scope=scope,metric=metric,**paired_summary(values,[model,scope,metric],two_sided=True)))
pd.DataFrame(failures).to_csv(OUTPUT_ROOT/'orthogonal_record_audit.csv',index=False)
pd.DataFrame(failure_summary).to_csv(OUTPUT_ROOT/'orthogonal_failure_causes.csv',index=False)
pd.DataFrame(control_effects).to_csv(OUTPUT_ROOT/'orthogonal_sensitivity.csv',index=False)
display(pd.DataFrame(failure_summary))

# %% AUDIT: Directly compare historical error and correct groups on the same matches
source=PROJECT_ROOT/'results/cross_model_followup_v1/equal_blocking'/SOURCE_RUN_ID
path=source/'analysis/intervention_records.json'
coverage.append(dict(stage='matched_blocking',model='qwen27b',path=str(path),available=path.is_file()))
interactions=[];family_rows=[];bypass_rows=[];baseline_status=[]
if path.is_file():
    inputs[str(path)]=file_hash(path)
    raw=read_json(path);f=pd.DataFrame(raw)
    assert not f.duplicated(['scene_id','mapping_variant','condition']).any()
    paths=[('ua','upstream_repair_block_l49','upstream_equal_l49'),
           ('ub','upstream_repair_block_l55','upstream_equal_l55'),('ab','l49_repair_block_l55','l49_equal_l55')]
    for name,target,prefix in paths:
        group_effects={};valid_by_group={}
        for group in ('right_error','both_correct'):
            sub=f[f.side.eq('right')&f.cohort.eq(group)]
            a=sub[sub.condition.eq(target)]
            b=sub[sub.condition.str.startswith(prefix+'_seed')]
            assert a.groupby('match_id').size().eq(2).all() and a.match_id.nunique()==32
            assert b.groupby(['match_id','scene_id','mapping_variant']).size().eq(4).all()
            assert set(b.match_id)==set(a.match_id)
            valid_by_group[group]=set(b.groupby('match_id').norm_valid.all().loc[lambda x:x].index)
            group_effects[group]={}
            for metric in ('rescued','margin_shift'):
                left=b.groupby('match_id')[metric].mean();right=a.groupby('match_id')[metric].mean()
                assert set(left.index)==set(right.index)
                difference=left-right
                group_effects[group][metric]=difference
                family_rows.extend(dict(path=name,group=group,match_id=i,metric=metric,effect=float(v)) for i,v in difference.items())
        common_valid=valid_by_group['right_error']&valid_by_group['both_correct']
        for metric in ('rescued','margin_shift'):
            difference=group_effects['right_error'][metric]-group_effects['both_correct'][metric]
            assert len(difference)==32 and difference.notna().all()
            for scope,values in [('all_attempted',difference),('all_controls_valid',difference[difference.index.isin(common_valid)])]:
                if len(values):interactions.append(dict(path=name,metric='accuracy' if metric=='rescued' else 'side_margin',scope=scope,
                    **paired_summary(values.to_numpy(float),['historical_interaction',name,metric,scope],two_sided=True)))
    table=pd.DataFrame(interactions);table['q_value']=np.nan
    primary=table.scope.eq('all_attempted');assert primary.sum()==6
    table.loc[primary,'q_value']=bh(table.loc[primary,'p_value'])
    table.to_csv(OUTPUT_ROOT/'historical_group_interactions.csv',index=False)
    pd.DataFrame(family_rows).to_csv(OUTPUT_ROOT/'historical_matched_effects.csv',index=False)
    sub=f[f.side.eq('right')&f.cohort.eq('right_error')]
    for metric in ('rescued','margin_shift'):
        a=sub[sub.condition.eq('l55_bypass_after_l49_block')].groupby('match_id')[metric].mean().reset_index()
        b=sub[sub.condition.eq('l55_repair')].groupby('match_id')[metric].mean().reset_index()
        assert len(a)==len(b)==32
        bypass_rows.append(dict(metric='accuracy' if metric=='rescued' else 'side_margin',
            **paired_summary(paired(a,b,'match_id',metric),['historical_bypass',metric],two_sided=True)))
    bypass=pd.DataFrame(bypass_rows);bypass['q_value']=bh(bypass.p_value)
    bypass.to_csv(OUTPUT_ROOT/'historical_bypass_vs_late_only.csv',index=False)
    for (sid,variant),sub in f.groupby(['scene_id','mapping_variant']):
        # A two-candidate margin cannot recover a three-class untreated prediction.
        baseline_status.append(dict(scene_id=sid,variant=variant,status='unverified_no_current_three_class_clean_record',
            historical_status=str(sub.baseline_status.iloc[0]),clean_margin_available=bool(sub.clean_margin.notna().all())))
    for root in (source,PROJECT_ROOT/'results/qwen27b_sequential_handoff_independent_replication_v1'):
        for pattern in ('*baseline*.json','*clean*.json'):
            for p in root.rglob(pattern):
                coverage.append(dict(stage='baseline_archive_candidate_requires_provenance_check',model='qwen27b',path=str(p),available=True))
    display(table);display(bypass)
pd.DataFrame(baseline_status).to_csv(OUTPUT_ROOT/'historical_baseline_coverage.csv',index=False)

# %% AUDIT: Save coverage and identify whether the small GPU diagnostic is still needed
for model,slug in MODEL_SLUGS.items():
    folder=PROJECT_ROOT/'results/cross_model_independent_confirmation/readout_repeatability'/RUN_ID/slug/'analysis'
    coverage.append(dict(stage='same_configuration_repeats',model=model,path=str(folder/'RUN_COMPLETE.json'),available=(folder/'RUN_COMPLETE.json').is_file()))
    if (folder/'repeatability_summary.csv').is_file():
        print(model);display(pd.read_csv(folder/'repeatability_summary.csv'))
pd.DataFrame(coverage).to_csv(OUTPUT_ROOT/'coverage.csv',index=False)
save_json(OUTPUT_ROOT/'input_manifest.json',dict(implementation=IMPLEMENTATION_SHA256,files=inputs))
status='complete' if len(failure_summary)==3 and len(interactions)>0 else 'partial_missing_inputs'
save_json(OUTPUT_ROOT/'RUN_COMPLETE.json',dict(status=status,readout_models=[r['model'] for r in failure_summary],
    historical_interactions_available=bool(interactions),new_scientific_cohort=False))
lines=['# 既有记录的定点核查','',f'覆盖状态：{status}。','正交失败原因见 orthogonal_failure_causes.csv；缺失记录见 coverage.csv。',
    '组间比较直接计算同一匹配家族内的“旧错误角色作用−旧正确角色作用”，不是比较两组是否各自显著。',
    '历史 margin_shift 仍是正确侧相对相反侧的分数差，不与新实验的正确候选相对最高错误候选分数混用。',
    '旧字段 rescued 在这里按准确率处理，不能解释为相对当前未编辑答案修好的比例。',
    '旧阻断分片没有完整的当前未编辑三分类答案；仅有 clean_margin 时标为未核对，不能用历史标签或单独复位替代。',
    '这些比较是对已查看数据的补充分析，不是新的独立确认。',
    '若尚无同配置重复状态，使用第 4 份 notebook 对 Gemma/Qwen-9B 的旧开发场景做诊断；不会重新拟合读出器。']
(OUTPUT_ROOT/'REPORT_ZH.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
display(pd.DataFrame(coverage));print('核查输出：',OUTPUT_ROOT)

# %% REPEAT: Load the original frozen readout and nineteen development scenes
from visual_torque_toolkit.behavior import BehaviorRunConfig,PromptCondition,load_behavior_examples
READOUT_ROOT=PROJECT_ROOT/'results/cross_model_readout_correction_v2/models'/MODEL_SLUG
probe_path=READOUT_ROOT/'frozen_probe.json';old_protocol_path=READOUT_ROOT/'protocol.json'
old_runtime_path=READOUT_ROOT/'runtime_identity.json'
for p in (probe_path,old_protocol_path,old_runtime_path):assert p.is_file(),p
old_protocol=read_json(old_protocol_path);old_runtime=read_json(old_runtime_path)
assert old_protocol['model_id']==MODEL_ID and old_protocol['revision']==MODEL_REVISION
assert old_protocol['dtype']=='bfloat16' and old_protocol['batch_size']==1
probe=read_json(probe_path);SELECTED_LAYER=int(probe['layer']);CAPTURE_LAYERS=(SELECTED_LAYER,)
PROBE_MEAN=np.asarray(probe['mean'],dtype=np.float32);PROBE_SCALE=np.asarray(probe['scale'],dtype=np.float32)
PROBE_COEF=np.asarray(probe['coef'],dtype=np.float32);PROBE_INTERCEPT=float(probe['intercept'])
assert PROBE_MEAN.shape==PROBE_SCALE.shape==PROBE_COEF.shape and (PROBE_SCALE>0).all()
assert all(np.isfinite(x).all() for x in (PROBE_MEAN,PROBE_SCALE,PROBE_COEF))
BALANCE_THRESHOLD=float(old_protocol['balance_threshold']);READOUT_TOLERANCE=float(old_protocol['readout_tolerance'])
manifest_path=PROJECT_ROOT/'results/behavior_comparison_v3/activation_pilot_manifest.jsonl'
manifest=read_jsonl(manifest_path)
dev_ids={str(r['scene_id']) for r in manifest if r['split']=='probe_dev'}
assert len(dev_ids)==19 and len(manifest)==254
assert dev_ids<=set(probe['fit_scene_ids'])
cfg=BehaviorRunConfig(dataset_dir=DATASET_DIR,output_dir=OUTPUT_ROOT/'unused',model_id=MODEL_ID,model_revision=MODEL_REVISION,
    conditions=(PromptCondition.IMAGE_RULE,),question_kinds=('outcome',),splits=('probe_dev',),batch_size=1,generation_every=0,seed=20260820)
assert cfg.dataset_fingerprint()==old_runtime['dataset'], 'Dataset differs from the original E1 inputs.'
by_scene=defaultdict(list)
for ex in load_behavior_examples(cfg):
    if str(ex.scene_id) in dev_ids:by_scene[str(ex.scene_id)].append(ex)
assert set(by_scene)==dev_ids and all(len(v)==6 and len({e.variant for e in v})==6 for v in by_scene.values())
for examples in by_scene.values():assert len({tuple(e.option_semantics) for e in examples})==6
print(f'只用旧开发集：19 场景 × 6 映射 × {REPEATS} 次独立前向。冻结读出层：{SELECTED_LAYER}。')

# %% REPEAT: Lock the current runtime without fitting or retuning the probe
adapter.model.eval()
runtime=current_runtime_identity()
assert runtime['torch']==old_runtime['torch'] and runtime['transformers']==old_runtime['transformers'], '恢复旧 E1 的 torch/transformers 版本后运行此同配置诊断。'
toolkit_identity={str(p.relative_to(TOOLKIT_ROOT)):'sha256:'+file_hash(p) for p in sorted((TOOLKIT_ROOT/'src/visual_torque_toolkit').rglob('*.py'))}
assert toolkit_identity==old_runtime['toolkit']
assert old_runtime['manifest']=='sha256:'+file_hash(manifest_path)
runtime.update(gpu=torch.cuda.get_device_name(0),cuda_capability=list(torch.cuda.get_device_capability(0)),
    attention_implementation=str(getattr(adapter.model.config,'_attn_implementation',None)),
    tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),cudnn_benchmark=bool(torch.backends.cudnn.benchmark))
repeat_identity=dict(protocol=PROTOCOL,runtime=runtime,probe=file_hash(probe_path),old_protocol=file_hash(old_protocol_path),
    old_runtime=file_hash(old_runtime_path),manifest=file_hash(manifest_path),
    inputs={s:[example_stamp(e) for e in sorted(by_scene[s],key=lambda e:e.variant)] for s in sorted(by_scene)})
freeze(OUTPUT_ROOT/'repeat_identity.json',repeat_identity)
print('原读出器已冻结；保存 BF16 前向状态的精确 float32 副本，不作 FP16 存储压缩。')

# %% REPEAT: Run three independent forwards per mapping and preserve raw states
repeat_records=[];states_by_scene={}
for sid in tqdm(sorted(by_scene),desc='Same-configuration independent repeats'):
    examples=sorted(by_scene[sid],key=lambda e:e.variant)
    array_path=OUTPUT_ROOT/'states'/(digest(sid)[:24]+'.npz')
    def compute(examples=examples):
        states=[];rows=[]
        for repeat in range(REPEATS):
            current=[]
            for ex in examples:
                record,h=forward_one(ex,(SELECTED_LAYER,))
                hidden=h[SELECTED_LAYER].astype('float32')
                current.append(hidden)
                rows.append(dict(scene_id=str(ex.scene_id),variant=ex.variant,repeat=repeat,
                    decoded=decode(hidden),decoded_outcome=probe_outcome(decode(hidden)),record=record))
            states.append(current)
        states=np.asarray(states,dtype='float32')
        assert states.shape[:2]==(REPEATS,6) and np.isfinite(states).all()
        # States and scores commit together. A crash cannot leave a completed array paired with different rerun scores.
        return dict(records=rows,states=states.tolist())
    cached=cached_json(OUTPUT_ROOT/'repeat_shards',sid,repeat_identity,compute)
    states=np.asarray(cached['states'],dtype='float32')
    frozen_arrays(array_path,{'states':states})
    states_by_scene[sid]=states
    repeat_records.extend(cached['records'])
assert len(repeat_records)==19*6*REPEATS
save_json(ANALYSIS_DIR/'repeat_records.json',repeat_records)

# %% REPEAT: Separate within-runtime variation from historical scene-mean differences
rows=[];scene_rows=[]
for sid in sorted(by_scene):
    records={(r['repeat'],r['variant']):r for r in repeat_records if r['scene_id']==sid}
    variants=[e.variant for e in sorted(by_scene[sid],key=lambda e:e.variant)]
    states=states_by_scene[sid]
    for a,b in combinations(range(REPEATS),2):
        for i,variant in enumerate(variants):
            left,right=records[a,variant],records[b,variant]
            x=np.array([left['record']['candidate_logprobs'][k] for k in LABELS])
            y=np.array([right['record']['candidate_logprobs'][k] for k in LABELS])
            decoded_delta=abs(left['decoded']-right['decoded'])
            contributions=(states[b,i].astype(float)-states[a,i].astype(float))*PROBE_COEF/PROBE_SCALE
            gap=float(np.max(abs((x-x.mean())-(y-y.mean()))))
            rows.append(dict(scene_id=sid,variant=variant,repeat_a=a,repeat_b=b,readout_absolute_change=decoded_delta,
                readout_over_tolerance=decoded_delta>READOUT_TOLERANCE,
                decoded_outcome_changed=left['decoded_outcome']!=right['decoded_outcome'],
                answer_changed=left['record']['predicted_semantic']!=right['record']['predicted_semantic'],
                centered_score_max_change=gap,hidden_max_change=float(np.max(abs(states[b,i]-states[a,i]))),
                effective_coordinate_l1=float(abs(contributions).sum()),
                largest_readout_contribution=float(abs(contributions).max())))
        x=decode(states[a].mean(0));y=decode(states[b].mean(0))
        scene_rows.append(dict(scene_id=sid,repeat_a=a,repeat_b=b,scene_mean_readout_change=abs(x-y),
            scene_mean_class_changed=probe_outcome(x)!=probe_outcome(y)))
changes=pd.DataFrame(rows);means=pd.DataFrame(scene_rows)
changes.to_csv(ANALYSIS_DIR/'within_runtime_pairs.csv',index=False)
means.to_csv(ANALYSIS_DIR/'within_runtime_scene_means.csv',index=False)
summary=[]
for metric in ('readout_absolute_change','readout_over_tolerance','decoded_outcome_changed','answer_changed','centered_score_max_change','hidden_max_change'):
    maxima=changes.groupby('scene_id')[metric].max()
    summary.append(dict(metric=metric,scenes=len(maxima),mean_of_scene_maxima=float(maxima.mean()),
        maximum=float(maxima.max()),nonzero_scenes=int(maxima.ne(0).sum())))
pd.DataFrame(summary).to_csv(ANALYSIS_DIR/'repeatability_summary.csv',index=False)
archive_comparison=[]
discovery=PROJECT_ROOT/'results'/MODEL_SLUG/'activation_discovery_v1'
archive_path=discovery/'scene_mean_activations.pt';config_path=discovery/'capture_config.json'
archive_status='missing_optional_original_discovery_archive'
if archive_path.is_file() and config_path.is_file():
    identity=read_json(config_path)['identity']
    assert identity['model_id']==MODEL_ID and identity['resolved_revision']==MODEL_REVISION
    assert identity['activation_manifest_sha256']=='sha256:'+file_hash(manifest_path)
    assert identity['dataset_fingerprint']==old_runtime['dataset']
    for path in (archive_path,config_path):
        expected=probe.get('fit_provenance',{}).get('files',{}).get(str(path))
        if expected is not None:assert expected=='sha256:'+file_hash(path), f'Original probe input changed: {path}'
    # This is a user-owned original archive; CPU-load only, no new fitting.
    archived=torch.load(archive_path,map_location='cpu',weights_only=False)
    meta=pd.DataFrame(archived['metadata']);assert meta.scene_id.astype(str).is_unique
    pos=list(map(int,archived['layer_indices'])).index(SELECTED_LAYER)
    old_h=archived['activations'][:,pos].float().numpy()
    for sid in sorted(by_scene):
        indices=np.flatnonzero(meta.scene_id.astype(str).eq(sid).to_numpy());assert len(indices)==1
        ix=int(indices[0]);assert meta.iloc[ix]['split']=='probe_dev'
        old_value=decode(old_h[ix])
        for repeat in range(REPEATS):
            current=states_by_scene[sid][repeat]
            exact_mean=current.mean(0)
            stored_mean=current.astype('float16').astype('float32').mean(0).astype('float16').astype('float32')
            assert np.isfinite(stored_mean).all(), 'FP16 storage diagnostic overflowed; preserve native results and inspect.'
            archive_comparison.append(dict(scene_id=sid,repeat=repeat,archived_readout=old_value,
                current_native_readout=decode(exact_mean),current_fp16_pipeline_readout=decode(stored_mean),
                archived_minus_current=old_value-decode(exact_mean),
                storage_roundtrip_change=decode(stored_mean)-decode(exact_mean)))
    archive_status='compared_historical_scene_means_separate_from_same_runtime_repeats'
    freeze(OUTPUT_ROOT/'archive_inputs.json',dict(archive=file_hash(archive_path),config=file_hash(config_path),identity=identity))
pd.DataFrame(archive_comparison).to_csv(ANALYSIS_DIR/'historical_scene_mean_comparison.csv',index=False)
result=dict(status='complete',model=MODEL_KEY,development_scenes=19,mappings_per_scene=6,repeats=REPEATS,
    forward_count=len(repeat_records),readout_over_tolerance=bool(changes.readout_over_tolerance.any()),
    any_answer_changed=bool(changes.answer_changed.any()),any_readout_class_changed=bool(changes.decoded_outcome_changed.any()),
    any_scene_mean_class_changed=bool(means.scene_mean_class_changed.any()),
    any_score_gap_over_0_15=bool(changes.centered_score_max_change.gt(.15).any()),archive_status=archive_status,
    probe_refit=False,new_scientific_cohort=False)
save_json(ANALYSIS_DIR/'RUN_COMPLETE.json',result)
lines=['# 读出重复性诊断','',json.dumps(result,ensure_ascii=False,indent=2),'',
    '同配置重复变化与历史缓存差异分开报告。没有重复变化不等于已解释旧缓存差异。',
    '本实验只复用旧开发场景，不新增科学测试集，不重新拟合预测器，也不修改旧实验结果。',
    '若读出变化超过原 0.05 容差或改变物理分类，先定位数值来源，再决定重跑哪些受影响条件。',
    '历史 scene-mean 对比保留其原捕获配置；不能仅凭相关差异把原因归给 FP16 存储。']
(ANALYSIS_DIR/'REPORT_ZH.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
display(pd.DataFrame(summary));display(pd.DataFrame(archive_comparison).head());print(result);print(ANALYSIS_DIR)
