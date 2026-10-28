"""Cells for the single next-step CPU notebook."""

# %% Locate the completed experiments and set a separate audit destination
RUN_ID='main'  # 本次核查输出名称。
EXPERIMENT_RUN_ID='main'  # 上一轮 Qwen/Gemma 新场景实验的运行名称。
REPEAT_RUN_ID='main'  # 上一轮两份读出重复性诊断的运行名称。
AUTO_DOWNLOAD=True  # 最后自动下载一个结果包，便于带回完整明细。
BASE=PROJECT_ROOT/'results/cross_model_independent_confirmation'
OUTPUT_ROOT=BASE/'completion_audit'/RUN_ID
OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
MODEL_SLUGS={'qwen27b':'Qwen__Qwen3.5-27B__373612cb','gemma12b':'google__gemma-4-12B-it','qwen9b':'Qwen__Qwen3.5-9B'}
SCIENCE_ROOTS={
    'qwen27b':BASE/'qwen_three_path'/EXPERIMENT_RUN_ID/MODEL_SLUGS['qwen27b'],
    'gemma12b':BASE/'gemma_l31_candidate'/EXPERIMENT_RUN_ID/MODEL_SLUGS['gemma12b'],
}
assert all(OUTPUT_ROOT.resolve()!=p.resolve() and p.resolve() not in OUTPUT_ROOT.resolve().parents for p in SCIENCE_ROOTS.values())
inputs={};coverage=[];science={};repeat_results={};export_files=set();report_sections=[]
pd.set_option('display.max_columns',40)
pd.set_option('display.width',180)

def export_csv(name,frame):
    path=OUTPUT_ROOT/name
    frame.to_csv(path,index=False,encoding='utf-8-sig')
    export_files.add(path)
    return path

def export_json(name,value):
    path=OUTPUT_ROOT/name;save_json(path,value);export_files.add(path)
    return path

def show(title,frame,limit=40):
    print('\n'+title)
    if frame.empty:
        print('无记录。');return
    with pd.option_context('display.max_rows',limit+2,'display.max_columns',None,'display.max_colwidth',80):
        display(frame.head(limit))
    if len(frame)>limit:print(f'这里显示前 {limit} 行；完整 {len(frame)} 行会放入结果包。')

export_json('audit_configuration.json',dict(schema='independent-results-completion-audit-1.0',implementation=IMPLEMENTATION_SHA256,
    experiment_run=EXPERIMENT_RUN_ID,repeat_run=REPEAT_RUN_ID,changes_to_source_results=False,
    scope='CPU inspection of saved records; no inference; no new confirmatory tests',numpy=np.__version__,pandas=pd.__version__))
print('本 notebook 一次读取所有相关模型，无需切换 MODEL_KEY，无需 GPU。')
print('核查输出：',OUTPUT_ROOT)

# %% Verify the original records, pairings, primary statistics and saved decisions
numeric_checks=[]
for model,root in tqdm(SCIENCE_ROOTS.items(),desc='Reading completed science experiments'):
    try:
        bundle=load_science_result(root,model,inputs)
        if bundle['status']=='validated_complete':
            verify_record_metrics(bundle)
            comparison,family_means=reconcile_primary(bundle)
            bundle['primary_numeric_match']=bool(comparison.numeric_match.all())
            bundle['family_means_recomputed']=family_means
            export_csv(model+'_primary_reproduction_check.csv',comparison)
            export_csv(model+'_original_primary_tests.csv',bundle['primary'])
            export_json(model+'_original_completion.json',bundle['completion'])
            for name,table in bundle['optional'].items():export_csv(model+'_original_'+name+'.csv',table)
            numeric_checks.append(dict(model=model,primary_tests=len(comparison),all_primary_numbers_match=bool(comparison.numeric_match.all()),
                original_decision_preserved=True,independent_families=len(bundle['families']),records=len(bundle['records'])))
    except (AssertionError,KeyError,ValueError,TypeError,OSError) as error:
        bundle=dict(model=model,status='input_integrity_error',issues=[type(error).__name__+': '+str(error)])
    science[model]=bundle
    coverage.append(dict(stage='science',model=model,status=bundle['status'],source=str(root),issues='; '.join(bundle['issues'])))
    if bundle['status']!='validated_complete':print(model,':',bundle['status'],bundle['issues'])
export_csv('primary_statistics_reproduction.csv',pd.DataFrame(numeric_checks))
show('原主结果与完整记录的核对',pd.DataFrame(numeric_checks))
report_sections.append('## 原结果一致性\n\n'+csv_markdown(pd.DataFrame(numeric_checks)))
print('这一步复核原来的 14/4 项检验及判定，不改变检验族、方向、样本量或容差。')

# %% Identify every failed Qwen control and its actual physical edit size
qwen=science['qwen27b']
qwen_diagnosis=dict(status=qwen['status'])
if qwen['status']=='validated_complete':
    operations,trials=inspect_saved_writes(qwen)
    failure_summary=condition_failure_summary(trials)
    failed=trials[~trials.stored_valid|~trials.saved_fields_consistent].copy()
    export_csv('qwen_all_write_operations.csv',operations)
    export_csv('qwen_all_trial_validity.csv',trials)
    export_csv('qwen_failure_counts_by_condition.csv',failure_summary)
    export_csv('qwen_failed_trial_details.csv',failed)
    keys={(r.family_id,r.scene_id,r.variant,r.condition,int(r.random_seed)) for r in failed.itertuples()}
    failed_records=[r for r in qwen['records'] if (r['family_id'],r['scene_id'],r['variant'],r['condition'],r['random_seed']) in keys]
    export_json('qwen_failed_records_with_write_audits.json',failed_records)
    failed_ids=set(failed.family_id)
    export_json('qwen_failed_family_physics.json',[f for f in qwen['families'] if f['family_id'] in failed_ids])
    for condition in ('ua_equal','ub_equal','ab_equal'):
        sub=trials[trials.condition.eq(condition)]
        qwen_diagnosis[condition]=dict(attempted=len(sub),failed=int((~sub.stored_valid).sum()),
            affected_families=int(sub.loc[~sub.stored_valid,'family_id'].nunique()),
            local_match_failures=int(sub.local_match_failed.sum()),cross_match_failures=int(sub.cross_match_failed.sum()),
            inconsistent_audits=int((~sub.saved_fields_consistent).sum()))
    qwen_diagnosis.update(all_original_failed_records=int((~trials.stored_valid).sum()),
        all_affected_families=int(trials.loc[~trials.stored_valid,'family_id'].nunique()),
        all_audit_fields_consistent=bool(trials.saved_fields_consistent.all()),
        originally_fully_valid_families=qwen['completion']['fully_valid_families'],
        original_decision=qwen['completion']['decision'])
    qwen_diagnosis['status']='diagnosed_from_saved_records'
    show('全部条件的失败条数（local/cross 可能同时失败，不能直接相加）',failure_summary,limit=40)
    show('失败明细：local=单次修改内部，cross=相对实际执行的目标条件',failed[[
        'family_id','scene_id','variant','condition','random_seed','failure_reasons',
        'executed_target_delta_norm','actual_delta_norm','absolute_norm_gap','cross_relative_error']],limit=25)
    bad_ops=operations.merge(failed[['family_id','scene_id','variant','condition','random_seed']],
        on=['family_id','scene_id','variant','condition','random_seed'],how='inner',validate='many_to_one')
    export_csv('qwen_failed_write_operations.csv',bad_ops)
    show('失败记录对应的每层写入审计',bad_ops,limit=25)
    ub=qwen_diagnosis['ub_equal']
    print(f'L48→L55：{ub["failed"]}/{ub["attempted"]} 条随机移除记录未通过，涉及 {ub["affected_families"]} 个物理家族。')
    print(f'单次写入内部未匹配 {ub["local_match_failures"]} 条；与实际执行的指定阻断不匹配 {ub["cross_match_failures"]} 条。')
    print('以下结果只定位失败原因。原 2% 容差与原确认状态保持不变。')
    report_sections.append('## Qwen 随机对照的失败原因\n\n'+csv_markdown(failure_summary)+'\n\n'+
        f'L48→L55 随机移除失败 {ub["failed"]}/{ub["attempted"]} 条，涉及 {ub["affected_families"]} 个家族。\n\n'+
        '单次匹配失败与跨前向匹配失败分别计数，可能重叠。明细及原始 write_audits 已打包。')
export_json('qwen_technical_diagnosis.json',qwen_diagnosis)

# %% Show path-specific sensitivity without changing the original confirmation gates
if qwen['status']=='validated_complete':
    gates,sensitivity,influence=path_diagnostics(qwen,qwen['family_means_recomputed'])
    export_csv('qwen_original_path_gates.csv',gates)
    export_csv('qwen_path_valid_sensitivity_posthoc.csv',sensitivity)
    export_csv('qwen_family_effect_contributions.csv',influence)
    show('原来的三路径判定，逐项拆开显示',gates)
    view=sensitivity[sensitivity.metric.eq('accuracy_change')].copy()
    for column in ('mean','lower','upper'):view[column+'_percentage_points']=100*view[column]
    show('准确率：全体、路径对照有效组、路径对照无效组（事后敏感性）',
        view[['path','scope','n','mean_percentage_points','lower_percentage_points','upper_percentage_points']])
    print('只保留有效家族会改变分析样本；上述事后区间不能把原来的 False 改成 True。')
    report_sections.append('## 原判定与事后敏感性\n\n'+csv_markdown(gates)+'\n\n'+
        csv_markdown(view[['path','scope','n','mean_percentage_points','lower_percentage_points','upper_percentage_points']])+\
        '\n\n路径有效组是基于技术检查的事后敏感性分析，不替代原全体家族的主结果；失败组的区间仅用于描述。')

# %% Summarize which physical conditions improve, including repairs and harms
stratum_labels={'equal_weight_distance':'重量相同、距离不同','weight_distance_conflict':'重量与距离线索冲突',
    'multi_stack_sum':'多个物体，需要求和','aligned_cues':'重量和距离线索一致'}
for model,bundle in science.items():
    if bundle['status']!='validated_complete':continue
    summary=subgroup_summary(bundle)
    export_csv(model+'_all_subgroups_descriptive.csv',summary)
    selected=('u','a','b','bypass') if model=='qwen27b' else ('targeted','random')
    main=summary[summary.group.eq('stratum')&summary.condition.isin(selected)].copy()
    main['physical_condition']=main.value.map(stratum_labels).fillna(main.value)
    main['accuracy_change_pp']=100*main.accuracy_change
    main['accuracy_lower_pp']=100*main.accuracy_lower;main['accuracy_upper_pp']=100*main.accuracy_upper
    columns=['physical_condition','condition','families','scene_mapping_units','accuracy_change_pp',
             'accuracy_lower_pp','accuracy_upper_pp','repaired_units_seed_average','harmed_units_seed_average']
    show(model+'：四类物理场景分别表现如何',main[columns])
    show(model+'：左右、镜像配色与选项映射的完整描述另存 CSV',
        summary[summary.group.eq('physical_outcome')&summary.condition.isin(selected)][
            ['value','condition','families','scene_mapping_units','accuracy_change','margin_change']])
    report_sections.append('## '+model+'：物理条件分组\n\n'+csv_markdown(main[columns])+\
        '\n\n各子组保留家族内配对与种子平均；区间为事后描述，不重新筛选候选或改变主结论。')

# %% Read all nineteen historical readout comparisons for both models
repeat_summary=[];archive_summary=[];config_rows=[]
for model in ('gemma12b','qwen9b'):
    slug=MODEL_SLUGS[model];root=BASE/'readout_repeatability'/REPEAT_RUN_ID/slug
    try:
        item=load_repeat_result(root,PROJECT_ROOT,model,slug,inputs)
        if item['status']=='validated_complete':
            repeat_summary.append(item['repeat_summary'])
            if item['archive_summary']:archive_summary.append(dict(model=model,**item['archive_summary']))
            config_rows.extend(item['config_rows'])
            export_csv(model+'_historical_readout_all_57_rows.csv',item['historical'])
            export_csv(model+'_historical_readout_19_scene_summary.csv',item['by_scene'])
            export_json(model+'_repeat_original_completion.json',item['completion'])
            shown=item['by_scene'].sort_values('maximum_absolute_archive_gap',ascending=False) if not item['by_scene'].empty else item['by_scene']
            show(model+'：完整 19 场景的旧新读出差异',shown,limit=19)
    except (AssertionError,KeyError,ValueError,TypeError,OSError) as error:
        item=dict(model=model,status='input_integrity_error',issues=[type(error).__name__+': '+str(error)])
    repeat_results[model]=item
    coverage.append(dict(stage='repeatability',model=model,status=item['status'],source=str(root),issues='; '.join(item['issues'])))
    if item['status']!='validated_complete':print(model,':',item['status'],item['issues'])
export_csv('readout_current_repeatability_verified.csv',pd.DataFrame(repeat_summary))
export_csv('readout_archive_difference_summary.csv',pd.DataFrame(archive_summary))
export_csv('readout_available_capture_configurations.csv',pd.DataFrame(config_rows))
show('当前配置的重复性',pd.DataFrame(repeat_summary))
show('完整 19 场景：旧新差异与存储往返变化分别汇总',pd.DataFrame(archive_summary))
show('旧捕获和当前运行中可核对的配置字段（只作为线索）',pd.DataFrame(config_rows),limit=35)
report_sections.append('## 读出重复性与历史捕获差异\n\n'+csv_markdown(pd.DataFrame(repeat_summary))+'\n\n'+
    csv_markdown(pd.DataFrame(archive_summary))+'\n\n统计单位为 19 个不同场景，不是 57 次重复。配置差异是排查线索，不自动证明因果。')

# %% Save a complete report and download one compact evidence package
import zipfile
from datetime import datetime,timezone
coverage_frame=pd.DataFrame(coverage)
export_csv('coverage.csv',coverage_frame)
export_json('input_manifest.json',dict(implementation=IMPLEMENTATION_SHA256,files=inputs))
all_inputs_unchanged=all(Path(p).is_file() and file_hash(p)==sha for p,sha in inputs.items())
assert all_inputs_unchanged,'核查过程中输入发生变化，请在原实验写入结束后重新运行本 CPU 核查。'
complete=bool(len(coverage_frame)==4 and coverage_frame.status.eq('validated_complete').all())
numeric_ok=bool(len(numeric_checks)==2 and all(r['all_primary_numbers_match'] for r in numeric_checks))
audit_fields_ok=bool(qwen_diagnosis.get('all_audit_fields_consistent',False))
status='complete' if complete and numeric_ok and audit_fields_ok else 'complete_with_items_to_resolve'
summary=dict(status=status,utc_time=datetime.now(timezone.utc).isoformat(),coverage=coverage,
    primary_statistics_reproduced=numeric_ok,qwen_technical_diagnosis=qwen_diagnosis,
    repeatability=repeat_summary,historical_readout=archive_summary,input_files_unchanged=all_inputs_unchanged,
    gpu_forward_count=0,original_scientific_decisions_unchanged=True)
export_json('AUDIT_COMPLETE.json',summary)
lines=['# 独立确认结果的补充核查','',
    '本次只读取已保存记录，没有运行模型，没有修改原样本、方向、剂量、容差或确认状态。','',
    '## 覆盖情况','',csv_markdown(coverage_frame),'',*report_sections,'',
    '## 使用范围','',
    '本报告区分单次写入匹配、跨前向实际幅度匹配与复位检查。技术失败、字段不一致和文件缺失分别保留。',
    '主结果重算使用原固定检验族；路径有效组与物理子组是事后描述，不是新的独立确认。',
    '先依据失败原因判断是否需要技术修正，再考虑是否需要重新采集。此 notebook 不自动启动 GPU 重跑。',
    '结果包包含完整失败明细、原 write_audits、家族物理配置、原完成状态、原主表和本次汇总。']
report_path=OUTPUT_ROOT/'REPORT_ZH.md';report_path.write_text('\n\n'.join(lines)+'\n',encoding='utf-8');export_files.add(report_path)
# Include only this run's registered outputs, never stale files from an earlier partial audit.
archive=OUTPUT_ROOT/'VisualTorque_Independent_Results_Audit_Output.zip'
temp=archive.with_suffix('.zip.tmp')
with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(export_files):bundle.write(path,path.name)
temp.replace(archive)
show('最终覆盖状态',coverage_frame)
print(json.dumps(summary,ensure_ascii=False,indent=2))
print('报告：',report_path)
print('请带回这个结果包：',archive)
if AUTO_DOWNLOAD:
    try:
        from google.colab import files
        files.download(str(archive))
    except Exception as error:
        print('自动下载未触发，可从左侧 Files 或上述 Drive 路径手动下载：',type(error).__name__,str(error))
