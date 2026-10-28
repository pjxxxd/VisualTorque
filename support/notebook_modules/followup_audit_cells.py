"""Cells for the CPU notebook; all observations come from supplied result files."""

# %% Locate completed outputs and freeze their identities
RUN_ID = 'main'
MODEL_SLUGS = {'qwen9b':'Qwen__Qwen3.5-9B','gemma12b':'google__gemma-4-12B-it','qwen27b':'Qwen__Qwen3.5-27B__373612cb'}
OUTPUT_ROOT=PROJECT_ROOT/'results/cross_model_followup_v1/full_audit'/RUN_ID
OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
READOUT_ROOT=PROJECT_ROOT/'results/cross_model_readout_correction_v2/models'
TRANSPORT_ROOT=PROJECT_ROOT/'results/cross_model_cross_query_v1/models'
DEPTH_ROOT=PROJECT_ROOT/'results/cross_model_semantic_label_depth_v1/models'
coverage=[]; inputs={}; readout={}
for model,slug in MODEL_SLUGS.items():
    folder=READOUT_ROOT/slug/'analysis'; path=folder/'intervention_records.json'
    if not path.is_file():
        coverage.append(dict(model=model,stage='readout',status='missing',path=str(path)));continue
    records=read_json(path)
    assert isinstance(records,list) and records, path
    inputs[str(path)]=file_hash(path);readout[model]=records
    for name in ('result_summary.json','test_capture_source.json','paired_control_accuracy.csv','control_match_failures.csv','oracle_write_failures.csv','tilt_margin_changes.csv','summary_by_cohort.csv'):
        f=folder/name
        coverage.append(dict(model=model,stage=name,status='available' if f.is_file() else 'missing',path=str(f)))
        if f.is_file():inputs[str(f)]=file_hash(f)
pd.DataFrame(coverage).to_csv(OUTPUT_ROOT/'coverage.csv',index=False)
display(pd.DataFrame(coverage))
assert readout, 'No complete readout records found. Copy the analysis folders listed above into the existing Drive project.'

# %% Recalculate accuracy, answer flips, and continuous answer changes
summary_rows=[]; pair_frames={}; control_rows=[]; failure_rows=[]
for model,records in readout.items():
    frame=pd.DataFrame([{**r,'gold_margin':correct_margin(r)} for r in records])
    zero=frame[frame.condition.eq('zero')].copy();oracle=frame[frame.condition.eq('oracle')].copy()
    assert len(zero)==len(oracle)==304 and zero.scene_id.is_unique and oracle.scene_id.is_unique
    assert set(zero.scene_id)==set(oracle.scene_id)
    join=oracle.merge(zero[['scene_id','predicted_semantic','correct','gold_margin','decoded_post']],on='scene_id',suffixes=('','_zero'),validate='one_to_one')
    assert np.array_equal(join.original_model_correct.astype(bool),join.correct_zero.astype(bool)), 'Oracle baseline is not the paired zero condition'
    join['accuracy_change']=join.correct.astype(float)-join.correct_zero.astype(float)
    join['margin_change']=join.gold_margin-join.gold_margin_zero
    join['answer_changed']=join.predicted_semantic.ne(join.predicted_semantic_zero)
    join['repaired']=~join.correct_zero.astype(bool)&join.correct.astype(bool)
    join['harmed']=join.correct_zero.astype(bool)&~join.correct.astype(bool)
    join['readout_already_within_tolerance']=(join.decoded_post_zero-join.net_torque).abs().le(.05)
    tilt=join[join.net_torque.ne(0)].copy()
    zero_by_id={r['scene_id']:r for r in records if r['condition']=='zero'}
    tilt['correct_side_margin_change']=[
        ((semantic_scores(r)['left']-semantic_scores(r)['right'])-
         (semantic_scores(zero_by_id[r['scene_id']])['left']-semantic_scores(zero_by_id[r['scene_id']])['right']))
        *(1 if r['net_torque']>0 else -1) for r in tilt.to_dict('records')]
    tilt.to_csv(OUTPUT_ROOT/f'{model}_paired_tilt_margins.csv',index=False)
    if len(tilt):summary_rows.append(dict(model=model,group='tilt',metric='correct_side_margin_change',
        **bootstrap(tilt.correct_side_margin_change.to_numpy(float),[model,'original_tilt_margin'])))
    join.to_csv(OUTPUT_ROOT/f'{model}_paired_oracle.csv',index=False);pair_frames[model]=join
    for group,sub in [('all',join),('tilt',join[join.net_torque.ne(0)]),('balance',join[join.net_torque.eq(0)]),
                      ('achieved',join[join.achieved.eq(True)]),('readout_originally_correct',join[join.original_readout_correct.eq(True)]),
                      ('readout_originally_wrong',join[join.original_readout_correct.eq(False)]),
                      ('readout_already_within_0.05',join[join.readout_already_within_tolerance])]:
        if not len(sub):continue
        for metric in ('accuracy_change','margin_change'):
            summary_rows.append(dict(model=model,group=group,metric=metric,answer_flips=int(sub.answer_changed.sum()),
                repaired=int(sub.repaired.sum()),harmed=int(sub.harmed.sum()),achieved=int(sub.achieved.eq(True).sum()),
                **bootstrap(sub[metric].to_numpy(float),[model,group,metric])))
    controls=frame[frame.condition.isin(['random','orthogonal'])].copy()
    assert set(controls.condition)=={'random','orthogonal'}, 'Full random and orthogonal records are required'
    for kind,sub in controls.groupby('condition'):
        assert sub.groupby('scene_id').size().eq(4).all() and sub.scene_id.nunique()==72
        assert sub.groupby('scene_id').random_seed.nunique().eq(4).all()
        sub=sub.merge(zero[['scene_id','correct','gold_margin']],on='scene_id',suffixes=('','_zero'),validate='many_to_one')
        sub['accuracy_change']=sub.correct.astype(float)-sub.correct_zero.astype(float)
        sub['margin_change']=sub.gold_margin-sub.gold_margin_zero
        sub['valid']=sub.norm_matched.eq(True)&(sub.condition.ne('orthogonal')|sub.orthogonal_preserved.eq(True))
        failure_rows.extend(dict(model=model,**r) for r in sub[~sub.valid].to_dict('records'))
        means=sub.groupby('scene_id')[['accuracy_change','margin_change']].mean().reset_index()
        valid=set(sub.groupby('scene_id').valid.all().loc[lambda x:x].index)&set(join[join.achieved.eq(True)].scene_id)
        for scope,ids in [('all_attempted',set(means.scene_id)),('achieved_and_all_four_controls_valid',valid)]:
            if not ids:continue
            left=join[join.scene_id.isin(ids)];right=means[means.scene_id.isin(ids)]
            for metric in ('accuracy_change','margin_change'):
                control_rows.append(dict(model=model,control=kind,scope=scope,metric=metric,
                    invalid_records=int((~sub.valid).sum()),**bootstrap(paired(left,right,'scene_id',metric),[model,kind,scope,metric])))
pd.DataFrame(summary_rows).to_csv(OUTPUT_ROOT/'readout_answer_changes.csv',index=False)
pd.DataFrame(control_rows).to_csv(OUTPUT_ROOT/'oracle_vs_controls.csv',index=False)
pd.DataFrame(failure_rows).to_csv(OUTPUT_ROOT/'control_failures_rechecked.csv',index=False)
display(pd.DataFrame(summary_rows).query("group == 'all'"))
display(pd.DataFrame(control_rows))

# %% Inspect saved numerical diagnostics and actual write magnitudes
numerical=[];native=[];probe_rows=[]
for model,slug in MODEL_SLUGS.items():
    folder=READOUT_ROOT/slug/'analysis';path=folder/'test_capture_source.json'
    if path.is_file():
        obj=read_json(path)
        for r in obj.get('audit',[]):numerical.append(dict(model=model,capture_mode=obj.get('mode'),**r))
        diagnostic=Path(obj.get('decision_path','missing')).parent/'parity_diagnostics.csv'
        if diagnostic.is_file():inputs[str(diagnostic)]=file_hash(diagnostic)
    # The preceding E1 notebook stores the frozen scaler and ridge coefficients in JSON.
    frozen=READOUT_ROOT/slug/'frozen_probe.json'
    if frozen.is_file():
        obj=read_json(frozen);scale=np.asarray(obj['scale'],float);coef=np.asarray(obj['coef'],float)
        assert scale.ndim==coef.ndim==1 and scale.shape==coef.shape and (scale>0).all()
        effective=coef/scale
        probe_rows.append(dict(model=model,path=str(frozen),status='checked',layer=int(obj['layer']),
            minimum_scale=float(scale.min()),median_scale=float(np.median(scale)),
            effective_coefficient_norm=float(np.linalg.norm(effective)),maximum_effective_coefficient=float(np.abs(effective).max())))
        inputs[str(frozen)]=file_hash(frozen)
    else:
        probe_rows.append(dict(model=model,path=str(frozen),status='missing_frozen_probe_json'))
    # Locate frozen probe arrays by explicit keys; never guess tensor dimension order.
    probe_paths=sorted((READOUT_ROOT/slug).glob('**/*probe*.npz'))
    for path in probe_paths:
        with np.load(path,allow_pickle=False) as archive:
            keys=set(archive.files)
            scale_key=next((k for k in ('scale','scaler_scale','probe_scale') if k in keys),None)
            coef_key=next((k for k in ('coef','coefficient','probe_coef') if k in keys),None)
            if not scale_key or not coef_key:
                probe_rows.append(dict(model=model,path=str(path),status='unrecognized_array_keys',keys=sorted(keys)));continue
            scale=np.asarray(archive[scale_key],float).ravel();coef=np.asarray(archive[coef_key],float).ravel()
            assert scale.shape==coef.shape and (scale>0).all()
            effective=coef/scale
            probe_rows.append(dict(model=model,path=str(path),status='checked',minimum_scale=float(scale.min()),
                median_scale=float(np.median(scale)),effective_coefficient_norm=float(np.linalg.norm(effective)),
                maximum_effective_coefficient=float(np.abs(effective).max())))
        inputs[str(path)]=file_hash(path)
    for stage,root in [('cross_query',TRANSPORT_ROOT),('depth',DEPTH_ROOT)]:
        path=root/slug/'analysis/native_displacement_audit.csv'
        if not path.is_file():continue
        f=pd.read_csv(path);inputs[str(path)]=file_hash(path)
        for column in ('actual_delta_norm','requested_delta_norm','cast_error_norm'):assert column in f
        f['requested_actual_relative_gap']=(f.actual_delta_norm-f.requested_delta_norm).abs()/f.requested_delta_norm.clip(lower=1e-12)
        f['model']=model;f['stage']=stage;native.extend(f.to_dict('records'))
        # This gap is NOT donor/control equality. Compare paired interventions below.
        key=[c for c in ('family_id','layer_index','donor_task','target_task','mapping_type','variant') if c in f]
        condition='condition' if 'condition' in f else 'intervention'
        pairs=[('identity_donor','identity_random'),('diagnostic_donor','diagnostic_random'),('donor_delta_norm','random_norm')]
        comparisons=[]
        for a,b in pairs:
            l=f[f[condition].eq(a)];r=f[f[condition].eq(b)]
            if l.empty and r.empty:continue
            assert not l.duplicated(key).any() and not r.duplicated(key).any()
            joined=l.merge(r,on=key,suffixes=('_donor','_random'),validate='one_to_one',how='outer',indicator=True)
            assert joined['_merge'].eq('both').all(), f'Unpaired actual-norm records: {model}/{stage}'
            joined['actual_norm_relative_error']=(joined.actual_delta_norm_donor-joined.actual_delta_norm_random).abs()/joined.actual_delta_norm_donor.clip(lower=1e-12)
            joined['matched']=joined.actual_norm_relative_error.le(.02)
            comparisons.append(joined)
        if comparisons:
            out=pd.concat(comparisons,ignore_index=True)
            out.to_csv(OUTPUT_ROOT/f'{model}_{stage}_actual_control_norms.csv',index=False)
pd.DataFrame(numerical).to_csv(OUTPUT_ROOT/'archived_parity_diagnostics.csv',index=False)
pd.DataFrame(native).to_csv(OUTPUT_ROOT/'native_displacements.csv',index=False)
pd.DataFrame(probe_rows).to_csv(OUTPUT_ROOT/'probe_scale_audit.csv',index=False)
display(pd.DataFrame(numerical));display(pd.DataFrame(probe_rows))

# %% Direct paired source-question contrasts and precision planning
transport_rows=[];precision=[]
for model,slug in MODEL_SLUGS.items():
    path=TRANSPORT_ROOT/slug/'analysis/cross_query_estimands.csv'
    if not path.is_file():continue
    f=pd.read_csv(path);inputs[str(path)]=file_hash(path)
    assert {'family_id','donor_task','target_task','donor_aligned_effect'}<=set(f)
    assert not f.duplicated(['family_id','donor_task','target_task']).any()
    a=f.groupby(['family_id','donor_task']).donor_aligned_effect.mean().unstack('donor_task')
    assert a.notna().all().all() and {'weight_more','weight_less'}<=set(a)
    values=(a.weight_more-a.weight_less).to_numpy(float)
    transport_rows.append(dict(model=model,contrast='source_more_minus_source_less',**bootstrap(values,model+'source_contrast')))
    a.to_csv(OUTPUT_ROOT/f'{model}_source_question_effects.csv')
    for n in (48,64,96,128):
        precision.append(dict(model=model,metric='source_question_difference',planning_families=n,
            normal_approx_95pct_halfwidth=1.96*float(np.std(values,ddof=1))/np.sqrt(n),basis_families=len(values)))
    path=DEPTH_ROOT/slug/'analysis/family_depth_metrics.csv'
    if path.is_file():
        d=pd.read_csv(path);inputs[str(path)]=file_hash(path)
        for layer,sub in d.groupby('layer_index'):
            assert sub.family_id.is_unique
            for metric in ('semantic_effect','label_effect','identity_carrier'):
                for n in (64,96,128):
                    precision.append(dict(model=model,metric=metric,layer=int(layer),planning_families=n,
                        normal_approx_95pct_halfwidth=1.96*float(sub[metric].std(ddof=1))/np.sqrt(n),basis_families=len(sub)))
pd.DataFrame(transport_rows).to_csv(OUTPUT_ROOT/'source_question_paired_contrast.csv',index=False)
pd.DataFrame(precision).to_csv(OUTPUT_ROOT/'precision_planning.csv',index=False)
display(pd.DataFrame(transport_rows))
print('Precision estimates use historical variance; they are not power guarantees or new measurements.')

# %% Save a complete human-readable audit and reproducible input index
save_json(OUTPUT_ROOT/'input_manifest.json',dict(implementation=IMPLEMENTATION_SHA256,files=inputs))
lines=['# 完整记录分析','', '准确率变化、答案是否改变和答案分数变化分别报告。','']
for model,j in pair_frames.items():
    ci=bootstrap(j.accuracy_change.to_numpy(float),model+'report')
    lines += [f'## {model}',f'- 共 {len(j)} 场景；修正旧错误 {int(j.repaired.sum())}，新增错误 {int(j.harmed.sum())}，答案实际改变 {int(j.answer_changed.sum())}。',
              f'- 准确率变化 {100*ci["mean"]:.2f} 个百分点，95% 区间 [{100*ci["lower"]:.2f}, {100*ci["upper"]:.2f}]。']
lines += ['', '## 解释范围','随机和正交比较见 oracle_vs_controls.csv；actual_control_norms 表按成对实际位移核对。',
          '缓存偏差原因不能由最大差值单独确定；预测器参数缺失时，probe_scale_audit 不代表完成稳定性验证。',
          '来源问题对比是对已查看数据的补充分析。精度规划仅用于在新采集前锁定样本量。',
          '本 notebook 不自动增大新实验样本量，不加载模型，不修改输入结果。']
(OUTPUT_ROOT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
save_json(OUTPUT_ROOT/'RUN_COMPLETE.json',dict(status='analysis_complete',models=list(readout),input_identity=digest(inputs)))
print('报告与完整表格：',OUTPUT_ROOT)
