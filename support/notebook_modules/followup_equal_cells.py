"""Additional same-actual-magnitude controls for the exact archived Qwen route."""

# %% Load the existing cohort without repeating behavior selection
from visual_torque_toolkit.behavior.data import BehaviorExample
from visual_torque_toolkit.behavior import PromptCondition
import shutil
OLD_ROOT=PROJECT_ROOT/'results/qwen27b_sequential_handoff_independent_replication_v1'
for name in ('frozen_scene_manifest.jsonl','frozen_match_manifest.jsonl','frozen_protocol.json','RUN_COMPLETE.json'):
    assert (OLD_ROOT/name).is_file(),OLD_ROOT/name
scene_manifest=read_jsonl(OLD_ROOT/'frozen_scene_manifest.jsonl')
match_manifest=read_jsonl(OLD_ROOT/'frozen_match_manifest.jsonl')
MATCHED_UNITS=len(match_manifest)
assert MATCHED_UNITS==32 and len(scene_manifest)==4*MATCHED_UNITS
assert Counter((r['cohort'],r['side']) for r in scene_manifest)=={(c,s):32 for c in COHORTS for s in SIDES}
RUNTIME_IMAGE_DIR.mkdir(parents=True,exist_ok=True)
outcome_example_by_scene={};image_hashes={}
for row in scene_manifest:
    src=OLD_ROOT/'selected_images'/Path(row['image_path']).name;assert src.is_file(),src
    dst=RUNTIME_IMAGE_DIR/src.name
    if not dst.is_file() or file_hash(dst)!=file_hash(src):shutil.copy2(src,dst)
    image_hashes[str(src)]=file_hash(src)
    semantics=('left','balanced','right');gold=row['exact_semantic']
    outcome_example_by_scene[row['scene_id']]=BehaviorExample(example_id=row['scene_id'],question_id=row['scene_id'],
        scene_id=row['scene_id'],condition=PromptCondition.IMAGE_RULE,question_kind='outcome',variant='base',split='historical_control_extension',
        problem_type=row['diagnostic_cell'],outcome=gold,net_torque=int(row['net_torque']),prompt='replaced by original target prompt',
        image_path=dst,labels=LABELS,option_text=semantics,option_semantics=semantics,gold_label=LABELS[semantics.index(gold)],gold_semantic=gold)
direction_support=read_jsonl(DEPTH_SUPPORT_PATH)
DIRECTION_RUNTIME_DIR=RUNTIME_ROOT/'direction_images';DIRECTION_RUNTIME_DIR.mkdir(exist_ok=True)
for family in direction_support:
    for role in family['roles'].values():
        src=INVERSE_SCENE_DIR/role['image_filename'];dst=DIRECTION_RUNTIME_DIR/src.name;assert src.is_file(),src
        image_hashes[str(src)]=file_hash(src)
        if not dst.is_file() or file_hash(dst)!=file_hash(src):shutil.copy2(src,dst)
write_jsonl_atomic(SCENE_MANIFEST_PATH,scene_manifest)
write_jsonl_atomic(MATCH_MANIFEST_PATH,match_manifest)
print('Historical cohort:',MATCHED_UNITS,'families. No re-selection and no new-family claim.')

# %% Extend the original operations with four actual-magnitude matched random removals
legacy_build_path_operations=build_path_operations
def build_path_operations(row,variant,mapping,untreated_selected):
    operations,axes=legacy_build_path_operations(row,variant,mapping,untreated_selected)
    for seed in RANDOM_SEEDS:
        for stem,parent,layer in [('upstream_equal_l49','upstream_repair_block_l49',SEMANTIC_LAYER),
                                  ('upstream_equal_l55','upstream_repair_block_l55',LABEL_LAYER),
                                  ('l49_equal_l55','l49_repair_block_l55',LABEL_LAYER)]:
            ops=dict(operations[parent]);old=ops[layer]
            ops[layer]=dict(old,action='equal_remove',random_key=[SEED,row['scene_id'],variant,stem,seed])
            operations[f'{stem}_seed{seed}']=ops
    return operations,axes

def forward_path_condition(example,operations,untreated_selected,metadata):
    prepared=prepare_examples([example]);mask=final_mask(prepared)
    captured={};removed={};audits={};handles=[]
    def make_hook(layer):
        operation=operations.get(layer)
        def hook(_module,_inputs,output):
            hidden=output_hidden_tensor(output);m=mask.to(hidden.device);pre=hidden[0,m];post=pre
            assert int(m.sum())==1 and hidden.dtype==torch.bfloat16
            if operation is not None:
                action=operation['action']
                if action=='add':
                    payload=operation['payload'].to(device=pre.device,dtype=pre.dtype)
                    post=pre+payload;audits[layer]=dict(action=action,matched=True,requested_norm=float(payload.float().norm()))
                elif action in ('clamp_axis','equal_remove'):
                    reference,a=clamp_selected_coordinate(pre,operation['untreated'],operation['axis'])
                    removed[layer]=a;reference_norm=float((reference.float()-pre.float()).norm())
                    if action=='clamp_axis':
                        post=reference;audits[layer]=dict(action=action,matched=True,reference_actual_norm=reference_norm)
                    else:
                        h=pre.detach().float().cpu().numpy()
                        direction=-np.sign(a if a!=0 else 1)*random_unit(h.shape,operation['random_key'])
                        cast=lambda x:torch.as_tensor(np.asarray(x).copy(),dtype=pre.dtype).float().numpy()
                        value,audit=native_norm_edit(h,direction,reference_norm,cast,tolerance=NORM_RELATIVE_TOLERANCE)
                        post=torch.as_tensor(value,device=pre.device,dtype=pre.dtype)
                        audits[layer]=dict(action=action,reference_actual_norm=reference_norm,**audit)
                else:raise ValueError(action)
                audits.setdefault(layer,dict(action=action,matched=True))
                audits[layer]['actual_norm']=float((post.float()-pre.float()).norm())
                audits[layer]['hidden_norm']=float(pre.float().norm())
                changed=hidden.clone();changed[0,m]=post
            captured[layer]=post.detach().cpu().clone()
            return replace_output_hidden(output,changed) if operation is not None else None
        return hook
    for layer in sorted(set(operations)|{SEMANTIC_LAYER,LABEL_LAYER}):
        handles.append(DECODER_LAYERS[layer].register_forward_hook(make_hook(layer)))
    try:
        with torch.inference_mode():out=adapter.model(**prepared['inputs'],use_cache=False)
    finally:
        for h in handles:h.remove()
    record=score_records(prepared,out.logits)[0]
    record.update(metadata,patched_tokens=int(mask.sum()),write_audits={str(k):v for k,v in audits.items()},
        norm_valid=all(v.get('matched',True) for v in audits.values()),
        removed_projection_l49=float(removed.get(SEMANTIC_LAYER,0.)),removed_projection_l55=float(removed.get(LABEL_LAYER,0.)))
    return record,captured

# %% Freeze reconstructed vectors and input/runtime provenance before interventions
arrays={'upstream':UPSTREAM_DIRECTION.float().numpy().ravel(),
        'semantic':SEMANTIC_DIRECTION.float().numpy().ravel(),
        **{'code_'+a+'_'+b:v.float().numpy().ravel() for (a,b),v in CODE_DIRECTIONS.items()}}
vector_path=OUTPUT_ROOT/'frozen_directions.npz'
if vector_path.is_file():
    with np.load(vector_path,allow_pickle=False) as old:
        assert set(old.files)==set(arrays) and all(np.array_equal(old[k],v) for k,v in arrays.items()), 'Frozen direction reconstruction changed'
else:np.savez_compressed(vector_path,**arrays)
import transformers
runtime=dict(torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,dtype=MODEL_DTYPE,
    batch_size=1,checkpoint=MODEL_REVISION,layer_path=LAYER_PATH,
    toolkit={str(p.relative_to(TOOLKIT_ROOT)):file_hash(p) for p in sorted((TOOLKIT_ROOT/'src').rglob('*.py'))})
input_files={str(p):file_hash(p) for p in (OLD_ROOT/'frozen_scene_manifest.jsonl',OLD_ROOT/'frozen_match_manifest.jsonl',
    OLD_ROOT/'frozen_protocol.json',PARENT_ROUTE_PATH,PARENT_DIRECTION_PATH,DEPTH_SUPPORT_PATH,AUDIT_OUTCOME_MANIFEST_PATH)}
input_files.update(image_hashes)
input_files.update({str(p):file_hash(p) for p in AUDIT_OUTCOME_SHARD_DIR.glob('outcome_*.pt')})
freeze(OUTPUT_ROOT/'runtime_and_inputs.json',dict(runtime=runtime,inputs=input_files,implementation=IMPLEMENTATION_SHA256,
    directions=file_hash(vector_path),conditions=list(ALL_CONDITIONS)))
# This explicit reference can be used on the separate new shared confirmation cohort.
reference=dict(model='qwen27b',model_id=MODEL_ID,revision=MODEL_REVISION,eligible=True,
    upstream_layer=UPSTREAM_LAYER,downstream_layer=LABEL_LAYER,dose=MATCHED_UPSTREAM_ALPHA,
    directions_sha256=file_hash(vector_path),method='historical_pls4_prototype_frozen_reference',runtime=runtime,
    evidence_role='reference_selected_in_prior_experiments_not_current_development')
freeze(OUTPUT_ROOT/'selected_route.json',reference)

# %% Validate zero replay and one new removal path, then keep all scientific outcomes
smoke_row=scene_manifest[0];examples,mappings=build_target_examples(smoke_row)
ex=examples['forward'];base,h,m=capture_examples({'x':ex},CAPTURE_LAYERS)
untreated={layer:h['x'][layer][m['x']] for layer in (SEMANTIC_LAYER,LABEL_LAYER)}
ops,axes=build_path_operations(smoke_row,'forward',mappings['forward'],untreated)
zero,_=forward_path_condition(ex,{},untreated,{'condition':'zero'})
assert np.max(np.abs(centered_scores(zero)-centered_scores(base['x'])))<=.15
trial,_=forward_path_condition(ex,ops['upstream_equal_l49_seed0'],untreated,{'condition':'technical_write_check'})
print('One-path numerical check:',trial['write_audits'])
# A discrete norm-matching failure remains an explicit observation, not a relaxed threshold.

# %% Analyse direct same-magnitude contrasts without relabelling the old cohort
all_records=[]
for path in sorted(SHARD_DIR.glob('*.json')):
    payload=read_json(path)
    assert payload['capture_identity_sha256']==CAPTURE_SHA256
    all_records.extend(payload['records'])
frame=pd.DataFrame([{k:v for k,v in r.items() if k not in ('candidate_logprobs','candidate_probabilities','write_audits')} for r in all_records])
save_json(ANALYSIS_DIR/'intervention_records.json',all_records)
frame.to_csv(ANALYSIS_DIR/'all_conditions.csv',index=False)
frame[~frame.norm_valid.astype(bool)].to_csv(ANALYSIS_DIR/'norm_match_failures.csv',index=False)
contrasts=[('upstream_l49','upstream_repair_block_l49','upstream_equal_l49'),
           ('upstream_l55','upstream_repair_block_l55','upstream_equal_l55'),
           ('l49_l55','l49_repair_block_l55','l49_equal_l55')]
tests=[];family_tables=[]
for path_name,target,random_prefix in contrasts:
    for group,sub in [('right_error',frame[frame.cohort.eq('right_error')&frame.side.eq('right')]),
                      ('both_correct_right',frame[frame.cohort.eq('both_correct')&frame.side.eq('right')])]:
        selected=sub[sub.condition.eq(target)]
        random=sub[sub.condition.str.startswith(random_prefix+'_seed')]
        if selected.empty:continue
        assert random.groupby(['match_id','scene_id','mapping_variant']).size().eq(4).all()
        for metric in ('margin_shift','rescued'):
            a=selected.groupby('match_id')[metric].mean().reset_index();b=random.groupby('match_id')[metric].mean().reset_index()
            values=paired(b,a,'match_id',metric)
            tests.append(dict(path=path_name,group=group,metric=metric,
                scope='all_attempted',**bootstrap(values,[path_name,group,metric])))
            family_tables.extend(dict(path=path_name,group=group,metric=metric,family_id=fid,difference=float(v)) for fid,v in zip(b.match_id,values))
        valid=set(random.groupby('match_id').norm_valid.all().loc[lambda x:x].index)
        if valid:
            a=selected[selected.match_id.isin(valid)].groupby('match_id').margin_shift.mean().reset_index()
            b=random[random.match_id.isin(valid)].groupby('match_id').margin_shift.mean().reset_index()
            tests.append(dict(path=path_name,group=group,metric='margin_shift',scope='all_four_controls_valid',
                **bootstrap(paired(b,a,'match_id','margin_shift'),[path_name,group,'valid'])))
    # Forward/reverse are compared directly using the same family IDs.
    subset=frame[frame.cohort.eq('right_error')&frame.side.eq('right')]
    mapping_effects=[]
    for variant,sub in subset.groupby('mapping_variant'):
        a=sub[sub.condition.eq(target)].groupby('match_id').margin_shift.mean().reset_index()
        b=sub[sub.condition.str.startswith(random_prefix+'_seed')].groupby('match_id').margin_shift.mean().reset_index()
        if a.empty:continue
        vals=paired(b,a,'match_id','margin_shift')
        mapping_effects.extend(dict(match_id=fid,variant=variant,effect=float(v)) for fid,v in zip(b.match_id,vals))
    pd.DataFrame(mapping_effects).to_csv(ANALYSIS_DIR/f'{path_name}_mapping_effects.csv',index=False)
    if mapping_effects:
        p=pd.DataFrame(mapping_effects).pivot(index='match_id',columns='variant',values='effect')
        result=bootstrap((p.forward-p.reverse).to_numpy(),[path_name,'mapping_interaction'])
        result['p_value']=min(1.,2*min(result['p_value'],1-result['p_value']+1/10001))
        tests.append(dict(path=path_name,group='mapping_difference',metric='margin_shift',scope='secondary',**result))
test_frame=pd.DataFrame(tests);test_frame['q_value']=np.nan
primary=(test_frame.group.eq('right_error')&test_frame.metric.eq('margin_shift')&test_frame.scope.eq('all_attempted'))
assert primary.sum()==3
test_frame.loc[primary,'q_value']=bh(test_frame.loc[primary,'p_value'])
test_frame.to_csv(ANALYSIS_DIR/'same_magnitude_tests.csv',index=False)
pd.DataFrame(family_tables).to_csv(ANALYSIS_DIR/'family_differences.csv',index=False)
late=frame[frame.condition.isin(['l55_repair','l55_bypass_after_l49_block'])&frame.cohort.eq('right_error')&frame.side.eq('right')]
a=late[late.condition.eq('l55_bypass_after_l49_block')].groupby('match_id').margin_shift.mean().reset_index()
b=late[late.condition.eq('l55_repair')].groupby('match_id').margin_shift.mean().reset_index()
save_json(ANALYSIS_DIR/'bypass_vs_late_only.json',bootstrap(paired(a,b,'match_id','margin_shift'),'bypass_vs_late_only'))
display(test_frame);display(frame.groupby(['cohort','side','condition']).rescued.mean().unstack('condition'))
save_json(ANALYSIS_DIR/'RUN_COMPLETE.json',dict(status='complete',historical_families=MATCHED_UNITS,
    records=len(frame),new_family_replication=False,all_norms_valid=bool(frame.norm_valid.all())))
(ANALYSIS_DIR/'REPORT_ZH.md').write_text('# 同幅度随机阻断对照\n\n原 32 家族及所有原条件保留；新增三条路径的四个随机方向。\n'
    '主指标是同幅度随机移除减去指定阻断的答案分数差；正值表示指定阻断造成更大损失。\n'
    '结合实际匹配失败明细、正确率和映射差异解释结果，不自动宣布完整物理计算路径。\n',encoding='utf-8')
print('Complete:',ANALYSIS_DIR)
