"""New support and cross-model analysis cells for fixed-site confirmation."""

# %% Generate and freeze new physical support before loading any model
from visual_torque_toolkit.render import RenderStyle, render_scene
from visual_torque_toolkit.world import Scene, Side, Stack
import shutil
excluded,exclusion_identity=exclusion_snapshot(PROJECT_ROOT,FOLLOWUP_ROOT,DATASET_DIR)
support_dir=FOLLOWUP_ROOT/'support'/('fixed_content_'+RUN_ID)
support_dir.mkdir(parents=True,exist_ok=True)
support_file=support_dir/'support.json'
support_config=dict(n=EXPECTED_FAMILIES,seed=SEED,generator='inverse_equal_net_mirror_disjoint_1',
                    exclusions=digest(exclusion_identity))
freeze(support_dir/'protocol.json',support_config)
if support_file.is_file():
    support_payload=read_json(support_file)
    assert support_payload['config']==support_config
    depth_support=support_payload['families']
else:
    # Explicitly avoid any repair-confirmation support already created in this bundle.
    for path in (FOLLOWUP_ROOT/'support').glob('repair_*/support.json'):
        excluded.update(geometries_in(read_json(path)))
    depth_support=inverse_families(EXPECTED_FAMILIES,SEED,excluded)
    save_json(support_file,dict(config=support_config,families=depth_support))
assert len(depth_support)==EXPECTED_FAMILIES
all_signatures=[geometry_signature(role['stacks']) for r in depth_support for role in r['roles'].values()]
assert len(set(all_signatures))==2*EXPECTED_FAMILIES
assert not set(all_signatures)&excluded
PRIOR_SCENE_DIR=support_dir/'images';PRIOR_SCENE_DIR.mkdir(exist_ok=True)
image_filenames=[]
for row in tqdm(depth_support,desc='Rendering fixed confirmation families'):
    a,b=row['roles']['base'],row['roles']['aligned_source']
    assert a['features']['net_torque']==b['features']['net_torque']
    assert [(x['side'],x['distance']) for x in a['stacks']]==[(x['side'],x['distance']) for x in b['stacks']]
    assert a['features']['weight_score']*b['features']['weight_score']<0
    for role in row['roles'].values():
        path=PRIOR_SCENE_DIR/role['image_filename'];image_filenames.append(role['image_filename'])
        if not path.is_file():
            scene=Scene(role['scene_id'],tuple(Stack(str(i),Side(s['side']),int(s['distance']),int(s['units'])) for i,s in enumerate(role['stacks'])))
            render_scene(scene,path,RenderStyle(ticks_per_side=6))
RUNTIME_SCENE_DIR=Path('/content/visualtorque_fixed_confirmation')/RUN_ID
RUNTIME_SCENE_DIR.mkdir(parents=True,exist_ok=True)
for name in image_filenames:
    src,dst=PRIOR_SCENE_DIR/name,RUNTIME_SCENE_DIR/name
    if not dst.is_file() or file_hash(dst)!=file_hash(src):shutil.copy2(src,dst)
support_identity=dict(schema_version='new-fixed-content-support-1.0',support_sha256=file_hash(support_file),
    family_ids=[r['family_id'] for r in depth_support],geometry_disjoint=True,mirror_canonical=True,
    prior_manifest_sha256=digest(exclusion_identity),analysis_status='new_fixed_site_confirmation')
remap_manifest_sha256=file_hash(support_file)
print('New families:',len(depth_support),'| selected without model outputs')

# %% Export per-model results without prematurely changing the 37-test family
family_frame['model_key']=MODEL_KEY
family_frame.to_csv(ANALYSIS_DIR/'family_depth_metrics.csv',index=False)
native=[]
for path in sorted(SHARD_DIR.glob('*.json')):
    obj=read_json(path)
    for r in obj['patch_records']:
        native.append({k:v for k,v in r.items() if k not in ('candidate_logprobs','candidate_probabilities','diagnostics')})
pd.DataFrame(native).to_csv(ANALYSIS_DIR/'native_displacement_audit.csv',index=False)
random_writes=[r for r in native if str(r['condition']).endswith('_random')]
assert random_writes and all('native_norm_matched' in r for r in random_writes)
all_controls_valid=all(r['native_norm_matched'] for r in random_writes)
pd.DataFrame([r for r in random_writes if not r['native_norm_matched']]).to_csv(ANALYSIS_DIR/'norm_match_failures.csv',index=False)
save_json(ANALYSIS_DIR/'completion.json',dict(model_key=MODEL_KEY,protocol=digest(canonical_protocol_identity),
    support=support_identity,implementation=IMPLEMENTATION_SHA256,smoke_errors=smoke_errors,
    family_table_sha256=file_hash(ANALYSIS_DIR/'family_depth_metrics.csv'),
    all_native_controls_valid=all_controls_valid,status='model_capture_complete_global_tests_pending'))
display(family_frame.groupby('layer_index')[['identity_carrier','semantic_effect','label_effect','position_effect']].mean())

# %% Global 37-test confirmation and fixed-endpoint pairwise comparisons
PRIMARY_METRICS=('identity_carrier','semantic_effect','label_effect','position_effect',
                 'semantic_dominance','label_dominance','position_dominance')
all_frames={};coverage=[];valid_by_model={}
for key,spec in MODEL_SPECS.items():
    folder=SHARED_ROOT/'models'/spec['model_slug']/'analysis'
    complete=folder/'completion.json';table=folder/'family_depth_metrics.csv'
    if not complete.is_file() or not table.is_file():
        coverage.append(dict(model=key,status='not_run'));continue
    marker=read_json(complete)
    assert marker['protocol']==digest(canonical_protocol_identity)
    assert marker['support']==support_identity and marker['implementation']==IMPLEMENTATION_SHA256
    assert marker['family_table_sha256']==file_hash(table)
    data=pd.read_csv(table)
    assert set(data.layer_index)==set(spec['scan_layers']) and not data.duplicated(['family_id','layer_index']).any()
    for layer,sub in data.groupby('layer_index'):assert set(sub.family_id)==set(support_identity['family_ids'])
    all_frames[key]=data;valid_by_model[key]=bool(marker['all_native_controls_valid'])
    coverage.append(dict(model=key,status='complete',all_native_controls_valid=valid_by_model[key]))
display(pd.DataFrame(coverage));pd.DataFrame(coverage).to_csv(CROSS_MODEL_DIR/'coverage.csv',index=False)
tests=[]
for key,data in all_frames.items():
    for layer,sub in data.groupby('layer_index'):
        for metric in PRIMARY_METRICS:
            tests.append(dict(model=key,layer=int(layer),metric=metric,**bootstrap(sub[metric].to_numpy(float),[key,layer,metric])))
    if len(MODEL_SPECS[key]['scan_layers'])==2:
        earlier,later=MODEL_SPECS[key]['scan_layers']
        pivot=data.assign(relative=data.label_effect-data.semantic_effect).pivot(index='family_id',columns='layer_index',values='relative')
        tests.append(dict(model=key,layer=-1,metric='later_minus_earlier_label_relative_to_semantic',
                          **bootstrap((pivot[later]-pivot[earlier]).to_numpy(float),[key,'two_site_difference'])))
test_frame=pd.DataFrame(tests);test_frame['q_value']=np.nan
complete=len(all_frames)==3
decisions=[]
if complete:
    assert len(test_frame)==37
    test_frame['q_value']=bh(test_frame.p_value)
    for key,spec in MODEL_SPECS.items():
        for layer in spec['scan_layers']:
            sub=test_frame[(test_frame.model==key)&(test_frame.layer==layer)].set_index('metric')
            passes=lambda metric:bool(sub.loc[metric,'lower']>.1 and sub.loc[metric,'q_value']<.05)
            classes=[c for c in ('semantic','label','position') if passes('identity_carrier') and passes(c+'_effect') and passes(c+'_dominance')]
            classification=('technical_control_mismatch' if not valid_by_model[key] else classes[0] if len(classes)==1 else 'unresolved')
            decisions.append(dict(model=key,layer=layer,classification=classification,
                scope='fixed_site_only_not_a_continuous_transition'))
    pairwise=[]
    for left,right in combinations(MODEL_SPECS,2):
        a=all_frames[left][all_frames[left].layer_index.eq(MODEL_SPECS[left]['endpoint_layer'])]
        b=all_frames[right][all_frames[right].layer_index.eq(MODEL_SPECS[right]['endpoint_layer'])]
        result=bootstrap(paired(a,b,'family_id','label_effect'),[left,right,'endpoint'])
        result['p_value']=min(1.,2*min(result['p_value'],1-result['p_value']+1/10001))
        pairwise.append(dict(left_model=left,right_model=right,**result))
    p=pd.DataFrame(pairwise);p['q_value']=bh(p.p_value);p.to_csv(CROSS_MODEL_DIR/'endpoint_pairwise_secondary.csv',index=False)
test_frame.to_csv(CROSS_MODEL_DIR/'fixed_37_tests.csv',index=False)
pd.DataFrame(decisions).to_csv(CROSS_MODEL_DIR/'fixed_site_classification.csv',index=False)
display(test_frame.round(4));display(pd.DataFrame(decisions))
save_json(CROSS_MODEL_DIR/'status.json',dict(complete=complete,expected_tests=37,available_tests=len(tests),
    classification=decisions,interpretation='Two fixed sites can show a content difference; this does not establish a continuous conversion chain.'))
lines=['# 固定位置的新家族确认','',f'已完成模型：{", ".join(all_frames)}。',
       '全部三模型完成后才统一校正预定的 37 个主指标；未完成时不提前给出阶段判定。',
       '单项效应、完整类别判据、两个位置的配对差异分别展示。',
       '样本是新家族；层位来自此前探索。两个位置不能代表完整连续转换轨迹。']
(CROSS_MODEL_DIR/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Results:',CROSS_MODEL_DIR)
