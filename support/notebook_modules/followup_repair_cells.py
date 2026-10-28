"""Shared development/confirmation cells. The builder selects marked sections."""

# %% COMMON: Load only historical training/development and response-direction inputs
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from visual_torque_toolkit.behavior import BehaviorRunConfig,PromptCondition,load_behavior_examples
from visual_torque_toolkit.behavior.data import BehaviorExample
from dataclasses import replace as dataclass_replace
import shutil
ACTIVATION_MANIFEST=PROJECT_ROOT/'results/behavior_comparison_v3/activation_pilot_manifest.jsonl'
DIRECTION_SUPPORT_PATH=PROJECT_ROOT/'results/semantic_label_depth_localization_v1/depth_support.jsonl'
DIRECTION_SCENE_DIR=PROJECT_ROOT/'results/inverse_query_weight_route_specificity_v1/rendered_scenes'
for path in (ACTIVATION_MANIFEST,DIRECTION_SUPPORT_PATH):assert path.is_file(),path
development_manifest=read_jsonl(ACTIVATION_MANIFEST)
assert len(development_manifest)==254 and {r['split'] for r in development_manifest}<={'probe_train','probe_dev'}
direction_support=read_jsonl(DIRECTION_SUPPORT_PATH)
assert len(direction_support)==32
RULE_TEXT=('Each square is one unit of weight. Distance is measured in tick intervals from the center pivot. '
           'Torque equals weight times distance. Total torque on a side is the sum over stacks on that side.')
LABELS=('A','B','C');SEMANTIC_TEXT={'left':'left side','right':'right side','balanced':'equal'}
def repair_prompt(task,semantics):
    question={'outcome':'Which way will the balance scale tip?','weight_more':'Which side has more total weight?',
              'weight_less':'Which side has less total weight?'}[task]
    return '\n\n'.join(['Answer this question using only the supplied balance-scale scene.',RULE_TEXT,'Question: '+question,
        'Options:\n'+'\n'.join(f'{l}. {SEMANTIC_TEXT[s]}' for l,s in zip(LABELS,semantics)),
        'Do not explain your reasoning. Reply with exactly one option letter and nothing else. Valid replies are: A, B, C.'])
def new_example(sid,path,features,semantics,variant,split,task='outcome'):
    val=features['net_torque'] if task=='outcome' else features['weight_score']*(1 if task=='weight_more' else -1)
    gold='left' if val>0 else 'right' if val<0 else 'balanced'
    return BehaviorExample(example_id=f'followup:{sid}:{task}:{variant}',question_id=f'followup_{sid}_{task}_{variant}',
        scene_id=sid,condition=PromptCondition.IMAGE_RULE,question_kind=task,variant=variant,split=split,
        problem_type='fixed_physics_repair',outcome=gold,net_torque=int(features['net_torque']),
        prompt=repair_prompt(task,semantics),image_path=Path(path),labels=LABELS,
        option_text=tuple(SEMANTIC_TEXT[s] for s in semantics),option_semantics=tuple(semantics),
        gold_label=LABELS[semantics.index(gold)],gold_semantic=gold)
def two_mappings(scene, family_id):
    # Rotation is family-fixed; mirror and render variants stay in the same family.
    offset=int(digest(['repair_mapping',family_id])[:8],16)%3
    base=['left','balanced','right'];base=base[offset:]+base[:offset]
    reverse=[{'left':'right','right':'left','balanced':'balanced'}[x] for x in base]
    return {'forward':tuple(base),'reverse':tuple(reverse)}
def code_key(wrong_label,correct_label):return str(wrong_label)+'_'+str(correct_label)

# %% DEV: Freeze fitting scenes, held-out development scenes and source identities
cfg=BehaviorRunConfig(dataset_dir=DATASET_DIR,output_dir=OUTPUT_ROOT/'unused_behavior',model_id=MODEL_ID,
    model_revision=MODEL_REVISION,conditions=(PromptCondition.IMAGE_RULE,),question_kinds=('outcome',),
    splits=('probe_train','probe_dev'),batch_size=1,generation_every=0,seed=SEED)
ids={str(r['scene_id']):r['split'] for r in development_manifest}
by_scene=defaultdict(list)
for ex in load_behavior_examples(cfg):
    if str(ex.scene_id) not in ids:continue
    semantics=tuple(map(str,ex.option_semantics))
    by_scene[str(ex.scene_id)].append(dataclass_replace(ex,prompt=repair_prompt('outcome',semantics)))
assert len(by_scene)==254 and all(len(v)==6 for v in by_scene.values())
for sid,examples in by_scene.items():assert {e.split for e in examples}=={ids[sid]}
TRAIN_IDS=sorted(s for s in ids if ids[s]=='probe_train')
DEV_IDS=sorted(s for s in ids if ids[s]=='probe_dev')
assert TRAIN_IDS and DEV_IDS and not set(TRAIN_IDS)&set(DEV_IDS)
runtime=current_runtime_identity()
train_identity=dict(protocol=PROTOCOL, runtime=runtime,manifest=file_hash(ACTIVATION_MANIFEST),
    support=file_hash(DIRECTION_SUPPORT_PATH),inputs={s:[example_stamp(e) for e in by_scene[s]] for s in sorted(by_scene)})
freeze(OUTPUT_ROOT/'development_identity.json',train_identity)
print('Direction fit scenes:',len(TRAIN_IDS),'| held-out development scenes:',len(DEV_IDS))

# %% DEV: Capture final-token states and fit supervised PLS4 prototype directions
train_states={};raw_baselines={}
for sid in tqdm(sorted(by_scene),desc='Capturing training and development scenes'):
    examples=sorted(by_scene[sid],key=lambda e:e.variant)
    def compute(examples=examples):
        states=defaultdict(list);scores=[]
        for ex in examples:
            record,h=final_forward(ex,CANDIDATE_LAYERS);scores.append(record)
            for layer in CANDIDATE_LAYERS:states[str(layer)].append(h[layer].ravel())
        return dict(scores=scores,mean_hidden={k:np.mean(v,axis=0).tolist() for k,v in states.items()})
    cached=cached_json(OUTPUT_ROOT/'capture',sid,train_identity,compute)
    train_states[sid]=cached['mean_hidden'];raw_baselines[sid]=cached['scores']
U_DIRECTIONS={};U_BASE_DOSES={}
y=np.array([int(by_scene[s][0].net_torque) for s in TRAIN_IDS],float)
assert (y>0).any() and (y<0).any()
for layer in CANDIDATE_LAYERS:
    X=np.array([train_states[s][str(layer)] for s in TRAIN_IDS],float)
    scaler=StandardScaler().fit(X);Z=scaler.transform(X)
    pls=PLSRegression(n_components=4,scale=False,max_iter=1000).fit(Z,y.reshape(-1,1))
    scores=pls.transform(Z);difference=scores[y>0].mean(0)-scores[y<0].mean(0)
    decoder=np.linalg.lstsq(scores-scores.mean(0),Z-Z.mean(0),rcond=1e-5)[0]
    direction=(difference@decoder)*scaler.scale_/2
    assert np.isfinite(direction).all() and np.linalg.norm(direction)>0
    U_DIRECTIONS[layer]=direction.astype('float32')
    # The new-model reference dose is defined using training states only.
    # It is NOT the legacy Qwen-27B mediator-matched alpha.
    U_BASE_DOSES[layer]=float(.01*np.median(np.linalg.norm(X,axis=1))/np.linalg.norm(direction))
print('Frozen base dose: 1% of median training hidden-state norm; development compares factors 0.5 and 1.')

# %% DEV: Build later answer directions from the disjoint historical response support
code_vectors=defaultdict(list);code_norms=[]
direction_hashes={}
for family in tqdm(direction_support,desc='Building later answer directions'):
    # Exact prior rotating-label seed and family identity; no confirmation outcomes.
    reference_key='options:'+family['family_id']
    offset=int(hashlib.sha256(f'20260901:{reference_key!r}'.encode()).hexdigest()[:8],16)%3
    semantics=['left','balanced','right'];semantics=semantics[offset:]+semantics[:offset]
    for task in ('weight_more','weight_less'):
        captured={};gold={}
        for role in ('base','aligned_source'):
            r=family['roles'][role];path=DIRECTION_SCENE_DIR/r['image_filename'];assert path.is_file(),path
            direction_hashes[str(path)]=file_hash(path)
            ex=new_example(r['scene_id'],path,r['features'],semantics,task,'historical_direction_fit',task)
            def compute(ex=ex):
                record,h=final_forward(ex,(DOWNSTREAM_LAYER,));return dict(record=record,state=h[DOWNSTREAM_LAYER].ravel().tolist())
            value=cached_json(OUTPUT_ROOT/'code_capture',ex.example_id,dict(runtime=runtime,input=example_stamp(ex),layer=DOWNSTREAM_LAYER),compute)
            captured[role]=np.asarray(value['state'],float);gold[role]=ex.gold_label
        delta=captured['aligned_source']-captured['base'];norm=float(np.linalg.norm(delta))
        if norm==0:continue
        code_vectors[code_key(gold['base'],gold['aligned_source'])].append(delta/norm);code_norms.append(norm)
CODE_DIRECTIONS={k:np.mean(v,axis=0) for k,v in code_vectors.items()}
assert len(CODE_DIRECTIONS)==6,'Historical support must span all six label transitions'
code_size=float(np.median(code_norms))
for key,value in CODE_DIRECTIONS.items():
    assert np.linalg.norm(value)>0
    CODE_DIRECTIONS[key]=(value/np.linalg.norm(value)*code_size).astype('float32')
freeze(OUTPUT_ROOT/'code_inputs.json',direction_hashes)

# %% DEV: Evaluate the bounded layer-dose grid against four matched random directions
dev_rows=[]
for sid in tqdm(DEV_IDS,desc='Development layer/dose comparison'):
    for ex in sorted(by_scene[sid],key=lambda e:e.variant):
        def evaluate(ex=ex):
            base,_=final_forward(ex);rows=[];sign=1. if ex.net_torque>0 else -1. if ex.net_torque<0 else 0.
            for layer in CANDIDATE_LAYERS:
                for factor in (.5,1.):
                    payload=sign*U_DIRECTIONS[layer]*U_BASE_DOSES[layer]*factor
                    for condition,seed in [('targeted',-1)]+[('random',s) for s in RANDOM_SEEDS]:
                        op=dict(action='add',payload=payload) if condition=='targeted' else dict(action='matched_add',reference=payload,key=[SEED,sid,ex.variant,layer,factor,seed])
                        record,_=final_forward(ex,operations={layer:op})
                        rows.append(dict(scene_id=sid,variant=ex.variant,layer=layer,factor=factor,condition=condition,random_seed=seed,
                            baseline_correct=base['correct'],accuracy_change=float(record['correct'])-float(base['correct']),
                            margin_change=correct_margin(record)-correct_margin(base),norm_valid=all(a.get('matched',True) for a in record['write_audits'].values()),record=record))
            return rows
        dev_rows.extend(cached_json(OUTPUT_ROOT/'development_trials',ex.example_id,
            dict(training=digest(train_identity),directions=digest(U_DIRECTIONS),doses=U_BASE_DOSES,input=example_stamp(ex)),evaluate))
dev=pd.DataFrame([{k:v for k,v in r.items() if k!='record'} for r in dev_rows]);dev.to_csv(ANALYSIS_DIR/'development_trials.csv',index=False)
save_json(ANALYSIS_DIR/'development_records.json',dev_rows)
rows=[]
for (layer,factor),sub in dev.groupby(['layer','factor']):
    target=sub[sub.condition.eq('targeted')].groupby('scene_id')[['accuracy_change','margin_change']].mean().reset_index()
    random=sub[sub.condition.eq('random')].groupby('scene_id')[['accuracy_change','margin_change']].mean().reset_index()
    direct=bootstrap(target.accuracy_change.to_numpy(float),[layer,factor,'accuracy'])
    specificity=bootstrap(paired(target,random,'scene_id','accuracy_change'),[layer,factor,'specificity'])
    valid=bool(sub.norm_valid.all())
    rows.append(dict(layer=int(layer),factor=float(factor),accuracy_change=direct['mean'],accuracy_lower=direct['lower'],
        specificity=specificity['mean'],specificity_lower=specificity['lower'],all_norms_valid=valid,
        eligible=bool(valid and direct['lower']>0 and specificity['lower']>0)))
selection_table=pd.DataFrame(rows);selection_table.to_csv(ANALYSIS_DIR/'development_selection.csv',index=False);display(selection_table)
eligible=selection_table[selection_table.eligible].sort_values(['specificity','layer','factor'],ascending=[False,True,True])
selected=eligible.iloc[0] if len(eligible) else None
selection=dict(model=MODEL_KEY,model_id=MODEL_ID,revision=MODEL_REVISION,downstream_layer=DOWNSTREAM_LAYER,
    eligible=selected is not None,method='pls4_prototype_train_only_six_map_mean',
    dose_rule='training median residual norm 1%, times frozen factor 0.5 or 1',
    training_identity=digest(train_identity),runtime=runtime,implementation=IMPLEMENTATION_SHA256,
    development_scenes=DEV_IDS,fit_scenes=TRAIN_IDS,direction_support_sha256=file_hash(DIRECTION_SUPPORT_PATH),
    conclusion='selected_for_new_confirmation' if selected is not None else 'no_upstream_repair_established_in_fixed_grid')
if selected is not None:
    layer=int(selected.layer);dose=U_BASE_DOSES[layer]*float(selected.factor)
    arrays={'upstream':U_DIRECTIONS[layer],**{'code_'+k:v for k,v in CODE_DIRECTIONS.items()}}
    vector_path=OUTPUT_ROOT/'frozen_directions.npz'
    if vector_path.is_file():
        with np.load(vector_path,allow_pickle=False) as old:
            assert set(old.files)==set(arrays) and all(np.array_equal(old[k],v) for k,v in arrays.items())
    else:np.savez_compressed(vector_path,**arrays)
    selection.update(upstream_layer=layer,dose=dose,directions_sha256=file_hash(OUTPUT_ROOT/'frozen_directions.npz'))
freeze(OUTPUT_ROOT/'selected_route.json',selection)
save_json(ANALYSIS_DIR/'RUN_COMPLETE.json',dict(status='development_complete',selection=selection))
(ANALYSIS_DIR/'REPORT_ZH.md').write_text('# 修正位置开发\n\n'+('已冻结候选，接下来只在新家族上确认。' if selected is not None else '固定候选范围内未建立有效上游修正；不开展依赖解释。')+'\n',encoding='utf-8')
print('Development finished:',OUTPUT_ROOT,'| eligible:',selection['eligible'])

# %% CONFIRM: Generate a physically new, model-independent confirmation cohort
from visual_torque_toolkit.render import RenderStyle,render_scene
from visual_torque_toolkit.world import Scene,Side,Stack
excluded,exclusion_identity=exclusion_snapshot(PROJECT_ROOT,FOLLOWUP_ROOT,DATASET_DIR)
support_dir=FOLLOWUP_ROOT/'support'/('repair_'+RUN_ID);support_dir.mkdir(parents=True,exist_ok=True)
support_file=support_dir/'support.json'
support_config=dict(n=CONFIRM_FAMILIES,seed=CONFIRM_SEED,generator='four_strata_mirror_render_1',exclusions=digest(exclusion_identity))
freeze(support_dir/'protocol.json',support_config)
for path in (FOLLOWUP_ROOT/'support').glob('fixed_content_*/support.json'):excluded.update(geometries_in(read_json(path)))
if support_file.is_file():
    payload=read_json(support_file);assert payload['config']==support_config;families=payload['families']
else:
    families=repair_families(CONFIRM_FAMILIES,CONFIRM_SEED,excluded);save_json(support_file,dict(config=support_config,families=families))
assert len(families)==CONFIRM_FAMILIES
assert not geometries_in(families)&excluded
images=support_dir/'images';images.mkdir(exist_ok=True)
for family in tqdm(families,desc='Rendering new repair families'):
    for row in family['scenes']:
        path=images/row['image_filename']
        if not path.is_file():
            style=RenderStyle(ticks_per_side=6) if row['style']=='standard' else RenderStyle(ticks_per_side=6,background=(239,245,255),block_fill=(37,99,235))
            scene=Scene(row['scene_id'],tuple(Stack(str(i),Side(s['side']),int(s['distance']),int(s['units'])) for i,s in enumerate(row['stacks'])))
            render_scene(scene,path,style)
runtime=current_runtime_identity()
for key in ('torch','transformers','dtype','batch_size','checkpoint','layer_path','toolkit'):
    assert runtime[key]==selection['runtime'][key], f'Development/reference runtime differs at {key}; keep paired protocols consistent'
confirm_identity=dict(protocol=PROTOCOL,model=MODEL_ID,revision=MODEL_REVISION,runtime=runtime,selection=selection,
    selection_sha256=file_hash(SELECTION_PATH),directions_sha256=file_hash(DIRECTIONS_PATH),support=file_hash(support_file),
    images={r['image_filename']:file_hash(images/r['image_filename']) for f in families for r in f['scenes']})
freeze(OUTPUT_ROOT/'confirmation_identity.json',confirm_identity)

# %% CONFIRM: Run upstream repair, local blocking, random controls and direct late repair
records=[]
for family in tqdm(families,desc='Confirming fixed repair and blocking'):
    fid=family['family_id']
    def evaluate(family=family):
        rows=[]
        for scene in family['scenes']:
            for variant,semantics in two_mappings(scene,fid).items():
                ex=new_example(scene['scene_id'],images/scene['image_filename'],scene['features'],semantics,variant,'new_repair_confirmation')
                clean,h=final_forward(ex,(DOWNSTREAM_LAYER,))
                sign=1. if ex.net_torque>0 else -1.
                opposite='right' if ex.gold_semantic=='left' else 'left'
                wrong_label=LABELS[semantics.index(opposite)]
                code=CODE_DIRECTIONS[code_key(wrong_label,ex.gold_label)]
                upstream=sign*UPSTREAM_DIRECTION*UPSTREAM_DOSE
                add={UPSTREAM_LAYER:dict(action='add',payload=upstream)}
                block=dict(action='clamp',untreated=h[DOWNSTREAM_LAYER],axis=code)
                conditions=[('zero',-1,{}),('upstream',-1,add),('upstream_block',-1,{**add,DOWNSTREAM_LAYER:block}),
                            ('late_only',-1,{DOWNSTREAM_LAYER:dict(action='add',payload=code)}),('block_only',-1,{DOWNSTREAM_LAYER:block})]
                for seed in RANDOM_SEEDS:
                    key=[CONFIRM_SEED,fid,scene['scene_id'],variant,seed]
                    conditions.extend([
                        ('random_upstream',seed,{UPSTREAM_LAYER:dict(action='matched_add',reference=upstream,key=key+['upstream'])}),
                        ('upstream_random_coordinate',seed,{**add,DOWNSTREAM_LAYER:dict(block,action='random_coordinate',key=key+['late'])}),
                        ('upstream_equal_remove',seed,{**add,DOWNSTREAM_LAYER:dict(block,action='equal_remove',key=key+['late'])}),
                        ('random_block_only',seed,{DOWNSTREAM_LAYER:dict(block,action='random_coordinate',key=key+['late'])})])
                for condition,seed,operations in conditions:
                    out,_=final_forward(ex,operations=operations)
                    if condition=='zero':
                        gap=np.max(np.abs(centered_scores(out)-centered_scores(clean)))
                        assert gap<=.15, 'Current zero replay failed'
                    rows.append(dict(family_id=fid,stratum=family['stratum'],scene_id=scene['scene_id'],variant=variant,
                        condition=condition,random_seed=seed,correct=out['correct'],baseline_correct=clean['correct'],
                        accuracy_change=float(out['correct'])-float(clean['correct']),margin_change=correct_margin(out)-correct_margin(clean),
                        answer_changed=out['predicted_semantic']!=clean['predicted_semantic'],
                        valid=all(a.get('matched',True) for a in out['write_audits'].values()),record=out))
        return rows
    records.extend(cached_json(OUTPUT_ROOT/'family_shards',fid,confirm_identity,evaluate))
save_json(ANALYSIS_DIR/'intervention_records.json',records)
frame=pd.DataFrame([{k:v for k,v in r.items() if k!='record'} for r in records]);frame.to_csv(ANALYSIS_DIR/'all_conditions.csv',index=False)
expected=4*2*(5+4*len(RANDOM_SEEDS));assert frame.groupby('family_id').size().eq(expected).all()
family_means=frame.groupby(['family_id','condition'])[['accuracy_change','margin_change','correct','answer_changed']].mean().reset_index()
family_means.to_csv(ANALYSIS_DIR/'family_metrics.csv',index=False)
contrasts=[('upstream_vs_zero','upstream','zero'),('upstream_vs_random','upstream','random_upstream'),
           ('specific_block_loss','upstream_equal_remove','upstream_block')]
tests=[]
for contrast,a,b in contrasts:
    for metric in ('accuracy_change','margin_change'):
        left=family_means[family_means.condition.eq(a)];right=family_means[family_means.condition.eq(b)]
        tests.append(dict(contrast=contrast,metric=metric,**bootstrap(paired(left,right,'family_id',metric),[MODEL_KEY,contrast,metric])))
tests=pd.DataFrame(tests);tests['q_value']=bh(tests.p_value);tests.to_csv(ANALYSIS_DIR/'primary_tests.csv',index=False)
valid_families=set(frame.groupby('family_id').valid.all().loc[lambda x:x].index)
valid_tests=[]
for contrast,a,b in contrasts:
    left=family_means[family_means.condition.eq(a)&family_means.family_id.isin(valid_families)]
    right=family_means[family_means.condition.eq(b)&family_means.family_id.isin(valid_families)]
    if not len(left):continue
    for metric in ('accuracy_change','margin_change'):
        valid_tests.append(dict(contrast=contrast,metric=metric,**bootstrap(paired(left,right,'family_id',metric),[MODEL_KEY,'valid',contrast,metric])))
pd.DataFrame(valid_tests).to_csv(ANALYSIS_DIR/'all_controls_valid_sensitivity.csv',index=False)
frame[~frame.valid].to_csv(ANALYSIS_DIR/'write_failures.csv',index=False)
groups=[]
for (condition,baseline),sub in frame.groupby(['condition','baseline_correct']):
    counts=sub.groupby('family_id').correct.mean()
    groups.append(dict(condition=condition,originally_correct=bool(baseline),families=len(counts),
        scene_mapping_units=sub[['scene_id','variant']].drop_duplicates().shape[0],accuracy=float(counts.mean())))
pd.DataFrame(groups).to_csv(ANALYSIS_DIR/'original_errors_and_correct.csv',index=False)
def passes(name):
    r=tests[(tests.contrast==name)&tests.metric.eq('accuracy_change')].iloc[0]
    return bool(r.lower>0 and r.q_value<.05)
upstream_controls_valid=bool(frame[frame.condition.eq('random_upstream')].valid.all())
repair_confirmed=passes('upstream_vs_zero') and passes('upstream_vs_random') and upstream_controls_valid
dependence=repair_confirmed and passes('specific_block_loss') and bool(frame.valid.all())
result=dict(model=MODEL_KEY,support=file_hash(support_file),families=len(families),all_controls_valid=bool(frame.valid.all()),
    valid_families=len(valid_families),upstream_repair_confirmed=repair_confirmed,selective_dependence_confirmed=dependence,
    status='confirmation_complete',scope='two_site_supervised_intervention_dependency',
    downstream_role='candidate_answer_site; semantic class requires the separate remapping evidence')
save_json(ANALYSIS_DIR/'RUN_COMPLETE.json',result)
(ANALYSIS_DIR/'REPORT_ZH.md').write_text('# 新家族修正与阻断\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n\n先解释上游修正是否成立，再解释指定阻断相对同幅度随机移除的损失。\n',encoding='utf-8')
display(tests);display(family_means.groupby('condition')[['correct','accuracy_change','margin_change']].mean());print(result)
