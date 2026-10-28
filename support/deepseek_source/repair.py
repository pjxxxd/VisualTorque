"""Section 4.5: train-only directions, bounded development selection, 64-family confirmation."""
from dataclasses import replace
from types import SimpleNamespace
from core import fit_examples, load_model, load_kernels, verify_scoring_path
from readout import capture_fit

U_LAYERS=(10,13,16,19)
A_LAYER=20
B_LAYER=23


def conditions(layers,sign,upstream,semantic,code,untreated,key,seeds=(0,1,2,3)):
    lu,la,lb=layers
    assert lu<la<lb and {la,lb}<=set(untreated)
    u={lu:dict(action='add',payload=sign*upstream)}
    a={la:dict(action='add',payload=sign*semantic)}
    b={lb:dict(action='add',payload=code)}
    blocks={la:dict(action='clamp',untreated=untreated[la],axis=sign*semantic),
            lb:dict(action='clamp',untreated=untreated[lb],axis=code)}
    rows=[('zero',-1,{}),('u',-1,u),('a',-1,a),('b',-1,b),('bypass',-1,{**u,la:blocks[la],**b})]
    for name,layer,payload in [('u',lu,sign*upstream),('a',la,sign*semantic)]:
        for s in seeds:rows.append((name+'_random',s,{layer:dict(action='matched_add',reference=payload,key=key+[name,s])}))
    for name,layer in [('a',la),('b',lb)]:
        rows.extend([(f'self_{name}',-1,{layer:blocks[layer]}),
            (f'self_random_{name}',0,{layer:dict(blocks[layer],action='random_coordinate',key=key+['self',layer])})])
    for name,start,layer in [('ua',u,la),('ub',u,lb),('ab',a,lb)]:
        rows.append((name+'_block',-1,{**start,layer:blocks[layer]}))
        rows.append((name+'_coordinate',0,{**start,layer:dict(blocks[layer],action='random_coordinate',key=key+[name,'coordinate'])}))
        for s in seeds:rows.append((name+'_equal',s,{**start,layer:dict(blocks[layer],action='equal_remove',key=key+[name,'equal',s])}))
    assert len(rows)==35 and len({(a,b) for a,b,_ in rows})==35
    return rows


def train_pls(n, means, groups):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cross_decomposition import PLSRegression
    np=n['np']; ids=n['TRAIN_IDS']; y=np.array([groups[s][0].net_torque for s in ids],float)
    directions={}; doses={}
    assert (y>0).any() and (y<0).any()
    for l in U_LAYERS:
        x=np.array([means[s][str(l)] for s in ids],float)
        scaler=StandardScaler().fit(x); z=scaler.transform(x)
        pls=PLSRegression(n_components=4,scale=False,max_iter=1000).fit(z,y.reshape(-1,1))
        scores=pls.transform(z); diff=scores[y>0].mean(0)-scores[y<0].mean(0)
        decoder=np.linalg.lstsq(scores-scores.mean(0),z-z.mean(0),rcond=1e-5)[0]
        d=(diff@decoder)*scaler.scale_/2
        assert np.isfinite(d).all() and np.linalg.norm(d)>0
        directions[l]=d.astype('float32')
        doses[l]=float(.01*np.median(np.linalg.norm(x,axis=1))/np.linalg.norm(d))
    return directions,doses


def answer_directions(n,root):
    np=n['np']; semantic=[]; semantic_norms=[]; codes=n['defaultdict'](list); code_norms=[]
    support_path=n['ROOT']/'prediction/support.json'
    support=n['read_json'](support_path)
    families=[f for f in support['families'] if f['split']=='calibration' and f['module']=='content']
    assert len(families)==32
    n['IMAGE_DIR']=n['ROOT']/'prediction/images'
    image_hashes=n['read_json'](n['ROOT']/'prediction/image_manifest.json')
    for f in families:
        for r in f['roles'].values():
            name=r['image_filename']
            assert n['file_hash'](n['IMAGE_DIR']/name)==image_hashes[name], 'Calibration image changed.'
    for f in n['tqdm'](families,desc='Answer directions: 32 calibration families only'):
        mapping=n['mapping_reference'](f['family_id'])
        for task in ('weight_more','weight_less'):
            states={}; gold={}
            for role in ('base','aligned_source'):
                ex=n['make_predictive_example'](f,role,task,mapping)
                def compute(ex=ex):
                    rec,h=n['final_forward'](ex,(A_LAYER,B_LAYER))
                    return dict(record=rec,states={str(k):v.ravel().tolist() for k,v in h.items()})
                v=n['cached_json'](root/'answer_capture',ex.example_id,
                    dict(runtime=n['RUNTIME_IDENTITY'],input=n['example_stamp'](ex),layers=[A_LAYER,B_LAYER]),compute)
                states[role]=v['states'];gold[role]=(ex.gold_semantic,ex.gold_label)
            for layer in (A_LAYER,B_LAYER):
                d=np.array(states['aligned_source'][str(layer)])-np.array(states['base'][str(layer)])
                size=float(np.linalg.norm(d)); assert size>0,'Degenerate answer direction; inspect rather than skip silently.'
                if layer==A_LAYER:
                    assert {gold['base'][0],gold['aligned_source'][0]}=={'left','right'}
                    sign=1 if gold['aligned_source'][0]=='left' else -1
                    semantic.append(sign*d/size);semantic_norms.append(size)
                else:
                    key=gold['base'][1]+'_'+gold['aligned_source'][1]
                    codes[key].append(d/size); code_norms.append(size)
    assert len(codes)==6
    a=np.mean(semantic,axis=0); a=a/np.linalg.norm(a)*np.median(semantic_norms)
    result={'semantic':a.astype('float32')}
    for key,values in codes.items():
        v=np.mean(values,axis=0); assert np.linalg.norm(v)>0
        result['code_'+key]=(v/np.linalg.norm(v)*np.median(code_norms)).astype('float32')
    return result


def development(n):
    fit_examples(n)
    assert (n['ROOT']/'prediction/support.json').is_file(),'Run notebook 01 calibration first (provides 32 direction-fit families).'
    load_model(n);load_kernels(n,['repair_examples.py'])
    root=n['ROOT']/'repair'; root.mkdir(exist_ok=True)
    verify_scoring_path(n,n['FIT_EXAMPLES'][n['TRAIN_IDS'][0]][0],root/'technical_check.json')
    groups={s:[replace(e,prompt=n['repair_prompt']('outcome',e.option_semantics)) for e in es]
            for s,es in n['FIT_EXAMPLES'].items()}
    means,_,identity=capture_fit(n,root,groups,U_LAYERS,'repair_training')
    directions,doses=train_pls(n,means,groups)
    later=answer_directions(n,root)
    all_vectors={**{'u_'+str(k):v for k,v in directions.items()},**later}
    vector_sha=n['frozen_arrays'](root/'development_vectors.npz',all_vectors)
    trials=[]
    for sid in n['tqdm'](n['DEV_IDS'],desc='Repair: fixed layer/dose development grid'):
        for ex in groups[sid]:
            def compute(ex=ex):
                base,_=n['final_forward'](ex); rows=[]
                sign=1. if ex.net_torque>0 else -1. if ex.net_torque<0 else 0.
                for layer in U_LAYERS:
                    for factor in (.5,1.):
                        payload=sign*directions[layer]*doses[layer]*factor
                        for name,seed in [('targeted',-1)]+[('random',s) for s in range(4)]:
                            op=dict(action='add',payload=payload) if name=='targeted' else dict(action='matched_add',
                                reference=payload,key=[n['SEED'],ex.example_id,layer,factor,seed])
                            rec,_=n['final_forward'](ex,operations={layer:op})
                            rows.append(dict(scene_id=str(ex.scene_id),variant=ex.variant,layer=layer,factor=factor,
                                condition=name,random_seed=seed,accuracy_change=float(rec['correct'])-float(base['correct']),
                                margin_change=n['correct_margin'](rec)-n['correct_margin'](base),
                                valid=all(a.get('matched',True) for a in rec['write_audits'].values()),record=rec))
                return rows
            trials.extend(n['cached_json'](root/'development_trials',ex.example_id,
                dict(runtime=n['RUNTIME_IDENTITY'],fit=n['digest'](identity),vectors=vector_sha,doses=doses,input=n['example_stamp'](ex)),compute))
    n['save_json'](root/'development_records.json',trials)
    frame=n['pd'].DataFrame([{k:v for k,v in r.items() if k!='record'} for r in trials]); results=[]
    for (layer,factor),g in frame.groupby(['layer','factor']):
        fm=g.groupby(['scene_id','condition']).accuracy_change.mean().unstack()
        direct=n['paired_summary'](fm.targeted.values,['dev',int(layer),float(factor),'gain'])
        specific=n['paired_summary']((fm.targeted-fm.random).values,['dev',int(layer),float(factor),'specificity'])
        valid=bool(g.valid.all())
        results.append(dict(layer=int(layer),factor=float(factor),accuracy_gain=direct['mean'],accuracy_lower=direct['lower'],
            specificity=specific['mean'],specificity_lower=specific['lower'],all_norms_valid=valid,
            eligible=bool(valid and direct['lower']>0 and specific['lower']>0)))
    table=n['pd'].DataFrame(results);table.to_csv(root/'development_selection.csv',index=False)
    # Even when no candidate passes, freeze the bounded-grid best and test it once.
    best=max(results,key=lambda r:(r['eligible'],r['specificity'],-r['layer'],-r['factor']))
    vectors=dict(upstream=directions[best['layer']],**later)
    sha=n['frozen_arrays'](root/'frozen_directions.npz',vectors)
    selection=dict(model=n['MODEL_ID'],revision=n['MODEL_REVISION'],runtime=n['RUNTIME_IDENTITY'],
        selected=best,upstream_layer=best['layer'],semantic_layer=A_LAYER,symbol_layer=B_LAYER,
        dose=doses[best['layer']]*best['factor'],semantic_dose=1.,vectors_sha256=sha,
        training_ids=n['TRAIN_IDS'],development_ids=n['DEV_IDS'],direction_support_sha256=n['file_hash'](n['ROOT']/'prediction/support.json'),
        rule='development eligibility then maximum specificity; ties lower layer then dose; no validation reselection',
        direction_source='32 content calibration families; no validation families')
    n['freeze'](root/'selected_route.json',selection)
    n['save_json'](root/'development_complete.json',dict(status='development_complete',eligible=best['eligible']))
    print('Frozen route:',best,'; initial repair and all three dependence claims still require confirmation.')


def audit_rows(n,rows,layers):
    np=n['np'];lu,la,lb=layers; by={(r['condition'],r['random_seed']):r for r in rows}
    refs={'u_random':('u',lu),'a_random':('a',la),'ua_equal':('ua_block',la),
          'ub_equal':('ub_block',lb),'ab_equal':('ab_block',lb)}
    for r in rows:
        if r['condition'] in refs:
            target,layer=refs[r['condition']]
            intended=by[(target,-1)]['record']['write_audits'][str(layer)]['actual_norm']
            actual=r['record']['write_audits'][str(layer)]['actual_norm']
            error=abs(actual-intended)/max(intended,1e-12)
            r.update(paired_norm_error=error,valid=bool(r['valid'] and error<=.02))
        if r['condition'].startswith('self_'):
            clean=by[('zero',-1)]['record']
            error=float(np.max(np.abs(n['center'](clean['candidate_logprobs'])-n['center'](r['record']['candidate_logprobs']))))
            r.update(self_score_error=error,valid=bool(r['valid'] and error<=1e-6 and
                clean['predicted_semantic']==r['record']['predicted_semantic']))
    return rows


CONTRASTS=[('u_vs_zero','u','zero'),('u_vs_random','u','u_random'),
    ('a_vs_zero','a','zero'),('a_vs_random','a','a_random'),
    ('ua_specific_loss','ua_equal','ua_block'),('ub_specific_loss','ub_equal','ub_block'),('ab_specific_loss','ab_equal','ab_block')]


def analyse(n,frame,scope='all_attempted'):
    metrics=['accuracy_change','margin_change'];tests=[]
    units=frame.groupby(['family_id','scene_id','variant','condition'])[metrics].mean().reset_index()
    fm=units.groupby(['family_id','condition'])[metrics].mean().reset_index()
    for name,a,b in CONTRASTS:
        for m in metrics:
            values=n['paired'](fm[fm.condition.eq(a)],fm[fm.condition.eq(b)],'family_id',m)
            tests.append(dict(contrast=name,metric=m,scope=scope,**n['paired_summary'](values,['repair',scope,name,m])))
    for r,q in zip(tests,n['bh']([r['p_value'] for r in tests])):r['q_value']=float(q)
    return n['pd'].DataFrame(tests),fm


def confirmation(n):
    fit_examples(n)
    root=n['ROOT']/'repair'; selection=n['read_json'](root/'selected_route.json')
    assert selection['vectors_sha256']==n['file_hash'](root/'frozen_directions.npz')
    assert selection['direction_support_sha256']==n['file_hash'](n['ROOT']/'prediction/support.json')
    with n['np'].load(root/'frozen_directions.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    layers=(selection['upstream_layer'],selection['semantic_layer'],selection['symbol_layer'])
    protocol=dict(model=n['MODEL_ID'],revision=n['MODEL_REVISION'],selection=n['digest'](selection),
        implementation=n['IMPLEMENTATION_SHA256'],families=64,seed=2026102601,conditions=35,
        layers=list(layers),bootstrap_unit='physical_family',replicates=10000,primary_tests=14,
        gate='repair advantage over zero and matched random before interpreting selective dependence')
    out=root/'confirmation';out.mkdir(exist_ok=True)
    n['freeze'](out/'protocol.json',protocol)
    families,support_path=n['independent_support'](n['PROJECT_ROOT'],n['DATASET_DIR'],out/'cohort',64,2026102601,protocol)
    from visual_torque_toolkit.render import RenderStyle,render_scene
    from visual_torque_toolkit.world import Scene,Side,Stack
    images=out/'cohort/images';images.mkdir(exist_ok=True)
    for family in families:
        for r in family['scenes']:
            p=images/r['image_filename']
            if not p.is_file():
                style=RenderStyle(ticks_per_side=6) if r['style']=='standard' else RenderStyle(ticks_per_side=6,background=(239,245,255),block_fill=(37,99,235))
                render_scene(Scene(r['scene_id'],tuple(Stack(str(i),Side(s['side']),int(s['distance']),int(s['units']))
                    for i,s in enumerate(r['stacks']))),p,style)
    load_model(n);load_kernels(n,['repair_examples.py'])
    assert selection['runtime']==n['RUNTIME_IDENTITY'],'Development runtime changed; restore the same environment.'
    identity=dict(protocol=protocol,runtime=n['RUNTIME_IDENTITY'],support_sha256=n['file_hash'](support_path),
        images={p.name:n['file_hash'](p) for p in sorted(images.glob('*.png'))})
    n['freeze'](out/'capture_identity.json',identity)
    records=[]; upstream=arrays['upstream']*selection['dose']; semantic=arrays['semantic']
    for family in n['tqdm'](families,desc='Repair confirmation: 64 independent families, 35 conditions'):
        fid=family['family_id']
        for scene in family['scenes']:
            for variant,semantics in n['two_mappings'](scene,fid).items():
                ex=n['new_example'](scene['scene_id'],images/scene['image_filename'],scene['features'],semantics,variant,'repair_validation')
                def compute(ex=ex,scene=scene,semantics=semantics,variant=variant,family=family):
                    clean,h=n['final_forward'](ex,layers);sign=1. if ex.net_torque>0 else -1.
                    assert ex.net_torque!=0
                    opposite='right' if ex.gold_semantic=='left' else 'left'
                    wrong=n['LABELS'][semantics.index(opposite)]
                    code=arrays['code_'+wrong+'_'+ex.gold_label]
                    ops=conditions(layers,sign,upstream,semantic,code,h,[2026102601,fid,ex.example_id])
                    rows=[]
                    for name,seed,operation in ops:
                        rec=clean if name=='zero' else n['final_forward'](ex,operations=operation)[0]
                        rows.append(dict(family_id=fid,stratum=family['stratum'],scene_id=str(ex.scene_id),variant=variant,
                            physical_outcome=ex.gold_semantic,mirror=scene['mirror'],style=scene['style'],
                            condition=name,random_seed=seed,correct=bool(rec['correct']),baseline_correct=bool(clean['correct']),
                            accuracy_change=float(rec['correct'])-float(clean['correct']),
                            margin_change=n['correct_margin'](rec)-n['correct_margin'](clean),
                            repaired=not clean['correct'] and rec['correct'],harmed=clean['correct'] and not rec['correct'],
                            answer_changed=rec['predicted_semantic']!=clean['predicted_semantic'],
                            valid=all(v.get('matched',True) for v in rec['write_audits'].values()),record=rec))
                    return audit_rows(n,rows,layers)
                records.extend(n['cached_json'](out/'trial_shards',ex.example_id,
                    dict(capture=identity,input=n['example_stamp'](ex)),compute))
    assert len(records)==64*8*35
    frame=n['pd'].DataFrame([{k:v for k,v in r.items() if k!='record'} for r in records])
    assert not frame.duplicated(['family_id','scene_id','variant','condition','random_seed']).any()
    assert frame.groupby('family_id').size().eq(8*35).all()
    frame.to_csv(out/'all_conditions.csv',index=False);n['save_json'](out/'intervention_records.json',records)
    frame[~frame.valid].to_csv(out/'technical_failures.csv',index=False)
    tests,fm=analyse(n,frame);tests.to_csv(out/'primary_tests.csv',index=False);fm.to_csv(out/'family_metrics.csv',index=False)
    valid_ids=set(frame.groupby('family_id').valid.all().loc[lambda x:x].index)
    if valid_ids:analyse(n,frame[frame.family_id.isin(valid_ids)],'valid_family_sensitivity')[0].to_csv(out/'valid_family_sensitivity.csv',index=False)
    counts=frame.groupby(['family_id','scene_id','variant','condition'])[['repaired','harmed','baseline_correct','correct','answer_changed']].mean().reset_index()
    counts.groupby('condition')[['repaired','harmed','baseline_correct','correct','answer_changed']].agg(['sum','count']).to_csv(out/'repair_harm_counts.csv')
    decisions={}
    def passes(name,metric):
        r=tests[tests.contrast.eq(name)&tests.metric.eq(metric)].iloc[0]
        return bool(r.lower>0 and r.q_value<.05)
    def valid(names):return bool(frame[frame.condition.isin(names)].valid.all())
    for metric in ('accuracy_change','margin_change'):
        repair={s:passes(s+'_vs_zero',metric) and passes(s+'_vs_random',metric) and valid([s,s+'_random']) for s in ('u','a')}
        depend={}
        for name,start,dest in [('ua','u','a'),('ub','u','b'),('ab','a','b')]:
            depend[name]=bool(repair[start] and passes(name+'_specific_loss',metric) and
                valid([name+'_equal',name+'_block','self_'+dest,'self_random_'+dest]))
        decisions[metric]=dict(repair_confirmed=repair,selective_dependence_confirmed=depend)
    n['save_json'](out/'completion.json',dict(status='heldout_repair_confirmation_complete',families=64,
        presentations=512,condition_records=len(records),development_eligible=selection['selected']['eligible'],
        fully_valid_families=len(valid_ids),decisions=decisions,
        interpretation='truth-guided local intervention; not a demonstration of a complete native physics circuit'))
    print(tests.to_string(index=False));print(decisions)
