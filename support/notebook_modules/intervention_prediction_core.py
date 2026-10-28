"""CPU-only protocol, support, prediction and analysis for prospective interventions.

Embedded with selected, unchanged followup_core utilities by the notebook builder.
No function in this module executes a language model.
"""
from datetime import datetime, timezone
import math

SEMANTICS = ('left', 'balanced', 'right')
LABELS = ('A', 'B', 'C')
SCHEMA = 'prospective-intervention-effects-v1'


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def center(x):
    x = np.asarray(x, dtype=float)
    return x - x.mean(axis=-1, keepdims=True)


def bf16(x):
    """Round finite FP32 values to BF16, ties-to-even; return FP32 numpy values."""
    a = np.asarray(x, dtype=np.float32).copy()
    assert np.isfinite(a).all()
    u = a.view(np.uint32)
    u = (u + np.uint32(0x7fff) + ((u >> 16) & 1)) & np.uint32(0xffff0000)
    return u.view(np.float32)


def state_hash(x):
    return hashlib.sha256(np.asarray(x, dtype='<f4').tobytes()).hexdigest()


def inventory_geometries(project, output_root, extra_roots=()):
    """Index available historical geometry, including JSON supports, once per run.

    The scope is the files present at initial freezing, not all experiments ever run.
    Unreadable candidate files stop the inventory instead of silently disappearing.
    """
    lock = output_root / 'historical_geometry_snapshot.json'
    if lock.exists():
        obj = read_json(lock)
        for name, expected in obj['files'].items():
            assert file_hash(name) == expected, 'A frozen historical input changed: ' + name
        return {tuple(tuple(t) for t in s) for s in obj['signatures']}, obj
    paths = set()
    for root in (project / 'data', project / 'results', *extra_roots):
        root = Path(root)
        if not root.exists():
            continue
        for p in root.rglob('*'):
            if not p.is_file() or output_root.resolve() in p.resolve().parents:
                continue
            if p.suffix.lower() not in ('.json', '.jsonl'):
                continue
            if any(k in p.name.lower() for k in ('manifest', 'support', 'famil', 'scene', 'geometr')):
                paths.add(p)
    signatures, hashes = set(), {}
    for p in sorted(paths):
        obj = read_jsonl(p) if p.suffix == '.jsonl' else read_json(p)
        signatures.update(geometries_in(obj))
        hashes[str(p.resolve())] = file_hash(p)
    assert signatures, 'No historical scenes found. Restore the project data/results before generating new support.'
    obj = dict(files=hashes, signatures=sorted(signatures), created_at_utc=utc_now(),
               scope='available geometry-bearing JSON/JSONL files only; unimported archives are not indexed')
    save_json(lock, obj)
    return signatures, obj


def physical_families(n, seed, excluded):
    """Cross donor distance x recipient distance, with exact one-block changes.

    Baseline recipient net torque is equal across its two distances. Background
    stacks compensate for changing distance; this is a factorial family, not a
    claim that changing distance leaves every other visible primitive unchanged.
    """
    assert n % 4 == 0
    rng = np.random.default_rng(seed)
    used = set(excluded)
    families = []
    pools = {}

    def sample_pair(d, net, dw, mirror, local, target_units):
        key=(d,net,target_units)
        if key not in pools:
            choices=[]
            for u2 in range(1,8):
                for r1,r2 in combinations(range(1,7),2):
                    for v1 in range(1,8):
                        numerator=target_units*d + u2*6-v1*r1-net
                        if numerator%r2==0 and 1<=numerator//r2<=7:
                            choices.append((u2,r1,r2,v1,numerator//r2))
            pools[key]=choices
        for index in rng.permutation(len(pools[key])):
            u2,r1,r2,v1,v2=pools[key][int(index)]
            # The changed stack is always the innermost stack on its side.
            a = [dict(side=s, distance=int(x), units=int(w)) for s,x,w in
                 [('left',d,target_units),('left',6,u2),('right',r1,v1),('right',r2,v2)]]
            b = [dict(s) for s in a]
            b[0]['units'] += dw
            if mirror:
                for x in a+b:
                    x['side'] = 'right' if x['side'] == 'left' else 'left'
            sigs = {geometry_signature(a), geometry_signature(b)}
            if len(sigs) == 2 and not sigs & (used | local):
                return a, b, sigs
        raise RuntimeError(f'No unused matched geometry: d={d}, net={net}, dw={dw}, target_units={target_units}. '
                           'Do not relax support rules after looking at outputs.')

    for i in range(n):
        mirror = bool(i % 2)
        dw = 1 if (i // 2) % 2 == 0 else -1
        near = int(rng.choice([1, 2]))
        far = int(rng.choice([4, 5]))
        target_units = int(rng.integers(2,7))
        clearance = int(rng.integers(near+1, far))
        donor_sign = 1 if (i // 4) % 2 == 0 else -1
        donor_net = donor_sign * int(rng.integers(far+2, far+10))
        local, roles = set(), {}
        for owner in ('source', 'recipient'):
            for dname, d in (('near', near), ('far', far)):
                net = donor_net if owner == 'source' else -dw*clearance
                a, b, sigs = sample_pair(d, net, dw, mirror, local, target_units)
                local.update(sigs)
                for endpoint, stacks in (('before', a), ('after', b)):
                    roles[f'{owner}_{dname}_{endpoint}'] = dict(stacks=stacks, features=physics(stacks))
        fid = 'physical_' + digest([seed, i, roles])[:20]
        for name, role in roles.items():
            role.update(scene_id=fid+'__'+name, image_filename=fid+'__'+name+'.png')
        side = 'right' if mirror else 'left'
        families.append(dict(family_id=fid, module='physical', stratum=f'{side}_dw{dw}',
                             near=near, far=far, dw=dw, target_side=side, roles=roles))
        used.update(local)
    return families


def assign_splits(families, n_calibration, n_validation, seed):
    groups = defaultdict(list)
    for f in families:
        groups[f['stratum']].append(f)
    assert len(groups) == 4
    for key, rows in groups.items():
        rows.sort(key=lambda f: digest([seed, f['family_id']]))
        assert len(rows) == (n_calibration+n_validation)//4
        for i, f in enumerate(rows):
            f['split'] = 'calibration' if i < n_calibration//4 else 'validation'
    for phase, n in [('calibration', n_calibration), ('validation', n_validation)]:
        selected = sorted([f for f in families if f['split'] == phase], key=lambda f:f['family_id'])
        # A fixed quarter of families, balanced across the four design strata.
        for key in groups:
            subset = [f for f in selected if f['stratum'] == key]
            for j, f in enumerate(subset):
                f['random_control_family'] = j < len(subset)//4
    return families


def generate_support(n_calibration, n_validation, seed, excluded):
    assert n_calibration % 16 == n_validation % 16 == 0
    n = n_calibration+n_validation
    content = inverse_families(n, seed, excluded)
    for f in content:
        f.update(module='content', stratum=f["margin_bin"]+'_'+f['roles']['base']['features']['outcome'])
    physical = physical_families(n, seed+1, excluded | geometries_in(content))
    rows = assign_splits(content, n_calibration, n_validation, seed)
    rows += assign_splits(physical, n_calibration, n_validation, seed)
    signatures = [geometry_signature(r['stacks']) for f in rows for r in f['roles'].values()]
    assert len(signatures) == len(set(signatures))
    assert not set(signatures) & excluded
    return rows


def mapping_reference(fid):
    offset = int(digest(['mapping', fid])[:8], 16) % 3
    sem = list(SEMANTICS[offset:] + SEMANTICS[:offset])
    return [dict(position=i, label=LABELS[i], semantic=sem[i]) for i in range(3)]


def mapping_conflict(reference, semantic, reverse=False):
    pos = next(x['position'] for x in reference if x['semantic'] == semantic)
    label = reference[pos]['label']
    ps = [i for i in range(3) if i != pos]
    ss = [s for s in SEMANTICS if s != semantic]
    ls = [s for s in LABELS if s != label]
    if reverse:
        ss.reverse()
        ls.reverse()
    result = [None]*3
    result[pos] = dict(position=pos, label=ls[0], semantic=ss[0])
    result[ps[0]] = dict(position=ps[0], label=label, semantic=ss[1])
    result[ps[1]] = dict(position=ps[1], label=ls[1], semantic=semantic)
    assert len({next(x['label'] for x in result if x['semantic'] == semantic),label,result[pos]['label']}) == 3
    return result


def transport_vector(values, source_mapping, recipient_mapping, coordinate):
    values = np.asarray(values, dtype=float)  # Always A/B/C order.
    source_index = {x[coordinate]: LABELS.index(x['label']) for x in source_mapping}
    target = {x['label']: values[source_index[x[coordinate]]] for x in recipient_mapping}
    return center([target[label] for label in LABELS])


def predictor_features(row):
    """Explicit allowlist. No recipient gold, CF scores or post-intervention state."""
    s0 = np.asarray(row['s0'], float)
    source_delta = np.asarray(row['source_delta'], float)
    sm, rm = row['source_mapping'], row['recipient_mapping']
    features = {}
    for name, coord in [('semantic','semantic'),('letter','label'),('position','position')]:
        features['source_'+name] = transport_vector(source_delta, sm, rm, coord).tolist()
        # Source argmax endpoint changes; unchanging source decisions give zero.
        cat = np.eye(3)[row['source_after_argmax']] - np.eye(3)[row['source_before_argmax']]
        features['choice_'+name] = transport_vector(cat, sm, rm, coord).tolist()
        features['copy_'+name] = transport_vector(np.eye(3)[row['source_after_argmax']],sm,rm,coord).tolist()
    weights = np.exp(s0-s0.max()); probs = weights/weights.sum()
    gap = float(np.sort(s0)[-1]-np.sort(s0)[-2])
    conf = [*s0, gap, float(-np.sum(probs*np.log(probs+1e-12))),
            float(row['planned_norm']), float(row['relative_norm'])]
    features['confidence'] = conf
    if row['module'] == 'physical':
        axis = np.array([{'left':1., 'balanced':0., 'right':-1.}[
            next(x['semantic'] for x in rm if x['label'] == label)] for label in LABELS])
        sign = 1 if row['target_side'] == 'left' else -1
        signed_dw = sign*int(row['dw'])
        features['recipient_weight'] = (axis*signed_dw*row['recipient_distance']).tolist()
        features['source_torque'] = (axis*signed_dw*row['source_distance']).tolist()
        # Privileged structured-input baseline, explicitly labelled in the protocol.
        features['physical_difficulty'] = conf + list(map(float, row['baseline_physics']))
    return features


def model_names(module):
    names = ['null','confidence', 'source_semantic','source_letter','source_position',
             'choice_semantic','choice_letter','choice_position',
             'copy_semantic','copy_letter','copy_position']
    if module == 'physical':
        names += ['physical_difficulty','recipient_weight','source_torque',
                  'behavior_plus_weight','behavior_plus_source_torque','natural_response']
    return names


def ridge_inputs(row, name):
    f = predictor_features(row)
    if name in ('confidence', 'physical_difficulty'):
        return f[name]
    # Same answer/confidence baseline on both sides of the nested comparison.
    values = f['physical_difficulty'] + f['source_semantic'] + f['source_letter'] + f['source_position']
    if name == 'behavior_plus_weight':
        return values + f['recipient_weight']
    if name == 'behavior_plus_source_torque':
        return values + f['source_torque']
    if name == 'behavior':
        return values
    raise KeyError(name)


def all_model_names(module):
    names = model_names(module)
    return names + (['behavior'] if module == 'physical' else [])


def fit_predictor(rows, name):
    assert rows and all(r['split'] == 'calibration' for r in rows)
    y = np.array([center(r['actual_delta']) for r in rows])
    if name == 'null':
        return dict(kind='null')
    if name in ('confidence','physical_difficulty','behavior','behavior_plus_weight','behavior_plus_source_torque'):
        x = np.asarray([ridge_inputs(r,name) for r in rows], float)
        mean, scale = x.mean(0), x.std(0)
        scale[scale < 1e-8] = 1.
        z = np.column_stack([np.ones(len(x)), (x-mean)/scale])
        penalty = np.eye(z.shape[1]); penalty[0,0] = 0
        coef = np.linalg.solve(z.T@z/len(z) + penalty, z.T@y/len(z))
        return dict(kind='ridge', name=name, mean=mean, scale=scale, coef=coef, ridge_alpha=1.)
    x = np.asarray([r['natural_input_delta'] if name=='natural_response' else predictor_features(r)[name] for r in rows])
    beta = max(0., float(np.sum(x*y)/max(float(np.sum(x*x)), 1e-12)))
    return dict(kind='scalar', name=name, beta=beta)


def predict_one(row, fitted):
    if fitted['kind'] == 'null':
        return np.zeros(3)
    if fitted['kind'] == 'scalar':
        value = row['natural_input_delta'] if fitted['name']=='natural_response' else predictor_features(row)[fitted['name']]
        return center(np.asarray(value)*fitted['beta'])
    x = np.asarray(ridge_inputs(row, fitted['name']))
    z = np.r_[1., (x-np.asarray(fitted['mean']))/np.asarray(fitted['scale'])]
    return center(z@np.asarray(fitted['coef']))


def operation_key(row):
    return f"{row['module']}:L{row['layer']}"


def fit_all(rows):
    """Five family folds select a strong answer baseline; test scores never select."""
    output = {}
    for key in sorted({operation_key(r) for r in rows}):
        sub = [r for r in rows if operation_key(r) == key]
        fids = sorted({r['family_id'] for r in sub}, key=lambda s:digest(['fold',s]))
        folds = {fid:i%5 for i,fid in enumerate(fids)}
        names = all_model_names(sub[0]['module'])
        cv = {}
        for name in names:
            losses = defaultdict(list)
            for fold in range(5):
                train = [r for r in sub if folds[r['family_id']] != fold]
                valid = [r for r in sub if folds[r['family_id']] == fold]
                if not valid:
                    continue
                fitted = fit_predictor(train,name)
                for r in valid:
                    losses[r['family_id']].append(float(np.mean((predict_one(r,fitted)-r['actual_delta'])**2)))
            cv[name] = float(np.mean([np.mean(v) for v in losses.values()]))
        eligible = [n for n in names if n not in ('recipient_weight','source_torque','behavior_plus_weight',
                                                'behavior_plus_source_torque','natural_response')]
        best = min(eligible, key=lambda n:(cv[n],n))
        output[key] = dict(models={n:fit_predictor(sub,n) for n in names}, cv_family_mse=cv,
                           selected_baseline=best, calibration_families=fids, folds=folds,
                           baseline_gap_quartiles=np.quantile(
                               [np.sort(r['s0'])[-1]-np.sort(r['s0'])[-2] for r in sub],[.25,.5,.75]).tolist())
    return output


def seal_predictions(path, rows, fitted, identity):
    """Write once and verify on resume. Call before the first validation patch."""
    assert len({r['trial_id'] for r in rows}) == len(rows)
    records = []
    for r in rows:
        models = fitted[operation_key(r)]['models']
        records.append(dict(trial_id=r['trial_id'], family_id=r['family_id'],
                            input_sha256=digest({k:v for k,v in r.items() if k not in ('gold','cf_gold')}),
                            predicted_deltas={n:predict_one(r,m).tolist() for n,m in models.items()}))
    body = dict(identity=identity, records=records)
    if Path(path).exists():
        old = read_json(path)
        assert old['payload'] == body, 'Predictions or their inputs changed after sealing.'
    else:
        save_json(path, dict(created_at_utc=utc_now(), payload=body))
    return file_hash(path)


def paired_interval(values, key, replicates=10000):
    """Family bootstrap CI and two-sided paired sign-flip test."""
    x = np.asarray(values, float)
    assert len(x) and np.isfinite(x).all()
    rng = np.random.default_rng(int(digest(key)[:16],16))
    draws, null = [], []
    for start in range(0,replicates,250):
        n = min(250,replicates-start)
        draws.extend(rng.choice(x,(n,len(x)),replace=True).mean(1))
        null.extend((rng.choice([-1.,1.],(n,len(x)))*x).mean(1))
    lo, hi = np.quantile(draws,[.025,.975])
    p = (1+int(np.sum(np.abs(null) >= abs(x.mean())-1e-15)))/(replicates+1)
    return dict(n_families=len(x), mean=float(x.mean()), lower=float(lo), upper=float(hi), p_value=float(p))


def holm(values):
    p = np.asarray(values,float); order = np.argsort(p); q = np.empty(len(p)); previous = 0.
    for rank,i in enumerate(order):
        previous = max(previous,min(1.,float(p[i])*(len(p)-rank)))
        q[i] = previous
    return q


def transition(before, after, gold):
    if before != gold and after == gold: return 'repair'
    if before == gold and after != gold: return 'harm'
    if before != after: return 'wrong_to_other_wrong'
    return 'unchanged'


def evaluate_predictions(rows, fitted, sealed, outdir):
    """All target rows, then family means. No exclusions based on desired effects."""
    outdir = Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    prediction_map = {r['trial_id']:r for r in sealed['payload']['records']}
    assert set(prediction_map) == {r['trial_id'] for r in rows}
    records = []
    for r in rows:
        s0, actual = np.asarray(r['s0']), np.asarray(r['actual_delta'])
        before, after = int(np.argmax(s0)), int(np.argmax(s0+actual))
        gold = LABELS.index(r['gold'])
        gap = float(np.sort(s0)[-1]-np.sort(s0)[-2])
        gap_bin = int(np.searchsorted(fitted[operation_key(r)]['baseline_gap_quartiles'],gap,side='right'))
        for name, predicted in prediction_map[r['trial_id']]['predicted_deltas'].items():
            p = np.asarray(predicted); pred_after = int(np.argmax(s0+p))
            records.append(dict(trial_id=r['trial_id'], family_id=r['family_id'],
                operation=operation_key(r), predictor=name, mapping=r['mapping_name'],
                source_distance=r.get('source_distance'), recipient_distance=r.get('recipient_distance'),
                source_argmax_stable=r['source_before_argmax']==r['source_after_argmax'],
                baseline_gap=gap,baseline_gap_bin=gap_bin,
                mse=float(np.mean((p-actual)**2)),
                sign_agreement=float(np.mean(np.sign(p)==np.sign(actual))),
                cosine=(float(np.dot(p,actual)/(np.linalg.norm(p)*np.linalg.norm(actual)))
                        if np.linalg.norm(p)*np.linalg.norm(actual)>1e-10 else None),
                post_answer_match=int(pred_after==after), actual_change=int(after!=before),
                predicted_change=int(pred_after!=before),
                actual_transition=transition(before,after,gold),
                predicted_transition=transition(before,pred_after,gold)))
    df = pd.DataFrame(records); df.to_csv(outdir/'trial_prediction_metrics.csv',index=False)
    metrics = ['mse','post_answer_match','actual_change','predicted_change','sign_agreement','cosine']
    fam = df.groupby(['operation','predictor','family_id'])[metrics].mean().reset_index()
    fam.to_csv(outdir/'family_prediction_metrics.csv',index=False)
    summary = fam.groupby(['operation','predictor'])[metrics].mean().reset_index()
    summary.to_csv(outdir/'prediction_summary.csv',index=False)
    tests = []
    for op, sub in fam.groupby('operation'):
        pivot = sub.pivot(index='family_id',columns='predictor',values='mse')
        baseline = fitted[op]['selected_baseline']
        if op.startswith('content:'):
            comparisons = [('source_semantic','source_letter'),('source_letter','source_position'),
                           (baseline,'null'), (baseline,'confidence')]
        else:
            comparisons = [('recipient_weight','source_torque'), ('recipient_weight',baseline),
                           ('behavior_plus_weight','behavior'),('behavior_plus_weight','behavior_plus_source_torque'),
                           ('recipient_weight','natural_response')]
        for index,(a,b) in enumerate(comparisons):
            result = paired_interval((pivot[b]-pivot[a]).values, [op,index,a,b])
            tests.append(dict(operation=op, better_if_positive=a, comparator=b, **result))
    for t,q in zip(tests,holm([t['p_value'] for t in tests])):
        t['holm_p_within_model'] = float(q)
    pd.DataFrame(tests).to_csv(outdir/'primary_comparisons.csv',index=False)
    # Rare events get explicit denominators; undefined recall is blank, never 100%.
    transitions = []
    for (op,name),g in df.groupby(['operation','predictor']):
        for cls in ('repair','harm','wrong_to_other_wrong','unchanged'):
            true = g.actual_transition.eq(cls); pred = g.predicted_transition.eq(cls)
            tp = int((true&pred).sum()); nt = int(true.sum()); npred = int(pred.sum())
            transitions.append(dict(operation=op,predictor=name,event=cls,n_actual=nt,n_predicted=npred,
                n_true_positive=tp,recall=tp/nt if nt else None,
                precision=tp/npred if npred else None,
                f1=2*tp/(nt+npred) if nt+npred else None))
    event_frame=pd.DataFrame(transitions)
    event_frame.to_csv(outdir/'transition_events.csv',index=False)
    event_frame.groupby(['operation','predictor'])[['recall','f1']].mean().rename(
        columns={'recall':'macro_recall_over_observed_classes','f1':'macro_f1_over_nonempty_classes'}).to_csv(
            outdir/'transition_macro_metrics.csv')
    for column in ('mapping','source_argmax_stable','baseline_gap_bin'):
        per_family = df.groupby(['operation','predictor',column,'family_id'])[metrics].mean().reset_index()
        per_family.groupby(['operation','predictor',column])[metrics].mean().to_csv(outdir/f'by_{column}.csv')
    return summary, pd.DataFrame(tests)
