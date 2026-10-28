"""CPU protocol helpers for the independent confirmation notebooks.

Embedded after followup_core.py. These functions do not execute a model.
"""


def frozen_arrays(path, arrays):
    arrays = {k: np.asarray(v, dtype=np.float32) for k, v in arrays.items()}
    assert arrays and all(np.isfinite(v).all() for v in arrays.values())
    path = Path(path)
    if path.is_file():
        with np.load(path, allow_pickle=False) as old:
            assert set(old.files) == set(arrays) and all(np.array_equal(old[k], v) for k, v in arrays.items()), path
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix('.npz.tmp')
        with temp.open('wb') as f:
            np.savez_compressed(f, **arrays)
        temp.replace(path)
    return file_hash(path)


def fresh_geometry_index(project, dataset_dir, current_support):
    """Read all known old cohorts, including already started v1 confirmations."""
    current_support = Path(current_support).resolve()
    paths = {Path(dataset_dir) / 'scenes.jsonl'}
    for pattern in ('*manifest*.jsonl', '*support*.jsonl', '*geometries*.jsonl', 'support.json'):
        paths.update(p for p in (Path(project) / 'results').rglob(pattern)
                     if current_support not in p.resolve().parents)
    excluded, files = set(), {}
    for path in sorted(paths):
        value = read_jsonl(path) if path.suffix == '.jsonl' else read_json(path)
        signatures = geometries_in(value)
        if signatures:
            excluded.update(signatures)
            files[str(path.resolve())] = file_hash(path)
    assert excluded and str((Path(dataset_dir) / 'scenes.jsonl').resolve()) in files
    return excluded, files


def independent_support(project, dataset_dir, directory, n, seed, protocol):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    excluded, current_files = fresh_geometry_index(project, dataset_dir, directory)
    snapshot = directory / 'exclusions.json'
    if snapshot.is_file():
        old = read_json(snapshot)
        for name, sha in old['files'].items():
            assert file_hash(name) == sha, f'Historical geometry input changed: {name}'
    else:
        save_json(snapshot, dict(files=current_files, signatures=sorted(excluded)))
    config = dict(n=n, seed=seed, generator='four_strata_mirror_render_1',
                  exclusions_sha256=file_hash(snapshot), protocol=digest(protocol))
    freeze(directory / 'protocol.json', config)
    path = directory / 'support.json'
    if path.is_file():
        saved = read_json(path)
        assert saved['config'] == config
        families = saved['families']
    else:
        families = repair_families(n, seed, excluded)
        save_json(path, dict(config=config, families=families))
    assert len(families) == n and len({f['family_id'] for f in families}) == n
    assert len(geometries_in(families)) == n
    assert not geometries_in(families) & excluded, (
        'This cohort overlaps a previous or newly registered experiment; changing RUN_ID does not establish independence.')
    return families, path


def recover_gemma_candidate(source, manifest_path, dataset_scenes, revision, progress=lambda rows: rows):
    """Reconstruct the one specified candidate from old TRAIN caches only."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.cross_decomposition import PLSRegression
    import sklearn
    source = Path(source)
    selection_path = source / 'selected_route.json'
    identity_path = source / 'development_identity.json'
    selection, identity = read_json(selection_path), read_json(identity_path)
    assert selection['model'] == 'gemma12b' and selection['revision'] == revision
    assert selection['eligible'] is False, 'This entry preserves the supplied negative development result.'
    assert selection['training_identity'] == digest(identity)
    assert identity['manifest'] == file_hash(manifest_path)
    manifest = read_jsonl(manifest_path)
    train_ids = sorted(str(r['scene_id']) for r in manifest if r['split'] == 'probe_train')
    dev_ids = sorted(str(r['scene_id']) for r in manifest if r['split'] == 'probe_dev')
    assert len(train_ids) == 235 and len(dev_ids) == 19 and not set(train_ids) & set(dev_ids)
    assert set(train_ids) == set(selection['fit_scenes'])
    scene_map = {str(r['scene_id']): r for r in read_jsonl(dataset_scenes)}
    states, targets, files = [], [], {str(selection_path): file_hash(selection_path),
        str(identity_path): file_hash(identity_path), str(manifest_path): file_hash(manifest_path),
        str(dataset_scenes): file_hash(dataset_scenes)}
    for sid in progress(train_ids):
        path = source / 'capture' / (digest(sid)[:24] + '.json')
        cached = read_json(path)
        assert cached['key'] == sid and cached['identity'] == digest(identity), path
        assert len(cached['result']['scores']) == 6
        assert len(identity['inputs'][sid]) == 6
        assert {e['split'] for e in identity['inputs'][sid]} == {'probe_train'}
        y_values = {int(e['torque']) for e in identity['inputs'][sid]}
        assert len(y_values) == 1
        target = next(iter(y_values))
        assert physics(scene_map[sid]['stacks'])['net_torque'] == target
        states.append(cached['result']['mean_hidden']['31'])
        targets.append(target)
        files[str(path)] = file_hash(path)
    X, y = np.asarray(states, float), np.asarray(targets, float)
    assert np.isfinite(X).all() and (y > 0).any() and (y < 0).any()
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    pls = PLSRegression(n_components=4, scale=False, max_iter=1000).fit(Z, y.reshape(-1, 1))
    scores = pls.transform(Z)
    difference = scores[y > 0].mean(0) - scores[y < 0].mean(0)
    decoder = np.linalg.lstsq(scores - scores.mean(0), Z - Z.mean(0), rcond=1e-5)[0]
    direction = (difference @ decoder) * scaler.scale_ / 2
    assert np.isfinite(direction).all() and np.linalg.norm(direction) > 0
    dose = float(.01 * np.median(np.linalg.norm(X, axis=1)) / np.linalg.norm(direction))
    reference = dict(model='gemma12b', revision=revision, layer=31, factor=1., dose=dose,
        method='pls4_prototype_train_only_six_map_mean', source_eligible=False,
        source_selection_sha256=file_hash(selection_path), runtime=selection['runtime'],
        reconstruction=dict(numpy=np.__version__, sklearn=sklearn.__version__, files=files,
                            fit_scene_ids=train_ids, note='Original formula; no original selected vector was exported.'))
    return direction.astype('float32'), reference


def intervention_conditions(model, sign, upstream, semantic, code, untreated, key, seeds=(0, 1, 2, 3)):
    """Condition topology is fixed before observing the recipient's answer."""
    if model == 'gemma12b':
        add = {31: dict(action='add', payload=sign * upstream)}
        return [('zero', -1, {}), ('targeted', -1, add)] + [
            ('random', s, {31: dict(action='matched_add', reference=sign * upstream, key=key + ['random', s])})
            for s in seeds]
    assert model == 'qwen27b' and set(untreated) >= {49, 55}
    u = {48: dict(action='add', payload=sign * upstream)}
    a = {49: dict(action='add', payload=sign * semantic)}
    b = {55: dict(action='add', payload=code)}
    blocks = {49: dict(action='clamp', untreated=untreated[49], axis=sign * semantic),
              55: dict(action='clamp', untreated=untreated[55], axis=code)}
    rows = [('zero', -1, {}), ('u', -1, u), ('a', -1, a), ('b', -1, b),
            ('bypass', -1, {**u, 49: blocks[49], **b})]
    for name, layer, payload in [('u', 48, sign * upstream), ('a', 49, sign * semantic)]:
        for s in seeds:
            rows.append((name + '_random', s, {layer: dict(action='matched_add', reference=payload, key=key + [name, s])}))
    for layer in (49, 55):
        rows += [(f'self_{layer}', -1, {layer: blocks[layer]}),
                 (f'self_random_{layer}', 0, {layer: dict(blocks[layer], action='random_coordinate', key=key + ['self', layer, 0])})]
    for name, start, layer in [('ua', u, 49), ('ub', u, 55), ('ab', a, 55)]:
        rows.append((name + '_block', -1, {**start, layer: blocks[layer]}))
        rows.append((name + '_coordinate', 0, {**start, layer: dict(blocks[layer], action='random_coordinate', key=key + [name, 'coordinate', 0])}))
        for s in seeds:
            rows.append((name + '_equal', s, {**start, layer: dict(blocks[layer], action='equal_remove', key=key + [name, 'equal', s])}))
    assert len(rows) == 35 and len({(n, s) for n, s, _ in rows}) == len(rows)
    return rows


def reference_for_condition(model, condition):
    if model == 'gemma12b':
        return ('targeted', 31) if condition == 'random' else None
    return {'u_random': ('u', 48), 'a_random': ('a', 49),
            'ua_equal': ('ua_block', 49), 'ub_equal': ('ub_block', 55),
            'ab_equal': ('ab_block', 55)}.get(condition)


def audit_paired_writes(rows, model, tolerance=.02):
    """Also compare against the actually executed target condition, across forwards."""
    by_condition = {(r['condition'], r['random_seed']): r for r in rows}
    clean = by_condition[('zero', -1)]['record']
    for row in rows:
        reference = reference_for_condition(model, row['condition'])
        if reference:
            target, layer = reference
            target_norm = by_condition[(target, -1)]['record']['write_audits'][str(layer)]['actual_norm']
            actual = row['record']['write_audits'][str(layer)]['actual_norm']
            error = abs(actual - target_norm) / max(target_norm, 1e-12)
            row.update(paired_reference_condition=target, paired_reference_actual_norm=target_norm,
                       paired_actual_relative_error=error, valid=bool(row['valid'] and error <= tolerance))
        if row['condition'].startswith('self_'):
            scores = np.asarray(row['record']['candidate_logprobs'], float)
            zero = np.asarray(clean['candidate_logprobs'], float)
            gap = float(np.max(np.abs((scores - scores.mean()) - (zero - zero.mean()))))
            row.update(self_score_gap=gap, self_prediction_same=row['record']['predicted_semantic'] == clean['predicted_semantic'])
            row['valid'] = bool(row['valid'] and gap <= .15 and row['self_prediction_same'])
    return rows


def primary_contrasts(model):
    if model == 'gemma12b':
        return [('target_vs_zero', 'targeted', 'zero'), ('target_vs_random', 'targeted', 'random')]
    return [('u_vs_zero', 'u', 'zero'), ('u_vs_random', 'u', 'u_random'),
            ('a_vs_zero', 'a', 'zero'), ('a_vs_random', 'a', 'a_random'),
            ('ua_specific_loss', 'ua_equal', 'ua_block'),
            ('ub_specific_loss', 'ub_equal', 'ub_block'), ('ab_specific_loss', 'ab_equal', 'ab_block')]


def paired_summary(values, key, two_sided=False, replicates=10000):
    """Family percentile CI; paired sign-flip test (symmetric-null assumption)."""
    x = np.asarray(values, float)
    result = bootstrap(x, key, replicates)
    rng = np.random.default_rng(int(digest(['sign_flip', key])[:16], 16))
    observed = abs(x.mean()) if two_sided else x.mean()
    exceed = 0
    for offset in range(0, replicates, 250):
        signs = rng.choice([-1., 1.], size=(min(250, replicates - offset), len(x)))
        null = (signs * x).mean(1)
        if two_sided:
            null = abs(null)
        exceed += int((null >= observed - 1e-15).sum())
    result['p_value'] = (exceed + 1) / (replicates + 1)
    return result


def test_family_contrasts(frame, model, scope='all_attempted'):
    metrics = ['accuracy_change', 'margin_change']
    # Seeds first, then the fixed scene/mapping variants, then independent families.
    units = frame.groupby(['family_id', 'scene_id', 'variant', 'condition'])[metrics].mean().reset_index()
    means = units.groupby(['family_id', 'condition'])[metrics].mean().reset_index()
    tests = []
    for name, a, b in primary_contrasts(model):
        left, right = means[means.condition.eq(a)], means[means.condition.eq(b)]
        for metric in metrics:
            values = paired(left, right, 'family_id', metric)
            tests.append(dict(contrast=name, metric=metric, scope=scope,
                              **paired_summary(values, [model, scope, name, metric])))
    tests = pd.DataFrame(tests)
    tests['q_value'] = bh(tests.p_value)
    assert len(tests) == (4 if model == 'gemma12b' else 14)
    return tests, means


def confirmation_decisions(tests, frame, model):
    def passes(name, metric):
        row = tests[tests.contrast.eq(name) & tests.metric.eq(metric)].iloc[0]
        return bool(row.lower > 0 and row.q_value < .05)
    def valid(conditions):
        sub = frame[frame.condition.isin(conditions)]
        return bool(len(sub) and sub.valid.all())
    if model == 'gemma12b':
        result = {metric: bool(passes('target_vs_zero', metric) and passes('target_vs_random', metric)
                              and valid(['targeted', 'random'])) for metric in ('accuracy_change', 'margin_change')}
        return dict(accuracy_repair_confirmed=result['accuracy_change'], score_effect_confirmed=result['margin_change'],
                    next_step='eligible_for_separate_L31_to_L47_experiment' if result['accuracy_change'] else 'stop_this_candidate_no_rescan')
    result = {}
    for metric in ('accuracy_change', 'margin_change'):
        repair = {site: bool(passes(site + '_vs_zero', metric) and passes(site + '_vs_random', metric)
                            and valid([site, site + '_random'])) for site in ('u', 'a')}
        paths = {}
        for name, start, layer in [('ua', 'u', 49), ('ub', 'u', 55), ('ab', 'a', 55)]:
            paths[name] = bool(repair[start] and passes(name + '_specific_loss', metric)
                               and valid([name + '_equal', name + '_block', f'self_{layer}', f'self_random_{layer}']))
        result[metric] = dict(repair_confirmed=repair, selective_dependence_confirmed=paths)
    return result


def validate_trial_grid(frame, families, model):
    expected = [('zero', -1), ('targeted', -1)] + [('random', s) for s in range(4)] if model == 'gemma12b' else None
    if expected is None:
        mock = intervention_conditions('qwen27b', 1, np.ones(2), np.ones(2), np.ones(2), {49: np.zeros(2), 55: np.zeros(2)}, [])
        expected = [(n, s) for n, s, _ in mock]
    assert not frame.duplicated(['family_id', 'scene_id', 'variant', 'condition', 'random_seed']).any()
    assert set(frame.family_id) == {f['family_id'] for f in families}
    for family in families:
        sub = frame[frame.family_id.eq(family['family_id'])]
        assert set(sub.scene_id) == {s['scene_id'] for s in family['scenes']}
        assert len(sub) == 4 * 2 * len(expected)
        for _, unit in sub.groupby(['scene_id', 'variant']):
            assert set(zip(unit.condition, unit.random_seed)) == set(expected)
        assert set(sub.variant) == {'forward', 'reverse'}


def orthogonal_failure_table(records):
    frame = pd.DataFrame(records)
    sub = frame[frame.condition.eq('orthogonal')].copy()
    assert len(sub) and {'norm_matched', 'orthogonal_preserved', 'random_seed'} <= set(sub)
    assert sub.groupby('scene_id').random_seed.nunique().eq(4).all()
    assert sub.groupby('scene_id').size().eq(4).all()
    norm = sub.norm_matched.eq(True)
    preserve = sub.orthogonal_preserved.eq(True)
    known = sub.norm_matched.notna() & sub.orthogonal_preserved.notna()
    sub['failure_kind'] = np.select([~known, norm & preserve, ~norm & preserve, norm & ~preserve],
        ['missing_audit', 'valid', 'norm_only', 'readout_only'], default='both')
    sub['valid'] = known & norm & preserve
    return sub
