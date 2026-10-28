"""CPU-only diagnosis of saved independent-confirmation results.

Embedded after the frozen analysis helpers. Never changes input records or gates.
"""


def bool_value(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    if isinstance(value, (int, float, np.number)) and value in (0, 1):
        return bool(value)
    raise ValueError(f'Missing or invalid boolean: {value!r}')


def tracked_json(path, inputs):
    inputs[str(path)] = file_hash(path)
    return read_json(path)


def tracked_csv(path, inputs):
    inputs[str(path)] = file_hash(path)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_science_result(root, model, inputs):
    root = Path(root)
    paths = {'completion': root/'analysis/RUN_COMPLETE.json', 'protocol': root/'protocol.json',
             'capture': root/'capture_identity.json', 'support': root/'cohort/support.json',
             'records': root/'analysis/intervention_records.json', 'primary': root/'analysis/primary_tests.csv',
             'vectors': root/'frozen_directions.npz', 'reference': root/'reference.json',
             'images': root/'cohort/images.json'}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        return dict(model=model, status='missing_inputs', issues=missing)
    result = tracked_json(paths['completion'], inputs)
    protocol = tracked_json(paths['protocol'], inputs)
    capture = tracked_json(paths['capture'], inputs)
    assert protocol['schema'] == ('qwen_three_path-1.0' if model == 'qwen27b' else 'gemma_l31_candidate-1.0')
    assert protocol['layers'] == ([48, 49, 55] if model == 'qwen27b' else [31])
    assert protocol['random_seeds'] == [0, 1, 2, 3] and protocol['norm_tolerance'] == .02
    for key in ('vectors', 'reference', 'images'):
        inputs[str(paths[key])] = file_hash(paths[key])
    assert result['status'] == 'complete' and result['model'] == protocol['model'] == model
    for key, path_key in [('protocol_sha256', 'protocol'), ('capture_sha256', 'capture'),
                          ('intervention_records_sha256', 'records')]:
        assert result[key] == file_hash(paths[path_key]), f'Completion hash mismatch: {paths[path_key]}'
    assert capture['protocol'] == protocol
    for key, path_key in [('support', 'support'), ('vectors', 'vectors'), ('reference', 'reference'), ('images', 'images')]:
        assert capture[key] == file_hash(paths[path_key]), f'Capture input changed: {paths[path_key]}'
    families = tracked_json(paths['support'], inputs)['families']
    records = tracked_json(paths['records'], inputs)
    primary = tracked_csv(paths['primary'], inputs)
    frame = pd.DataFrame([{k: v for k, v in r.items() if k not in ('record', 'clean_record')} for r in records])
    validate_trial_grid(frame, families, model)
    assert len(records) == result['condition_records'] and len(families) == result['families'] == protocol['n_families']
    assert len(primary) == protocol['primary_tests'] == (14 if model == 'qwen27b' else 4)
    assert primary.n.eq(len(families)).all() and primary.scope.eq('all_attempted').all()
    assert frame.valid.map(bool_value).all() == bool_value(result['all_controls_valid'])
    assert int(frame.groupby('family_id').valid.all().sum()) == result['fully_valid_families']
    optional = {}
    for name in ('technical_failures', 'valid_family_sensitivity', 'family_metrics', 'subgroup_descriptive',
                 'repair_and_harm_counts', 'secondary_bypass'):
        path = root/'analysis'/f'{name}.csv'
        if path.is_file():
            optional[name] = tracked_csv(path, inputs)
    return dict(model=model, status='validated_complete', issues=[], root=str(root), completion=result,
                protocol=protocol, capture=capture, families=families, records=records, frame=frame,
                primary=primary, optional=optional)


def verify_record_metrics(bundle):
    """Reconstruct baseline pairing and behavioral metrics from saved candidate scores."""
    rows = bundle['records']
    zeros = {(r['family_id'], r['scene_id'], r['variant']): r['record'] for r in rows if r['condition'] == 'zero'}
    for row in rows:
        clean = zeros[row['family_id'], row['scene_id'], row['variant']]
        assert digest(row['clean_record']) == digest(clean), 'Mixed current untreated baseline'
        out = row['record']
        assert clean['gold_semantic'] == out['gold_semantic'] == row['physical_outcome']
        correct = out['predicted_semantic'] == out['gold_semantic']
        baseline = clean['predicted_semantic'] == clean['gold_semantic']
        assert bool_value(row['correct']) == correct and bool_value(row['baseline_correct']) == baseline
        expected = {'accuracy_change': float(correct)-float(baseline),
                    'margin_change': correct_margin(out)-correct_margin(clean),
                    'repaired': not baseline and correct, 'harmed': baseline and not correct,
                    'answer_changed': out['predicted_semantic'] != clean['predicted_semantic']}
        for key, value in expected.items():
            assert abs(float(row[key])-float(value)) < 1e-10, f'Record metric mismatch: {key}'


def reconcile_primary(bundle):
    fresh, means = test_family_contrasts(bundle['frame'], bundle['model'])
    saved = bundle['primary']
    keys = ['contrast', 'metric', 'scope']
    assert not saved.duplicated(keys).any()
    joined = fresh.merge(saved, on=keys, suffixes=('_recomputed', '_saved'), how='outer', indicator=True, validate='one_to_one')
    assert joined['_merge'].eq('both').all()
    columns = ('n', 'mean', 'lower', 'upper', 'p_value', 'q_value')
    for key in columns:
        joined[key+'_absolute_difference'] = abs(joined[key+'_recomputed']-joined[key+'_saved'])
    joined['numeric_match'] = joined[[c+'_absolute_difference' for c in columns]].max(axis=1).le(1e-10)
    decision = confirmation_decisions(saved, bundle['frame'], bundle['model'])
    assert decision == bundle['completion']['decision'], 'Saved decision differs from its original rule'
    return joined, means


def saved_score_vector(record):
    values = record['candidate_logprobs']
    if isinstance(values, dict):
        return np.asarray([values[k] for k in record.get('candidate_labels', ('A', 'B', 'C'))], float)
    return np.asarray(values, float)


def inspect_saved_writes(bundle):
    rows, model = bundle['records'], bundle['model']
    by_key = {(r['family_id'], r['scene_id'], r['variant'], r['condition'], r['random_seed']): r for r in rows}
    tolerance = float(bundle['protocol']['norm_tolerance'])
    flat, trials = [], []
    for row in rows:
        local_failed = False
        local_numeric_mismatch = False
        for layer, audit in row['record']['write_audits'].items():
            has_match = 'matched' in audit
            matched = bool_value(audit['matched']) if has_match else True
            actual = float(audit['actual_norm'])
            target = audit.get('target_norm')
            numerical_error = None if target is None else abs(actual-float(target))/max(float(target), 1e-12)
            local_failed |= not matched
            if has_match and numerical_error is not None:
                # The recorded matched flag drives the original gate; this is an independent check.
                local_numeric_mismatch |= matched != (numerical_error <= tolerance)
            flat.append(dict(model=model, family_id=row['family_id'], scene_id=row['scene_id'], variant=row['variant'],
                condition=row['condition'], random_seed=row['random_seed'], layer=int(layer), action=audit['action'],
                actual_norm=actual, target_norm=target, local_relative_error_saved=audit.get('relative_error'),
                local_relative_error_recomputed=numerical_error, local_matched=matched,
                hidden_norm=audit.get('hidden_norm'), reference_coefficient=audit.get('coefficient')))
        cross_failed = False
        cross_error = target_actual = actual = None
        cross_field_consistent = True
        ref = reference_for_condition(model, row['condition'])
        if ref:
            name, layer = ref
            target_row = by_key[row['family_id'], row['scene_id'], row['variant'], name, -1]
            target_actual = float(target_row['record']['write_audits'][str(layer)]['actual_norm'])
            actual = float(row['record']['write_audits'][str(layer)]['actual_norm'])
            cross_error = abs(actual-target_actual)/max(target_actual, 1e-12)
            cross_failed = cross_error > tolerance
            cross_field_consistent = (row.get('paired_reference_condition') == name
                and abs(float(row['paired_reference_actual_norm'])-target_actual) < 1e-10
                and abs(float(row['paired_actual_relative_error'])-cross_error) < 1e-10)
        self_failed = False
        self_gap = None
        if row['condition'].startswith('self_'):
            a, b = saved_score_vector(row['record']), saved_score_vector(row['clean_record'])
            self_gap = float(np.max(abs((a-a.mean())-(b-b.mean()))))
            self_failed = self_gap > .15 or row['record']['predicted_semantic'] != row['clean_record']['predicted_semantic']
        recomputed_valid = not (local_failed or cross_failed or self_failed)
        stored_valid = bool_value(row['valid'])
        reasons = []
        if local_failed: reasons.append('local_norm_match')
        if cross_failed: reasons.append('cross_forward_norm_match')
        if self_failed: reasons.append('self_clamp_parity')
        if stored_valid != recomputed_valid: reasons.append('stored_valid_flag_inconsistent')
        if not cross_field_consistent: reasons.append('saved_cross_audit_inconsistent')
        if local_numeric_mismatch: reasons.append('local_flag_numeric_disagreement')
        trials.append(dict(model=model, family_id=row['family_id'], scene_id=row['scene_id'], variant=row['variant'],
            stratum=row['stratum'], style=row['style'], physical_outcome=row['physical_outcome'],
            condition=row['condition'], random_seed=row['random_seed'], stored_valid=stored_valid,
            recomputed_original_gate_valid=recomputed_valid, local_match_failed=local_failed,
            cross_match_failed=cross_failed, self_parity_failed=self_failed,
            saved_fields_consistent=stored_valid == recomputed_valid and cross_field_consistent and not local_numeric_mismatch,
            failure_reasons=';'.join(reasons) or 'none',
            actual_delta_norm=actual, executed_target_delta_norm=target_actual,
            absolute_norm_gap=None if actual is None else abs(actual-target_actual),
            cross_relative_error=cross_error, self_centered_score_gap=self_gap))
    return pd.DataFrame(flat), pd.DataFrame(trials)


def condition_failure_summary(trials):
    summaries = []
    for condition, sub in trials.groupby('condition'):
        failed = sub[~sub.stored_valid]
        summaries.append(dict(condition=condition, attempted_records=len(sub), failed_records=len(failed),
            affected_families=failed.family_id.nunique(), affected_scene_mappings=failed[['scene_id', 'variant']].drop_duplicates().shape[0],
            local_match_failures=int(sub.local_match_failed.sum()), cross_match_failures=int(sub.cross_match_failed.sum()),
            both_norm_checks_failed=int((sub.local_match_failed & sub.cross_match_failed).sum()),
            self_parity_failures=int(sub.self_parity_failed.sum()), inconsistent_audit_records=int((~sub.saved_fields_consistent).sum())))
    return pd.DataFrame(summaries)


def path_diagnostics(bundle, family_means):
    assert bundle['model'] == 'qwen27b'
    frame, tests, result = bundle['frame'], bundle['primary'], bundle['completion']
    gates, sensitivity, influences = [], [], []
    for path, start, layer in [('ua', 'u', 49), ('ub', 'u', 55), ('ab', 'a', 55)]:
        required = [start, start+'_random', path+'_equal', path+'_block', f'self_{layer}', f'self_random_{layer}']
        relevant = frame[frame.condition.isin(required)]
        valid_ids = set(relevant.groupby('family_id').valid.all().loc[lambda x:x].index)
        all_ids = set(frame.family_id)
        for metric in ('accuracy_change', 'margin_change'):
            row = tests[tests.contrast.eq(path+'_specific_loss') & tests.metric.eq(metric)].iloc[0]
            downstream = frame[frame.condition.isin([path+'_equal', path+'_block', f'self_{layer}', f'self_random_{layer}'])]
            gates.append(dict(path=path, metric=metric, upstream_repair_confirmed=result['decision'][metric]['repair_confirmed'][start],
                statistical_pass=bool(row.lower > 0 and row.q_value < .05), downstream_controls_valid=bool(downstream.valid.all()),
                original_confirmation=result['decision'][metric]['selective_dependence_confirmed'][path],
                all_families=len(all_ids), path_valid_families=len(valid_ids), path_invalid_families=len(all_ids-valid_ids)))
            left = family_means[family_means.condition.eq(path+'_equal')].set_index('family_id')[metric]
            right = family_means[family_means.condition.eq(path+'_block')].set_index('family_id')[metric]
            effects = left-right
            assert effects.notna().all() and len(effects) == len(all_ids)
            for scope, ids in [('all_attempted', all_ids), ('path_controls_valid_posthoc', valid_ids),
                               ('path_controls_invalid_descriptive', all_ids-valid_ids)]:
                values = effects[effects.index.isin(ids)].to_numpy(float)
                if len(values):
                    ci = bootstrap(values, ['completion_audit', path, metric, scope])
                    sensitivity.append(dict(path=path, metric=metric, scope=scope,
                        **{k: ci[k] for k in ('n', 'mean', 'lower', 'upper')}, original_gate_unchanged=True))
            influences.extend(dict(path=path, metric=metric, family_id=fid, effect=float(value),
                path_controls_valid=fid in valid_ids, contribution_to_all_family_mean=float(value)/len(all_ids))
                for fid, value in effects.items())
    return pd.DataFrame(gates), pd.DataFrame(sensitivity), pd.DataFrame(influences)


def subgroup_summary(bundle):
    frame = bundle['frame']
    output = []
    for group in ('stratum', 'physical_outcome', 'baseline_correct', 'style', 'variant'):
        for (value, condition), sub in frame.groupby([group, 'condition']):
            metrics = ['accuracy_change', 'margin_change', 'correct', 'baseline_correct', 'repaired', 'harmed']
            # Avoid using a grouping variable as both a column and index.
            units = sub.groupby(['family_id', 'scene_id', 'variant'])[metrics].mean().reset_index()
            means = units.groupby('family_id')[metrics].mean()
            ci = bootstrap(means.accuracy_change.to_numpy(float), ['subgroup', bundle['model'], group, str(value), condition])
            output.append(dict(model=bundle['model'], group=group, value=str(value), condition=condition,
                families=len(means), scene_mapping_units=len(units),
                baseline_accuracy=float(means.baseline_correct.mean()), accuracy=float(means.correct.mean()),
                accuracy_change=float(means.accuracy_change.mean()), margin_change=float(means.margin_change.mean()),
                accuracy_lower=ci['lower'], accuracy_upper=ci['upper'],
                original_wrong_units=float((1-units.baseline_correct).sum()), original_correct_units=float(units.baseline_correct.sum()),
                repaired_units_seed_average=float(units.repaired.sum()), harmed_units_seed_average=float(units.harmed.sum()),
                interpretation='posthoc_descriptive_family_aggregation_no_new_confirmation'))
    return pd.DataFrame(output)


def summarize_archived_readout(frame, threshold):
    assert not frame.duplicated(['scene_id', 'repeat']).any()
    assert frame.groupby('scene_id').size().eq(3).all()
    assert frame.groupby('scene_id').repeat.apply(lambda x: set(x) == {0, 1, 2}).all()
    assert frame.scene_id.nunique() == 19
    outcome = lambda x: 'left' if x > threshold else 'right' if x < -threshold else 'balanced'
    columns = ['archived_readout', 'current_native_readout', 'current_fp16_pipeline_readout',
               'archived_minus_current', 'storage_roundtrip_change']
    assert np.isfinite(frame[columns].to_numpy(float)).all()
    assert abs(frame.archived_readout-frame.current_native_readout-frame.archived_minus_current).max() < 1e-8
    assert abs(frame.current_fp16_pipeline_readout-frame.current_native_readout-frame.storage_roundtrip_change).max() < 1e-8
    rows = []
    for sid, sub in frame.groupby('scene_id'):
        assert sub.archived_readout.nunique() == 1
        native_gap = abs(sub.archived_readout-sub.current_native_readout)
        storage_shift = abs(sub.storage_roundtrip_change)
        remaining_gap = abs(sub.archived_readout-sub.current_fp16_pipeline_readout)
        old_class = outcome(float(sub.archived_readout.iloc[0]))
        current_classes = sub.current_native_readout.map(outcome)
        rows.append(dict(scene_id=sid, repeats=3, archived_readout=float(sub.archived_readout.iloc[0]),
            current_native_readout_mean=float(sub.current_native_readout.mean()),
            current_fp16_readout_mean=float(sub.current_fp16_pipeline_readout.mean()),
            mean_absolute_archive_gap=float(native_gap.mean()), maximum_absolute_archive_gap=float(native_gap.max()),
            mean_absolute_storage_shift=float(storage_shift.mean()), maximum_absolute_storage_shift=float(storage_shift.max()),
            mean_absolute_residual_after_storage=float(remaining_gap.mean()),
            maximum_absolute_residual_after_storage=float(remaining_gap.max()),
            archived_class=old_class, current_classes='|'.join(sorted(set(current_classes))),
            any_class_changed_from_archive=bool(current_classes.ne(old_class).any()),
            within_current_repeat_class_changed=bool(current_classes.nunique() > 1)))
    by_scene = pd.DataFrame(rows)
    summary = dict(scenes=19, repeat_rows=len(frame),
        mean_absolute_archive_gap=float(by_scene.mean_absolute_archive_gap.mean()),
        maximum_absolute_archive_gap=float(by_scene.maximum_absolute_archive_gap.max()),
        mean_absolute_storage_shift=float(by_scene.mean_absolute_storage_shift.mean()),
        maximum_absolute_storage_shift=float(by_scene.maximum_absolute_storage_shift.max()),
        mean_absolute_residual_after_storage=float(by_scene.mean_absolute_residual_after_storage.mean()),
        class_changed_scenes=int(by_scene.any_class_changed_from_archive.sum()),
        within_current_repeat_class_changed_scenes=int(by_scene.within_current_repeat_class_changed.sum()))
    return by_scene, summary


def load_repeat_result(root, project, model, slug, inputs):
    root = Path(root)
    paths = {'completion': root/'analysis/RUN_COMPLETE.json', 'protocol': root/'protocol.json',
             'identity': root/'repeat_identity.json', 'pairs': root/'analysis/within_runtime_pairs.csv',
             'means': root/'analysis/within_runtime_scene_means.csv',
             'historical': root/'analysis/historical_scene_mean_comparison.csv'}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        return dict(model=model, status='missing_inputs', issues=missing)
    completion = tracked_json(paths['completion'], inputs)
    protocol = tracked_json(paths['protocol'], inputs)
    identity = tracked_json(paths['identity'], inputs)
    assert completion['status'] == 'complete' and completion['model'] == protocol['model'] == model
    assert identity['protocol'] == protocol
    assert completion['development_scenes'] == 19 and completion['mappings_per_scene'] == 6
    assert completion['repeats'] == 3 and completion['forward_count'] == 342 and completion['probe_refit'] is False
    old = Path(project)/'results/cross_model_readout_correction_v2/models'/slug
    for key, name in [('probe', 'frozen_probe.json'), ('old_protocol', 'protocol.json'), ('old_runtime', 'runtime_identity.json')]:
        path = old/name
        assert path.is_file(), f'Missing frozen diagnostic input: {path}'
        inputs[str(path)] = file_hash(path)
        assert identity[key] == inputs[str(path)], f'Frozen readout input changed: {path}'
    old_protocol = read_json(old/'protocol.json')
    threshold = float(old_protocol['balance_threshold'])
    pairs = tracked_csv(paths['pairs'], inputs)
    means = tracked_csv(paths['means'], inputs)
    assert len(pairs) == 342 and pairs.scene_id.nunique() == 19
    assert not pairs.duplicated(['scene_id', 'variant', 'repeat_a', 'repeat_b']).any()
    assert pairs.groupby(['scene_id', 'variant']).size().eq(3).all()
    assert pairs.groupby('scene_id').variant.nunique().eq(6).all()
    for _, sub in pairs.groupby(['scene_id', 'variant']):
        assert set(zip(sub.repeat_a, sub.repeat_b)) == {(0, 1), (0, 2), (1, 2)}
    assert len(means) == 57 and means.scene_id.nunique() == 19
    assert not means.duplicated(['scene_id', 'repeat_a', 'repeat_b']).any()
    for key in ('readout_over_tolerance', 'decoded_outcome_changed', 'answer_changed'):
        pairs[key] = pairs[key].map(bool_value)
    means['scene_mean_class_changed'] = means.scene_mean_class_changed.map(bool_value)
    for result_key, column in [('readout_over_tolerance', 'readout_over_tolerance'),
                               ('any_answer_changed', 'answer_changed'), ('any_readout_class_changed', 'decoded_outcome_changed')]:
        assert bool_value(completion[result_key]) == bool(pairs[column].any())
    assert bool_value(completion['any_scene_mean_class_changed']) == bool(means.scene_mean_class_changed.any())
    historical = tracked_csv(paths['historical'], inputs)
    by_scene, summary = pd.DataFrame(), {}
    if not historical.empty:
        by_scene, summary = summarize_archived_readout(historical, threshold)
        assert set(by_scene.scene_id) == set(pairs.scene_id)
    config_rows = []
    archive_inputs = root/'archive_inputs.json'
    if archive_inputs.is_file():
        archived_identity = tracked_json(archive_inputs, inputs)['identity']
        current = identity['runtime']
        # Show available configuration fields; differences are clues, not a causal attribution.
        for source_name, value in [('historical_capture', archived_identity), ('current_repeat', current)]:
            for key, v in value.items():
                if any(token in key.lower() for token in ('batch', 'dtype', 'layer', 'version', 'position', 'pool', 'attention', 'cuda', 'torch', 'transformer', 'gpu')):
                    if key != 'toolkit':
                        config_rows.append(dict(model=model, source=source_name, field=key, value=json.dumps(v, ensure_ascii=False)))
    repeat_summary = dict(model=model, scenes=19, forwards=342,
        maximum_hidden_change=float(pairs.hidden_max_change.max()),
        maximum_readout_change=float(pairs.readout_absolute_change.max()),
        maximum_centered_score_change=float(pairs.centered_score_max_change.max()),
        any_answer_changed=bool(pairs.answer_changed.any()),
        any_readout_class_changed=bool(pairs.decoded_outcome_changed.any()))
    return dict(model=model, status='validated_complete', issues=[], completion=completion,
        repeat_summary=repeat_summary, by_scene=by_scene, archive_summary=summary,
        historical=historical, config_rows=config_rows)


def csv_markdown(frame, columns=None, limit=40):
    if frame.empty:
        return '无记录。'
    shown = frame if columns is None else frame[columns]
    shown = shown.head(limit)
    escape = lambda x: str(x).replace('|', '/').replace('\n', ' ')
    lines = ['| '+' | '.join(map(escape, shown.columns))+' |', '| '+' | '.join(['---']*len(shown.columns))+' |']
    for row in shown.itertuples(index=False, name=None):
        lines.append('| '+' | '.join(escape(f'{v:.6g}' if isinstance(v, float) else v) for v in row)+' |')
    if len(frame) > limit:
        lines += ['', f'此处显示前 {limit} 行，完整 {len(frame)} 行保存在 CSV。']
    return '\n'.join(lines)
