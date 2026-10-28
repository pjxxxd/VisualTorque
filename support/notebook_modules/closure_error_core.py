"""Saved-answer diagnostics. Embedded after the original CPU validation helpers."""


def record_semantics(record):
    if 'label_to_semantic' in record:
        mapping = {str(k): str(v) for k, v in record['label_to_semantic'].items()}
    else:
        labels, values = record['candidate_labels'], record['candidate_semantics']
        assert len(labels) == len(values) == 3
        mapping = dict(zip(map(str, labels), map(str, values)))
    assert set(mapping) == {'A', 'B', 'C'} and set(mapping.values()) == {'left', 'right', 'balanced'}
    scores = record['candidate_logprobs']
    labels = list(record.get('candidate_labels', mapping))
    scores = {k: float(scores[k]) for k in labels} if isinstance(scores, dict) else dict(zip(labels, map(float, scores)))
    assert set(scores) == set(mapping) and np.isfinite(list(scores.values())).all()
    pred_label = str(record.get('predicted_label', max(labels, key=scores.get)))
    assert pred_label in mapping and scores[pred_label] >= max(scores.values()) - 1e-12
    assert mapping[pred_label] == str(record['predicted_semantic'])
    if 'gold_label' in record:
        assert mapping[str(record['gold_label'])] == str(record['gold_semantic'])
    return mapping, pred_label


def swap_side(value):
    return {'left': 'right', 'right': 'left', 'balanced': 'balanced'}[value]


def stack_signature(stacks, mirror=False):
    return sorted((swap_side(s['side']) if mirror else s['side'], int(s['units']), int(s['distance'])) for s in stacks)


def behavior_rows(bundle):
    verify_record_metrics(bundle)
    lookup = {}
    for family in bundle['families']:
        assert len(family['scenes']) == 4
        for style in ('standard', 'alternate'):
            sub = {bool_value(s['mirror']): s for s in family['scenes'] if s['style'] == style}
            assert set(sub) == {False, True}
            assert stack_signature(sub[False]['stacks'], True) == stack_signature(sub[True]['stacks'])
        for scene in family['scenes']:
            assert scene['scene_id'] not in lookup
            features = physics(scene['stacks'])
            assert features == scene['features'], 'Saved geometry/features differ'
            lookup[scene['scene_id']] = (family, scene, features)
    out = []
    for row in bundle['records']:
        family, scene, features = lookup[row['scene_id']]
        record, clean = row['record'], row['clean_record']
        mapping, pred_label = record_semantics(record)
        clean_mapping, baseline_label = record_semantics(clean)
        assert mapping == clean_mapping
        assert family['family_id'] == row['family_id'] and family['stratum'] == row['stratum']
        assert bool_value(row['mirror']) == bool_value(scene['mirror']) and row['style'] == scene['style']
        pred, gold, base = str(record['predicted_semantic']), features['outcome'], str(clean['predicted_semantic'])
        assert gold == record['gold_semantic'] == clean['gold_semantic']
        assert int(record['net_torque']) == int(clean['net_torque']) == features['net_torque']
        w, d = features['strategy_predictions']['weight_only'], features['strategy_predictions']['distance_only']
        mistake = ('correct' if pred == gold else 'both_heuristics' if pred == w == d else
                   'weight_only_prediction' if pred == w else 'distance_only_prediction' if pred == d else
                   'other_balanced' if pred == 'balanced' else 'other_direction')
        out.append(dict(model=bundle['model'], family_id=row['family_id'], scene_id=row['scene_id'],
            stratum=row['stratum'], style=row['style'], mirror=bool_value(row['mirror']), variant=row['variant'],
            condition=row['condition'], random_seed=int(row['random_seed']), valid=bool_value(row['valid']),
            gold=gold, predicted=pred, baseline_predicted=base, predicted_label=pred_label, baseline_label=baseline_label,
            label_left=next(k for k, v in mapping.items() if v == 'left'),
            label_right=next(k for k, v in mapping.items() if v == 'right'),
            label_balanced=next(k for k, v in mapping.items() if v == 'balanced'),
            net_torque=features['net_torque'], left_units=features['left_units'], right_units=features['right_units'],
            left_torque=features['left_torque'], right_torque=features['right_torque'],
            weight_prediction=w, distance_prediction=d, correct=pred == gold, baseline_correct=base == gold,
            repaired=base != gold and pred == gold, harmed=base == gold and pred != gold,
            weight_disagrees_physics=w != gold, directional_weight_conflict=w != gold and w != 'balanced',
            follows_weight=pred == w, follows_distance=pred == d, mistake_category=mistake,
            margin_change=float(row['margin_change'])))
    return pd.DataFrame(out)


def family_ratio(frame, numerator, denominator, key, unit_columns=None, replicates=5000):
    """Seed means first; ratios of family totals with a family-cluster bootstrap.

    Report the pooled ratio and equally weighted within-family ratios separately.
    Undefined denominators stay missing; no confirmatory p values are produced.
    """
    unit_columns = unit_columns or ['family_id', 'scene_id', 'variant']
    x = frame[unit_columns].copy()
    x['num'] = np.asarray(numerator, float)
    x['den'] = np.asarray(denominator, float)
    assert np.isfinite(x[['num', 'den']]).all().all()
    assert (x.num >= 0).all() and (x.den >= x.num).all()
    units = x.groupby(unit_columns, sort=True)[['num', 'den']].mean()
    families = units.groupby(level='family_id').sum()
    total = float(families.den.sum())
    if total == 0:
        return dict(families=len(families), eligible_families=0, numerator_seed_average=0., denominator_seed_average=0.,
                    pooled_fraction=None, lower=None, upper=None, family_mean_fraction=None)
    rng = np.random.default_rng(int(digest(['error_structure', key])[:16], 16))
    counts = families[['num', 'den']].to_numpy(float)
    estimates = []
    for start in range(0, replicates, 250):
        sampled = counts[rng.integers(0, len(counts), (min(250, replicates-start), len(counts)))].sum(1)
        estimates.extend((sampled[:, 0][sampled[:, 1] > 0] / sampled[:, 1][sampled[:, 1] > 0]).tolist())
    return dict(families=len(families), eligible_families=int(families.den.gt(0).sum()),
        numerator_seed_average=float(families.num.sum()), denominator_seed_average=total,
        pooled_fraction=float(families.num.sum()/total), lower=float(np.quantile(estimates, .025)),
        upper=float(np.quantile(estimates, .975)),
        family_mean_fraction=float((families.num/families.den.replace(0, np.nan)).mean()))


def error_rate_tables(frame):
    rows, confusion = [], []
    for condition, block in frame.groupby('condition', sort=True):
        groups = [('all', 'all', block)]
        groups += [(column, str(value), sub) for column in ('stratum', 'gold', 'style', 'variant')
                   for value, sub in block.groupby(column, sort=True)]
        for group, value, sub in groups:
            all_rows = np.ones(len(sub), bool)
            specs = [('accuracy', sub.correct, all_rows), ('repair_of_original_errors', sub.repaired, ~sub.baseline_correct),
                ('harm_of_original_correct', sub.harmed, sub.baseline_correct),
                ('weight_choice_when_weight_disagrees', sub.follows_weight & sub.weight_disagrees_physics, sub.weight_disagrees_physics),
                ('weight_choice_on_directional_conflicts', sub.follows_weight & sub.directional_weight_conflict, sub.directional_weight_conflict),
                ('weight_agreement_among_errors', sub.follows_weight & ~sub.correct, ~sub.correct)]
            specs += [('predicted_'+s, sub.predicted.eq(s), all_rows) for s in ('left', 'right', 'balanced')]
            for metric, num, den in specs:
                rows.append(dict(model=sub.model.iloc[0], condition=condition, group=group, value=value, metric=metric,
                    **family_ratio(sub, num, den, [sub.model.iloc[0], condition, group, value, metric]),
                    interpretation='posthoc_descriptive_no_new_confirmation'))
        units = block.groupby(['family_id','scene_id','variant','gold','predicted','predicted_label','mistake_category']).size().rename('draws').reset_index()
        seeds = block.groupby(['family_id','scene_id','variant']).size().rename('seeds').reset_index()
        units = units.merge(seeds, on=['family_id','scene_id','variant'], validate='many_to_one')
        units['seed_average_units'] = units.draws/units.seeds
        for (gold, predicted, letter, category), sub in units.groupby(['gold','predicted','predicted_label','mistake_category']):
            confusion.append(dict(model=block.model.iloc[0], condition=condition, gold=gold, predicted=predicted,
                predicted_label=letter, mistake_category=category, seed_average_units=float(sub.seed_average_units.sum())))
    return pd.DataFrame(rows), pd.DataFrame(confusion)


def paired_behavior(frame, pairing):
    """Match the same family/render/mapping; never pair records by row order."""
    assert pairing in ('mirror', 'mapping')
    varying = 'mirror' if pairing == 'mirror' else 'variant'
    keep = ['model','family_id','style','condition','random_seed'] + (['variant'] if pairing == 'mirror' else ['scene_id','mirror'])
    rows = []
    for key, sub in frame.groupby(keep, sort=True):
        assert len(sub) == 2, (pairing, key, len(sub))
        ordered = sub.set_index(varying)
        assert set(ordered.index) == ({False, True} if pairing == 'mirror' else {'forward','reverse'})
        a, b = (ordered.loc[False], ordered.loc[True]) if pairing == 'mirror' else (ordered.loc['forward'], ordered.loc['reverse'])
        if pairing == 'mirror':
            assert b.gold == swap_side(a.gold)
            assert all(a[c] == b[c] for c in ('label_left','label_right','label_balanced'))
        else:
            assert a.gold == b.gold and a.label_balanced == b.label_balanced
            assert a.label_left == b.label_right and a.label_right == b.label_left
        rows.append(dict(zip(keep, key)) | dict(pairing=pairing, pair_id=digest([pairing, list(key)])[:24],
            scene_a=str(a.get('scene_id', key[-2] if pairing == 'mapping' else '')),
            gold_a=a.gold, gold_b=b.gold, prediction_a=a.predicted, prediction_b=b.predicted,
            label_a=a.predicted_label, label_b=b.predicted_label,
            semantic_consistent=b.predicted == (swap_side(a.predicted) if pairing == 'mirror' else a.predicted),
            both_correct=bool(a.correct and b.correct), both_wrong=bool(not a.correct and not b.correct),
            same_direction=bool(a.predicted == b.predicted and a.predicted != 'balanced'),
            same_letter=bool(a.predicted_label == b.predicted_label),
            same_letter_different_semantics=bool(a.predicted_label == b.predicted_label and a.predicted != b.predicted),
            original_error_count=int(not a.baseline_correct)+int(not b.baseline_correct),
            repaired_count=int(a.repaired)+int(b.repaired), harmed_count=int(a.harmed)+int(b.harmed)))
    return pd.DataFrame(rows)


def pair_summaries(pairs):
    rows = []
    for (model, pairing, condition), block in pairs.groupby(['model','pairing','condition'], sort=True):
        units = ['family_id','style','variant'] if pairing == 'mirror' else ['family_id','scene_id']
        for metric in ('semantic_consistent','both_correct','both_wrong','same_direction','same_letter','same_letter_different_semantics'):
            rows.append(dict(model=model, pairing=pairing, condition=condition, metric=metric,
                **family_ratio(block, block[metric], np.ones(len(block)), [model,pairing,condition,metric], units)))
    return pd.DataFrame(rows)


def paired_consistency_changes(pairs):
    rows = []
    for (model, pairing), block in pairs.groupby(['model','pairing'], sort=True):
        units = ['family_id','style','variant'] if pairing == 'mirror' else ['family_id','scene_id']
        for metric in ('semantic_consistent','both_correct'):
            means = block.groupby(units+['condition'])[metric].mean().reset_index()
            zero = means[means.condition.eq('zero')].drop(columns='condition').rename(columns={metric:'baseline'})
            for condition, sub in means[means.condition.ne('zero')].groupby('condition'):
                paired = sub.merge(zero, on=units, validate='one_to_one')
                assert len(paired) == len(zero)
                paired['change'] = paired[metric]-paired.baseline
                family = paired.groupby('family_id').change.mean().to_numpy(float)
                ci = bootstrap(family, ['paired_consistency',model,pairing,condition,metric], replicates=5000)
                rows.append(dict(model=model,pairing=pairing,condition=condition,metric=metric,
                    **{k:ci[k] for k in ('n','mean','lower','upper')},interpretation='paired_posthoc_difference_not_new_confirmation'))
    return pd.DataFrame(rows)
