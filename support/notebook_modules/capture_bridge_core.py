"""Pure pairing/scheduling and readout comparisons for a bounded capture bridge.

This reconstructs the checked local discovery template in the current runtime.
It never claims to restore an unlogged historical environment or batch schedule.
"""


def capture_key(record):
    return str(record['scene_id']), str(record['variant'])


def candidate_batch_schedule(ordered_keys, wanted, batch_size):
    assert isinstance(batch_size, int) and 1 < batch_size <= 8
    assert len(ordered_keys) == len(set(ordered_keys)) and set(wanted) <= set(ordered_keys)
    batches = [ordered_keys[i:i+batch_size] for i in range(0, len(ordered_keys), batch_size)]
    selected = [b for b in batches if set(b) & set(wanted)]
    covered = [k for b in selected for k in b if k in wanted]
    assert len(covered) == len(wanted) and set(covered) == set(wanted)
    return selected


def normalize_sha(value):
    return str(value).removeprefix('sha256:')


def exact_input_comparison(left, right):
    """Fingerprints include names, dtypes, shapes and full tensor bytes."""
    keys = sorted(set(left) | set(right))
    return [dict(field=k, equal=k in left and k in right and left[k] == right[k],
                 legacy=left.get(k), current=right.get(k)) for k in keys]


def aggregate_capture_scenes(keys, arrays, metadata, old_means, threshold, decode_fn):
    assert len(keys) == len(set(keys)) and len(metadata) == len(keys)
    assert all(len(a) == len(keys) for a in arrays.values())
    index = pd.DataFrame([dict(scene_id=s, variant=v, index=i) for i, (s, v) in enumerate(keys)])
    assert index.groupby('scene_id').size().eq(6).all()
    assert index.groupby('scene_id').variant.nunique().eq(6).all()
    out = []
    classify = lambda x: 'left' if x > threshold else 'right' if x < -threshold else 'balanced'
    for sid, group in index.groupby('scene_id', sort=True):
        inds = group['index'].to_numpy(int)
        row = dict(scene_id=sid, mappings=6)
        if sid in old_means:
            row['historical_readout'] = decode_fn(old_means[sid])
            row['historical_class'] = classify(row['historical_readout'])
        for name, array in arrays.items():
            # All variants are first represented in float32. Legacy storage then
            # rounds the six-map mean to FP16, exactly as the local template.
            mean = np.asarray(array[inds], np.float32).mean(0)
            if name.endswith('_fp16'):
                mean = mean.astype(np.float16).astype(np.float32)
            value = decode_fn(mean)
            row[name+'_readout'] = value
            row[name+'_class'] = classify(value)
            if sid in old_means:
                row[name+'_archive_gap'] = value-row['historical_readout']
                row[name+'_archive_class_changed'] = classify(value) != row['historical_class']
        out.append(row)
    return pd.DataFrame(out)


def scene_gap_summary(scenes, route_names):
    rows = []
    for name in route_names:
        key = name+'_archive_gap'
        if key not in scenes:
            continue
        gap = scenes[key].dropna()
        rows.append(dict(route=name, scenes=len(gap), mean_absolute_archive_gap=float(gap.abs().mean()),
            maximum_absolute_archive_gap=float(gap.abs().max()), mean_signed_archive_gap=float(gap.mean()),
            class_changed_scenes=int(scenes.loc[gap.index, name+'_archive_class_changed'].sum())))
    return pd.DataFrame(rows)


def bridge_classification(preparation_equal, storage_parity, original_mean_consistent, summaries, single_batch_gap):
    notes = []
    if not preparation_equal:
        notes.append('旧模板与当前单张流程的实际输入或候选计划不同；先定位具体输入字段，批量比较不能解释全部差异。')
    if not storage_parity:
        notes.append('同一次前向的旧式 FP16 捕获与原生状态转存不一致；检查捕获位置或输出处理。')
    if original_mean_consistent is False:
        notes.append('历史逐映射分片重建出的均值与历史 scene_mean 文件不一致；历史缓存内部存在待核对差异。')
    elif original_mean_consistent is None:
        notes.append('缺少完整历史逐映射分片；无法核对历史均值的来源或原批次成员。')
    if single_batch_gap is not None:
        notes.append('同一当前环境中，所选批处理上下文与单张处理的差异已测量；它同时包含批处理、填充及可能的图像处理差异。')
    names = set(summaries.route) if not summaries.empty else set()
    if {'legacy_single_fp16','legacy_batch_fp16'} <= names:
        a = summaries.set_index('route').loc['legacy_single_fp16', 'mean_absolute_archive_gap']
        b = summaries.set_index('route').loc['legacy_batch_fp16', 'mean_absolute_archive_gap']
        notes.append('批处理路线与历史均值更接近。' if b < a else '本次候选批处理路线没有降低平均历史读出差异。')
    notes.append('较接近历史缓存不等于已恢复历史流程；未记录的批次组成、软件或硬件仍不能自动归因。此诊断不更新任何科学确认状态。')
    return notes
