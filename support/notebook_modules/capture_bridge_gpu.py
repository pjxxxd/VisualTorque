"""No-edit model execution; embedded after Torch/model setup.

Old-style preparation and FP16 hook follow the checked discovery template;
current-style preparation/native hook follow the checked E1 notebook.
"""

from collections.abc import Mapping


def tensor_signature(value):
    if torch.is_tensor(value):
        v = value.detach().contiguous().cpu()
        raw = v.reshape(-1).view(torch.uint8).numpy().tobytes()
        return dict(dtype=str(v.dtype), shape=list(v.shape), sha256=hashlib.sha256(raw).hexdigest())
    if isinstance(value, (tuple, list)):
        return [tensor_signature(v) for v in value]
    # HF BatchFeature/BatchEncoding are mapping wrappers, not dict subclasses.
    # Inspect every contained value without changing the prepared model inputs.
    if isinstance(value, Mapping):
        return {str(k): tensor_signature(v) for k, v in sorted(value.items())}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError('Untracked prepared-input value: '+type(value).__name__)


def legacy_prepare(examples):
    assert examples and all(e.labels == examples[0].labels for e in examples)
    plan = adapter._candidate_plan(examples[0])
    assert plan is not None
    prefix, tokens = plan
    assert all(adapter._candidate_plan(e) == plan for e in examples)
    conversations = [e.messages() for e in examples]
    prepared = adapter._append_text_prefix(
        adapter._to_device(adapter._template(conversations, add_generation_prompt=True)), prefix)
    return prepared, dict(prefix=str(prefix), token_by_label={str(k): int(v) for k, v in tokens.items()})


def current_prepare(example):
    plan = adapter._candidate_plan(example)
    assert plan is not None
    prefix, tokens = plan
    prepared = adapter._append_text_prefix(adapter._to_device(adapter._template(
        [example.messages()], add_generation_prompt=True)), prefix)
    return prepared, dict(prefix=str(prefix), token_by_label={str(k): int(v) for k, v in tokens.items()})


def prepared_rows(inputs, examples):
    """Record unpadded text and model-supported per-image fingerprints.

    The full tensor dictionary is always fingerprinted separately. A missing
    per-image partition is explicitly unknown, never assumed equivalent.
    """
    mask = inputs['attention_mask'].detach().cpu()
    ids = inputs['input_ids'].detach().cpu()
    assert ids.ndim == mask.ndim == 2 and ids.shape == mask.shape and len(ids) == len(examples)
    pixels = inputs.get('pixel_values')
    grid = inputs.get('image_grid_thw')
    pixel_rows = [None for _ in examples]
    method = 'per_image_partition_unavailable'
    if torch.is_tensor(pixels) and torch.is_tensor(grid) and len(grid) == len(examples):
        sizes = grid.detach().cpu().to(torch.int64).prod(-1).tolist()
        if sum(sizes) == pixels.shape[0]:
            start = 0
            for i, size in enumerate(sizes):
                pixel_rows[i] = tensor_signature(pixels[start:start+size])
                start += size
            method = 'image_grid_patch_partition'
    elif torch.is_tensor(pixels) and len(pixels) == len(examples):
        pixel_rows = [tensor_signature(pixels[i:i+1]) for i in range(len(examples))]
        method = 'leading_batch_dimension'
    out = []
    for i, ex in enumerate(examples):
        active = torch.nonzero(mask[i], as_tuple=False).flatten()
        assert len(active)
        out.append(dict(scene_id=str(ex.scene_id), variant=str(ex.variant), example_id=str(ex.example_id),
            unpadded_input_ids=ids[i][mask[i].bool()].tolist(), sequence_length=int(ids.shape[1]),
            active_token_count=len(active), last_active_index=int(active[-1]),
            selected_index=int(ids.shape[1]-1), selected_token_active=bool(mask[i,-1]),
            selected_token_id=int(ids[i,-1]), image_fingerprint=pixel_rows[i], image_partition=method))
    return out


def execute_bridge_forward(inputs, examples, plan):
    global MODEL_FORWARD_ATTEMPTS, MODEL_EXAMPLES_ATTEMPTED
    # Separate observers preserve both original storage/capture conventions.
    captured = {}
    handles = []
    def legacy_hook(_module, _inputs, output):
        hidden = output_hidden_tensor(output)
        captured['fp16'] = hidden[:, -1, :].detach().to(device='cpu', dtype=torch.float16).float().numpy().copy()
    def native_hook(_module, _inputs, output):
        hidden = output_hidden(output)
        assert hidden.dtype == torch.bfloat16
        captured['native'] = hidden[:, -1].detach().float().cpu().numpy().copy()
    handles.append(DECODER_LAYERS[SELECTED_LAYER].register_forward_hook(legacy_hook))
    handles.append(DECODER_LAYERS[SELECTED_LAYER].register_forward_hook(native_hook))
    try:
        with torch.inference_mode():
            MODEL_FORWARD_ATTEMPTS += 1
            MODEL_EXAMPLES_ATTEMPTED += len(examples)
            result = adapter.model(**inputs, use_cache=False)
            scores = torch.log_softmax(result.logits[:, -1, :].float(), dim=-1)
            scored = [{label: float(scores[i, plan['token_by_label'][label]].cpu()) for label in ex.labels}
                      for i, ex in enumerate(examples)]
    finally:
        for handle in handles:
            handle.remove()
    assert set(captured) == {'native','fp16'}
    assert captured['native'].shape == captured['fp16'].shape == (len(examples), len(PROBE_MEAN))
    assert all(np.isfinite(x).all() for x in captured.values())
    records = []
    for ex, score in zip(examples, scored):
        label = max(ex.labels, key=score.get)
        records.append(dict(scene_id=str(ex.scene_id), variant=str(ex.variant), example_id=str(ex.example_id),
            gold=str(ex.gold_semantic), predicted=str(ex.label_to_semantic[label]), predicted_label=str(label),
            label_to_semantic={str(k): str(v) for k, v in ex.label_to_semantic.items()}, candidate_logprobs=score))
    return captured, records


def cached_bridge_capture(key, compute):
    folder = CAPTURE_CACHE/digest(key)[:24]
    folder.mkdir(parents=True, exist_ok=True)
    metadata_path, arrays_path = folder/'capture.json', folder/'states.npz'
    if metadata_path.is_file():
        metadata = read_json(metadata_path)
        assert metadata['identity'] == CAPTURE_IDENTITY and metadata['key'] == key
        assert arrays_path.is_file() and file_hash(arrays_path) == metadata['arrays_sha256']
        with np.load(arrays_path, allow_pickle=False) as source:
            arrays = {k: source[k].copy() for k in source.files}
        return metadata['result'], arrays, 0
    result, arrays, forwards = compute()
    temporary = arrays_path.with_suffix('.npz.tmp')
    with temporary.open('wb') as f:
        np.savez_compressed(f, **arrays)
    temporary.replace(arrays_path)
    save_json(metadata_path, dict(identity=CAPTURE_IDENTITY, key=key, arrays_sha256=file_hash(arrays_path), result=result))
    return result, arrays, forwards


def capture_single_bridge(example):
    old_inputs, old_plan = legacy_prepare([example])
    new_inputs, new_plan = current_prepare(example)
    old_fingerprint, new_fingerprint = tensor_signature(old_inputs), tensor_signature(new_inputs)
    differences = exact_input_comparison(old_fingerprint, new_fingerprint)
    equal = all(d['equal'] for d in differences) and old_plan == new_plan
    old_rows, new_rows = prepared_rows(old_inputs, [example]), prepared_rows(new_inputs, [example])
    assert new_rows[0]['selected_token_active'], 'Current single terminal token is padding.'
    current, current_records = execute_bridge_forward(new_inputs, [example], new_plan)
    forwards = 1
    if equal:
        legacy, legacy_records = current, current_records
    else:
        legacy, legacy_records = execute_bridge_forward(old_inputs, [example], old_plan)
        forwards += 1
    record = dict(key=list(capture_key(current_records[0])), shared_forward=equal,
        preparation_equal=equal, prepared_field_comparison=differences, legacy_plan=old_plan, current_plan=new_plan,
        legacy_prepared_rows=old_rows, current_prepared_rows=new_rows,
        legacy_record=legacy_records[0], current_record=current_records[0],
        native_to_fp16_observer_gap=float(np.max(abs(current['native'].astype('float16').astype('float32')-current['fp16']))))
    arrays = dict(current_single_native=current['native'], current_single_fp16=current['fp16'],
                  legacy_single_native=legacy['native'], legacy_single_fp16=legacy['fp16'])
    return record, arrays, forwards


def capture_batch_bridge(examples):
    inputs, plan = legacy_prepare(examples)
    rows = prepared_rows(inputs, examples)
    captures, records = execute_bridge_forward(inputs, examples, plan)
    result = dict(keys=[list(capture_key(r)) for r in records], plan=plan,
        full_inputs_fingerprint=tensor_signature(inputs), prepared_rows=rows, records=records,
        native_to_fp16_observer_gap=float(np.max(abs(captures['native'].astype('float16').astype('float32')-captures['fp16']))))
    # A padded terminal token is retained as a diagnostic finding rather than
    # silently repaired: this route reproduces the template's [:, -1] selection.
    return result, dict(legacy_batch_native=captures['native'],legacy_batch_fp16=captures['fp16']), 1
