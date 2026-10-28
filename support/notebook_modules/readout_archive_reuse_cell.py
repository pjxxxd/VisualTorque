# Replace the ENTIRE failing "12/15" cell in the original notebook with this code.
# Keep the current Colab runtime; this cell uses the model and frozen probe already loaded.
# This is a notebook cell, not a standalone Python program.
cell_stage('Auditing archived states; recapturing test states when reuse is not supported')
TEST_REUSE_IMPLEMENTATION_SHA = '__ARCHIVE_REUSE_CELL_SHA__'
required_globals = ('adapter', 'forward_one', 'decode', 'SELECTED_LAYER', 'test_by_scene',
                    'input_identities', 'INPUT_SHA', 'PROBE_SHA', 'NUMERIC_LOCK',
                    'PROTOCOL_SHA', 'RUNTIME_SHA', 'cached_scene')
missing = [name for name in required_globals if name not in globals()]
assert not missing, f'Run the earlier notebook cells first. Missing: {missing}'

def archived_parity_row(example, current_record, current_hidden, archived_record, archived_hidden):
    current = np.asarray(current_hidden, dtype=np.float64)
    archived = np.asarray(archived_hidden, dtype=np.float64)
    assert current.ndim == 1 and np.isfinite(current).all(), 'Current model produced an invalid hidden state'
    current_scores = np.asarray([current_record['candidate_logprobs'][k] for k in example.labels], dtype=float)
    archived_scores = np.asarray([archived_record['candidate_logprobs'][k] for k in example.labels], dtype=float)
    assert np.isfinite(current_scores).all(), 'Current model produced invalid candidate scores'
    current_value = decode(current)
    assert np.isfinite(current_value), 'Current readout is non-finite'
    shape_ok = archived.shape == current.shape
    old_finite = bool(shape_ok and np.isfinite(archived).all() and np.isfinite(archived_scores).all())
    archived_value = decode(archived) if old_finite else None
    old_finite = old_finite and bool(np.isfinite(archived_value))
    # Use the same storage representation for diagnosis; do not assume FP16 caused a discrepancy.
    with np.errstate(over='ignore', invalid='ignore'):
        stored_current = current.astype(np.float16).astype(np.float32)
    cast_value = decode(stored_current) if np.isfinite(stored_current).all() else None
    gap = abs(current_value-archived_value) if old_finite else None
    score_gap = float(np.max(np.abs((current_scores-current_scores.mean())-
                                    (archived_scores-archived_scores.mean())))) if old_finite else None
    return dict(example_id=example.example_id, scene_id=str(example.scene_id), variant=example.variant,
        current_readout=float(current_value), archived_readout=float(archived_value) if old_finite else None,
        readout_abs_difference=float(gap) if gap is not None else None,
        centered_score_max_difference=score_gap,
        current_fp16_roundtrip_readout=float(cast_value) if cast_value is not None and np.isfinite(cast_value) else None,
        hidden_relative_l2=float(np.linalg.norm(current-archived)/max(np.linalg.norm(current),1e-12)) if old_finite else None,
        hidden_shape_matches=bool(shape_ok), archived_values_finite=bool(old_finite),
        parity_passed=bool(old_finite and gap <= .15 and score_gap <= .15))

def archive_reuse_decision(expected_ids, available_ids, parity_rows):
    complete = set(available_ids) == set(expected_ids) and bool(expected_ids)
    passed = bool(parity_rows) and all(r['parity_passed'] for r in parity_rows)
    if not complete:
        return 'fresh_capture', 'archive_missing_or_incomplete'
    if not passed:
        return 'fresh_capture', 'archive_numerical_parity_not_established'
    return 'verified_archive', 'complete_archive_and_audit_passed'

legacy_test = {}
legacy_test_provenance = None
legacy_config = CONFIRMATORY_ROOT/'capture_config.json'
legacy_shards = sorted(CONFIRMATORY_ROOT.glob('shard_*.pt'))
config = None
if legacy_config.is_file() and legacy_shards:
    config = read_json(legacy_config)['identity']
    # Identity/corruption errors remain errors. Only numerical non-equivalence changes the source.
    assert config['model_id'] == MODEL_ID and config['resolved_revision'] == MODEL_REVISION
    assert int(config['selected_layer_index']) == SELECTED_LAYER and float(config['frozen_ridge_alpha']) == FROZEN_ALPHA
    assert config['dtype'] == MODEL_DTYPE
    assert config['dataset_fingerprint'] == runtime_identity['dataset']
    assert config['dataset_scenes_sha256'] == sha256_file(DATASET_DIR/'scenes.jsonl')
    assert config['dataset_splits_sha256'] == sha256_file(DATASET_DIR/'splits.json')
    assert config['discovery_selection_sha256'] == sha256_file(discovery_selection_path)
    for path in tqdm(legacy_shards, desc='Reading original test activations'):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        assert int(payload['selected_layer_index']) == SELECTED_LAYER
        assert len(payload['records']) == len(payload['activations']), f'Misaligned archive: {path}'
        for record, hidden in zip(payload['records'], payload['activations']):
            eid = record['example_id']
            assert eid in input_identities and eid not in legacy_test
            expected = input_identities[eid]
            for key in ('scene_id','variant','split','gold_semantic','label_to_semantic'):
                assert record[key] == expected[key], (eid,key)
            assert int(record['net_torque']) == expected['net_torque']
            legacy_test[eid] = (record, hidden.float().numpy())
    legacy_test_provenance = {str(p):sha256_file(p) for p in [legacy_config,*legacy_shards]}

# Freeze reuse once for these exact inputs/runtime/probe. The fix has its own code identity.
reuse_request = dict(schema='readout-test-reuse-2.0', implementation=TEST_REUSE_IMPLEMENTATION_SHA,
    protocol=PROTOCOL_SHA, runtime=RUNTIME_SHA, inputs=INPUT_SHA, probe=PROBE_SHA,
    numerical_lock=NUMERIC_LOCK, files=legacy_test_provenance,
    readout_parity_tolerance=.15, centered_score_parity_tolerance=.15,
    archive_policy='complete archive only; any failed audit means all 1007 scenes captured in current runtime')
reuse_directory = OUTPUT_ROOT/'test_reuse_v2'/identity_hash(reuse_request)
decision_path = reuse_directory/'decision.json'
if decision_path.is_file():
    reuse_payload = read_json(decision_path)
    assert reuse_payload['request'] == canonical(reuse_request), 'Reuse identity mismatch'
    parity_rows = reuse_payload['parity_rows']
else:
    parity_rows = []
    if legacy_test:
        # Same one-scene deterministic audit as before; no selection on correctness/effect size.
        audit_scene = min(test_by_scene, key=lambda s:stable_key('archive-parity:'+s))
        for ex in sorted(test_by_scene[audit_scene], key=lambda e:e.variant):
            if ex.example_id not in legacy_test:
                continue
            now, captured = forward_one(ex,(SELECTED_LAYER,))
            old, old_hidden = legacy_test[ex.example_id]
            parity_rows.append(archived_parity_row(ex,now,captured[SELECTED_LAYER],old,old_hidden))
    mode, reason = archive_reuse_decision(input_identities,legacy_test,parity_rows)
    reuse_payload = dict(request=canonical(reuse_request),mode=mode,reason=reason,
        parity_rows=parity_rows,expected_examples=len(input_identities),archived_examples=len(legacy_test))
    # Persist failure details before any long capture, including the original error's two readouts.
    write_json(decision_path,reuse_payload)
    if parity_rows:
        pd.DataFrame(parity_rows).to_csv(reuse_directory/'parity_diagnostics.csv',index=False)
mode, reason = archive_reuse_decision(input_identities,legacy_test,parity_rows)
assert (mode,reason) == (reuse_payload['mode'],reuse_payload['reason'])
TEST_REUSE_SHA = identity_hash(reuse_payload)
EVALUATION_STAGE = 'test_scene_shards_reuse_v2/' + TEST_REUSE_SHA
write_json(ANALYSIS_DIR/'test_capture_source.json',dict(mode=mode,reason=reason,
    decision_path=str(decision_path),reuse_sha=TEST_REUSE_SHA,probe_sha=PROBE_SHA,
    expected_scenes=len(test_by_scene),audit=parity_rows))
if parity_rows:
    display(pd.DataFrame(parity_rows)[['variant','current_readout','archived_readout',
        'readout_abs_difference','centered_score_max_difference','parity_passed']])
if mode == 'fresh_capture':
    legacy_test.clear()
    print('旧测试状态未获准复用：',reason)
    print('将使用当前模型与已冻结的读出器重新采集 1,007 个场景的六种映射，并按场景保存。')
else:
    print('旧测试状态通过核对，复用全部六映射归档。')
print('读出器、层号、0.05 修准标准均保持不变。诊断文件：',decision_path)

evaluation_rows = []
for sid in tqdm(sorted(test_by_scene),desc='1,007-scene readout/answer evaluation',unit='scene'):
    examples = sorted(test_by_scene[sid],key=lambda e:e.variant)
    def evaluate_scene(examples=examples):
        states,records = [],[]
        for e in examples:
            if mode == 'verified_archive':
                old,h = legacy_test[e.example_id]
                score = dict(predicted_semantic=old['predicted_semantic'],candidate_logprobs=old['candidate_logprobs'])
                provenance = 'historical_variant_activation'
            else:
                score,captured = forward_one(e,(SELECTED_LAYER,))
                h = captured[SELECTED_LAYER].astype(np.float16).astype(np.float32)
                provenance = 'new_capture'
            assert np.isfinite(h).all(), 'Test activation cannot be stored as float16'
            states.append(h)
            records.append(dict(example_id=e.example_id,variant=e.variant,provenance=provenance,
                decoded_torque=decode(h),predicted_semantic=score['predicted_semantic'],
                candidate_logprobs=score['candidate_logprobs'],correct=score['predicted_semantic']==str(e.gold_semantic)))
        scene_prediction = decode(np.mean(states,axis=0))
        fixed_variant = f"permutation_{int(stable_key('variant:'+str(examples[0].scene_id))[:8],16)%6}"
        fixed = next(r for r in records if r['variant']==fixed_variant)
        return dict(scene_id=str(examples[0].scene_id),gold_semantic=str(examples[0].gold_semantic),
            net_torque=int(examples[0].net_torque),scene_mean_prediction=scene_prediction,
            scene_mean_readout_correct=probe_outcome(scene_prediction)==str(examples[0].gold_semantic),
            six_map_model_accuracy=float(np.mean([r['correct'] for r in records])),
            fixed_variant=fixed_variant,fixed_readout_correct=probe_outcome(fixed['decoded_torque'])==str(examples[0].gold_semantic),
            fixed_model_correct=fixed['correct'],fixed_decoded_torque=fixed['decoded_torque'],variants=records,
            test_capture_mode=mode,test_reuse_sha=TEST_REUSE_SHA)
    evaluation_rows.append(cached_scene(EVALUATION_STAGE,sid,
        dict(probe=PROBE_SHA,inputs=INPUT_SHA,reuse=TEST_REUSE_SHA,numerical=NUMERIC_LOCK),evaluate_scene))
eval_frame = pd.DataFrame([{k:v for k,v in r.items() if k!='variants'} for r in evaluation_rows])
eval_frame.to_csv(ANALYSIS_DIR/'readout_1007_scenes.csv',index=False)
print('测试场景采集完成，可以继续运行后面的干预单元格。')
