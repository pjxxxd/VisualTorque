"""Cell source for the standalone E1 notebook. Execute the generated .ipynb in Colab."""

# %% 1. 选择模型并冻结运行参数
cell_stage('Locking model, numerical targets, controls, and output identity')
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import transformers

MODEL_KEY = 'qwen9b'  # qwen9b / gemma12b / qwen27b; one fresh runtime per model
MODEL_SPECS = {
    'qwen9b': ('Qwen/Qwen3.5-9B', 'Qwen__Qwen3.5-9B', 32,
               'c202236235762e1c871ad0ccb60c8ee5ba337b9a', 13, .01),
    'gemma12b': ('google/gemma-4-12B-it', 'google__gemma-4-12B-it', 48, None, 7, 100.),
    'qwen27b': ('Qwen/Qwen3.5-27B', 'Qwen__Qwen3.5-27B__373612cb', 64,
                '373612cb4f9288fcf550b526d86d85febedd9801', None, None),
}
MODEL_ID, MODEL_SLUG, EXPECTED_LAYERS, MODEL_REVISION, EXPECTED_LAYER, EXPECTED_ALPHA = MODEL_SPECS[MODEL_KEY]
MODEL_DTYPE = 'bfloat16'
BATCH_SIZE = 1
SEED = 20260820  # Exact original 304-scene / 72-control / mapping selection.
CALIBRATION_STEPS = 32
FIXED_DOSE_CALIBRATION_STEPS = 12  # Original v2 intervention.
READOUT_TOLERANCE = 0.05
NORM_TOLERANCE = 0.02
BALANCE_THRESHOLD = 0.5  # Same original outcome decoder; not the write-in tolerance.
PARITY_TOLERANCE = 0.001
RANDOM_SEEDS = (0, 1, 2, 3)
RIDGE_ALPHAS = (.01, .1, 1., 10., 100., 1000.)
DOSE_MULTIPLIERS = (-2., -1., 0., 1., 2.)
SCENE_CAPS = {'distance': 54, 'conflict_distance': 42, 'conflict_balance': 12,
              'balance': 6, 'weight': 40, 'conflict_weight': 40,
              'compatible_weight_distance': 40, 'multi_stack': 70}
SHARED_ROOT = PROJECT_ROOT / 'results' / 'cross_model_readout_correction_v2'
OUTPUT_ROOT = SHARED_ROOT / 'models' / MODEL_SLUG
ANALYSIS_DIR = OUTPUT_ROOT / 'analysis'
DISCOVERY_ROOT = PROJECT_ROOT / 'results' / MODEL_SLUG / 'activation_discovery_v1'
CONFIRMATORY_ROOT = PROJECT_ROOT / 'results' / MODEL_SLUG / 'activation_confirmatory_v1'
CAUSAL_ROOT = PROJECT_ROOT / 'results' / MODEL_SLUG / 'causal_torque_intervention_v2'
ACTIVATION_MANIFEST = PROJECT_ROOT / 'results' / 'behavior_comparison_v3' / 'activation_pilot_manifest.jsonl'
for folder in (OUTPUT_ROOT, ANALYSIS_DIR, SHARED_ROOT / 'cross_model_analysis'):
    folder.mkdir(parents=True, exist_ok=True)

def json_plain(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)

def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, default=json_plain, allow_nan=False))

def identity_hash(value):
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True).encode()).hexdigest()

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return 'sha256:' + digest.hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(canonical(value), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]

def lock_json(path, value):
    value = canonical(value)
    if Path(path).exists():
        assert read_json(path) == value, f'Identity differs; do not mix runs: {path}'
    else:
        write_json(path, value)
    return identity_hash(value)

discovery_config_path = DISCOVERY_ROOT / 'capture_config.json'
discovery_selection_path = DISCOVERY_ROOT / 'probe_results' / 'discovery_selection.json'
discovery_activation_path = DISCOVERY_ROOT / 'scene_mean_activations.pt'
if MODEL_REVISION is None:
    assert discovery_config_path.is_file(), f'Missing original pinned checkpoint: {discovery_config_path}'
    MODEL_REVISION = read_json(discovery_config_path)['identity']['resolved_revision']
assert re.fullmatch(r'[0-9a-f]{40}', MODEL_REVISION)
protocol = dict(schema='readout-correction-1.1', model_id=MODEL_ID, revision=MODEL_REVISION,
    implementation=IMPLEMENTATION_SHA256, dtype=MODEL_DTYPE, batch_size=BATCH_SIZE,
    readout_tolerance=READOUT_TOLERANCE, balance_threshold=BALANCE_THRESHOLD,
    norm_tolerance=NORM_TOLERANCE, random_seeds=RANDOM_SEEDS, calibration_steps=CALIBRATION_STEPS,
    fixed_dose_steps=FIXED_DOSE_CALIBRATION_STEPS, ridge_alphas=RIDGE_ALPHAS,
    dose_multipliers=DOSE_MULTIPLIERS, selection_seed=SEED, scene_caps=SCENE_CAPS,
    primary_estimand='paired model accuracy change, oracle correction minus zero replay, all 304 scenes',
    controls='raw random and standardized-orthogonal; match actual raw residual norm; 4 seeds averaged within scene',
    oracle_uses_true_torque=True, bootstrap_unit='scene', bootstrap_replicates=10000,
    accuracy_sesoi=.05, reuse_policy='verified original per-example records only; paired intervention baseline always fresh')
PROTOCOL_SHA = lock_json(OUTPUT_ROOT / 'protocol.json', protocol)
print('Model:', MODEL_ID, MODEL_REVISION)
print('Output:', OUTPUT_ROOT)

# %% 2. 读取原训练/开发清单与模型
cell_stage('Loading the 254 discovery scenes; test outcomes are not used for selection')
from visual_torque_toolkit.behavior import BehaviorRunConfig, PromptCondition, load_behavior_examples
assert ACTIVATION_MANIFEST.is_file(), f'Missing original 254-scene discovery manifest: {ACTIVATION_MANIFEST}'
discovery_manifest = read_jsonl(ACTIVATION_MANIFEST)
assert len(discovery_manifest) == 254
fit_ids = {str(row['scene_id']) for row in discovery_manifest}
assert len(fit_ids) == 254
assert {row['split'] for row in discovery_manifest} <= {'probe_train', 'probe_dev'}
def example_config(splits):
    return BehaviorRunConfig(dataset_dir=DATASET_DIR, output_dir=Path('/content/unused_e1'),
        model_id=MODEL_ID, model_revision=MODEL_REVISION, conditions=(PromptCondition.IMAGE_RULE,),
        question_kinds=('outcome',), splits=tuple(splits), batch_size=1, generation_every=0, seed=SEED)
fit_examples = [e for e in load_behavior_examples(example_config(('probe_train', 'probe_dev')))
                if str(e.scene_id) in fit_ids]
assert len(fit_examples) == 254 * 6
fit_by_scene = defaultdict(list)
for example in fit_examples:
    fit_by_scene[str(example.scene_id)].append(example)
assert all(len(v) == 6 and len({e.variant for e in v}) == 6 for v in fit_by_scene.values())
for row in discovery_manifest:
    assert {e.split for e in fit_by_scene[str(row['scene_id'])]} == {row['split']}

def example_identity(example):
    return dict(example_id=example.example_id, scene_id=str(example.scene_id), split=example.split,
        variant=example.variant, prompt=example.prompt, labels=list(example.labels),
        label_to_semantic={str(k): str(v) for k,v in example.label_to_semantic.items()},
        image_sha256=sha256_file(example.image_path), net_torque=int(example.net_torque),
        gold_semantic=str(example.gold_semantic), problem_type=example.problem_type)

toolkit_identity = {str(p.relative_to(TOOLKIT_ROOT)): sha256_file(p)
                    for p in sorted((TOOLKIT_ROOT / 'src' / 'visual_torque_toolkit').rglob('*.py'))}
runtime_identity = dict(torch=torch.__version__, transformers=transformers.__version__,
    toolkit=toolkit_identity, dataset=example_config(('probe_train','probe_dev')).dataset_fingerprint(),
    manifest=sha256_file(ACTIVATION_MANIFEST))
RUNTIME_SHA = lock_json(OUTPUT_ROOT / 'runtime_identity.json', runtime_identity)

import gc
if 'adapter' in globals():
    del adapter
    gc.collect()
    torch.cuda.empty_cache()
adapter = HuggingFaceMultimodalAdapter(MODEL_ID, revision=MODEL_REVISION, dtype=MODEL_DTYPE, device_map='auto')
assert adapter.metadata['resolved_revision'] == MODEL_REVISION
DECODER_LAYERS = None
for candidate in ('model.language_model.layers', 'model.text_model.layers', 'language_model.model.layers',
                  'language_model.layers', 'model.model.layers', 'model.layers'):
    module = adapter.model
    try:
        for component in candidate.split('.'):
            module = getattr(module, component)
    except AttributeError:
        continue
    if isinstance(module, torch.nn.ModuleList) and len(module) == EXPECTED_LAYERS:
        DECODER_LAYERS, LAYER_PATH = module, candidate
        break
assert DECODER_LAYERS is not None, 'Cannot find the expected decoder layers'
assert not any(v == 'disk' for v in getattr(adapter.model, 'hf_device_map', {}).values()), 'Use enough memory; disk offload is unsuitable here'
print('Decoder layers:', len(DECODER_LAYERS), '| native checkpoint:', MODEL_REVISION)

# %% 3. 定义单样本前向、原生精度写入与断点保存
cell_stage('Defining exact final-prompt capture and scoring')
def output_hidden(output):
    if torch.is_tensor(output):
        hidden = output
    elif isinstance(output, (tuple, list)) and torch.is_tensor(output[0]):
        hidden = output[0]
    else:
        raise TypeError(f'Unsupported decoder output: {type(output).__name__}')
    assert hidden.ndim == 3
    return hidden

def replace_hidden(output, value):
    if torch.is_tensor(output):
        return value
    if isinstance(output, tuple):
        return (value,) + output[1:]
    if isinstance(output, list):
        return [value] + output[1:]
    raise TypeError(type(output).__name__)

def forward_one(example, layers=(), edit=None):
    plan = adapter._candidate_plan(example)
    assert plan is not None, 'A/B/C must have the existing single-token scoring plan'
    prefix, token_by_label = plan
    inputs = adapter._append_text_prefix(adapter._to_device(adapter._template(
        [example.messages()], add_generation_prompt=True)), prefix)
    assert bool(inputs['attention_mask'][0, -1]), 'Final position is padded'
    captured, audits, handles = {}, {}, []
    for layer in layers:
        def hook(_module, _inputs, output, layer=layer):
            hidden = output_hidden(output)
            assert hidden.dtype == torch.bfloat16, hidden.dtype
            pre = hidden[0, -1].detach().float().cpu().numpy().copy()
            captured[layer] = pre
            if edit is None:
                return None
            assert len(layers) == 1
            cast = lambda x: torch.as_tensor(np.asarray(x).copy(), dtype=hidden.dtype).float().numpy()
            post, audit = edit(pre, cast)
            modified = hidden.clone()
            modified[0, -1] = torch.as_tensor(post, dtype=hidden.dtype, device=hidden.device)
            actual = modified[0, -1].detach().float().cpu().numpy()
            audit.update(actual_delta_norm=float(np.linalg.norm(actual - pre)),
                         standardized_delta_norm=float(np.linalg.norm((actual-pre)/PROBE_SCALE)),
                         decoded_pre=decode(pre), decoded_post=decode(actual),
                         hidden_norm=float(np.linalg.norm(pre)))
            audits.update(audit)
            return replace_hidden(output, modified)
        handles.append(DECODER_LAYERS[layer].register_forward_hook(hook))
    try:
        with torch.inference_mode():
            output = adapter.model(**inputs, use_cache=False)
            scores = torch.log_softmax(output.logits[0, -1].float(), dim=-1)
            label_scores = {label: float(scores[token_by_label[label]].cpu()) for label in example.labels}
    finally:
        for handle in handles:
            handle.remove()
    assert set(captured) == set(layers)
    predicted_label = max(example.labels, key=lambda label: label_scores[label])
    semantic_scores = {str(example.label_to_semantic[label]): value for label,value in label_scores.items()}
    result = dict(example_id=example.example_id, scene_id=str(example.scene_id), variant=example.variant,
        split=example.split, problem_type=example.problem_type, net_torque=int(example.net_torque),
        gold_semantic=str(example.gold_semantic), predicted_semantic=str(example.label_to_semantic[predicted_label]),
        correct=str(example.label_to_semantic[predicted_label]) == str(example.gold_semantic),
        label_to_semantic={str(k):str(v) for k,v in example.label_to_semantic.items()},
        candidate_logprobs=label_scores, semantic_logprobs=semantic_scores, **audits)
    return result, captured

def cached_scene(stage, scene_id, identity, compute):
    directory = OUTPUT_ROOT / stage
    directory.mkdir(parents=True, exist_ok=True)
    filename = hashlib.sha256(str(scene_id).encode()).hexdigest()[:24] + '.json'
    path = directory / filename
    expected = identity_hash(dict(protocol=PROTOCOL_SHA, runtime=RUNTIME_SHA, stage=stage, identity=identity))
    if path.is_file():
        payload = read_json(path)
        assert payload['identity'] == expected and payload['scene_id'] == str(scene_id), path
        return payload['result']
    result = compute()
    write_json(path, dict(identity=expected, scene_id=str(scene_id), result=result))
    return canonical(result)

# %% 4. 复用 ridge；27B 缺少归档时在原开发清单选层
cell_stage('Fitting the readout without test outcomes')
archive_available = all(p.is_file() for p in (discovery_config_path, discovery_selection_path, discovery_activation_path))
if EXPECTED_LAYER is not None:
    assert archive_available, 'The small-model experiment requires its original discovery artifacts; do not refit on a different sample'
if archive_available:
    di = read_json(discovery_config_path)['identity']
    selection = read_json(discovery_selection_path)
    assert di['model_id'] == MODEL_ID and di['resolved_revision'] == MODEL_REVISION
    assert di['dataset_fingerprint'] == runtime_identity['dataset']
    assert di['activation_manifest_sha256'] == sha256_file(ACTIVATION_MANIFEST)
    assert selection['probe_test_used'] is False
    assert selection['selection_data'] == ['probe_train', 'probe_dev']
    selected = selection['candidate_net_torque_layer']
    SELECTED_LAYER = int(selected['layer_index'])
    FROZEN_ALPHA = float(selected['selected_alpha'])
    if EXPECTED_LAYER is not None:
        assert (SELECTED_LAYER, FROZEN_ALPHA) == (EXPECTED_LAYER, EXPECTED_ALPHA)
    payload = torch.load(discovery_activation_path, map_location='cpu', weights_only=False)
    meta = pd.DataFrame(payload['metadata'])
    assert meta['scene_id'].astype(str).is_unique and set(meta['scene_id'].astype(str)) == fit_ids
    assert set(meta['split']) <= {'probe_train', 'probe_dev'}
    for row in meta.to_dict('records'):
        ex = fit_by_scene[str(row['scene_id'])][0]
        assert row['split'] == ex.split and int(row['net_torque']) == int(ex.net_torque)
    li = [int(x) for x in payload['layer_indices']].index(SELECTED_LAYER)
    X_fit_raw = payload['activations'][:, li].float().numpy()
    y_fit = meta['net_torque'].to_numpy(dtype=float)
    fit_provenance = dict(mode='original_discovery_archive', files={str(p):sha256_file(p)
        for p in (discovery_config_path, discovery_selection_path, discovery_activation_path)})
else:
    # Same eight-point grid and original 254-scene pilot as the two small models.
    grid = tuple(sorted({int(round(x)) for x in np.linspace(0, EXPECTED_LAYERS-1, 8)}))
    rows, activations = [], []
    for scene_id in tqdm(sorted(fit_ids), desc='Qwen-27B discovery scene means', unit='scene'):
        examples = sorted(fit_by_scene[scene_id], key=lambda e:e.variant)
        def capture_fit(examples=examples):
            states = []
            for example in examples:
                _, captured = forward_one(example, grid)
                states.append(np.stack([captured[layer] for layer in grid]).astype(np.float16).astype(np.float32))
            # Match original float16 scene-mean storage before probe fitting.
            return {'mean': np.mean(states, axis=0).astype(np.float16).astype(np.float32).tolist()}
        result = cached_scene('discovery_shards', scene_id,
            dict(layers=grid, examples=[example_identity(e) for e in examples]), capture_fit)
        activations.append(np.asarray(result['mean'], dtype=np.float32))
        rows.append(dict(scene_id=scene_id, split=examples[0].split, net_torque=int(examples[0].net_torque)))
    X_all = np.stack(activations)
    meta = pd.DataFrame(rows)
    y_fit = meta['net_torque'].to_numpy(dtype=float)
    train, dev = meta['split'].eq('probe_train').to_numpy(), meta['split'].eq('probe_dev').to_numpy()
    assert train.sum() > 1 and dev.sum() > 1
    selection_rows = []
    for pos, layer in enumerate(grid):
        scaler = StandardScaler().fit(X_all[train, pos])
        for alpha in RIDGE_ALPHAS:
            estimator = Ridge(alpha=alpha, solver='lsqr', max_iter=5000).fit(scaler.transform(X_all[train, pos]), y_fit[train])
            score = float(r2_score(y_fit[dev], estimator.predict(scaler.transform(X_all[dev, pos]))))
            assert np.isfinite(score)
            selection_rows.append(dict(layer=layer, alpha=alpha, dev_r2=score))
    best = max(selection_rows, key=lambda row:(row['dev_r2'], -row['alpha'], -row['layer']))
    SELECTED_LAYER, FROZEN_ALPHA = int(best['layer']), float(best['alpha'])
    X_fit_raw = X_all[:, grid.index(SELECTED_LAYER)]
    fit_provenance = dict(mode='same_254_scene_ridge_discovery', manifest_sha=sha256_file(ACTIVATION_MANIFEST), grid=grid)
    write_json(OUTPUT_ROOT / 'development_selection.json', dict(all_candidates=selection_rows, selected=best))
scaler = StandardScaler()
X_fit = scaler.fit_transform(X_fit_raw)
probe = Ridge(alpha=FROZEN_ALPHA, solver='lsqr', max_iter=5000).fit(X_fit, y_fit)
PROBE_MEAN = np.asarray(scaler.mean_, dtype=np.float32)
PROBE_SCALE = np.asarray(scaler.scale_, dtype=np.float32)
PROBE_COEF = np.asarray(probe.coef_, dtype=np.float32)
PROBE_INTERCEPT = float(probe.intercept_)
TORQUE_SHIFT_UNIT = .5 * float(np.std(y_fit, ddof=0))
assert TORQUE_SHIFT_UNIT > 0
probe_artifact = dict(layer=SELECTED_LAYER, alpha=FROZEN_ALPHA, mean=PROBE_MEAN, scale=PROBE_SCALE,
    coef=PROBE_COEF, intercept=PROBE_INTERCEPT, torque_shift_unit=TORQUE_SHIFT_UNIT,
    fit_scene_ids=meta['scene_id'].astype(str).tolist(), fit_provenance=fit_provenance)
PROBE_SHA = lock_json(OUTPUT_ROOT / 'frozen_probe.json', probe_artifact)
print('Frozen layer / ridge alpha:', SELECTED_LAYER, FROZEN_ALPHA)
print('Fit scenes:', len(y_fit), '| five-dose unit:', TORQUE_SHIFT_UNIT)

# %% 5. 数值修准、原五剂量与匹配随机变化
cell_stage('Defining the numerical correction and finite-precision controls')
# EMBED_EDIT_MATH
def decode(hidden):
    return readout_value(hidden, PROBE_MEAN, PROBE_SCALE, PROBE_COEF, PROBE_INTERCEPT)

def probe_outcome(value):
    return 'left' if value > BALANCE_THRESHOLD else 'right' if value < -BALANCE_THRESHOLD else 'balanced'

def oracle_edit(gold):
    def apply(pre, cast):
        candidate, audit = numerical_correction(pre, float(gold), PROBE_MEAN, PROBE_SCALE,
            PROBE_COEF, PROBE_INTERCEPT, cast, READOUT_TOLERANCE, CALIBRATION_STEPS)
        audit.update(condition='oracle', target_torque=float(gold))
        return candidate, audit
    return apply

def zero_edit(pre, cast):
    return pre.copy(), dict(condition='zero', planned_delta_norm=0.)

def fixed_edit(multiplier):
    def apply(pre, cast):
        # Reproduce v2: float32 correction, all 12 steps, keep the smallest realized error.
        coefficient = torch.as_tensor(PROBE_COEF)
        scale = torch.as_tensor(PROBE_SCALE)
        unit = scale * coefficient / torch.dot(coefficient, coefficient)
        h = torch.as_tensor(pre, dtype=torch.float32)
        read = lambda v: ((v.float()-torch.as_tensor(PROBE_MEAN))/scale) @ coefficient + PROBE_INTERCEPT
        before = read(h)
        target_shift = float(multiplier * TORQUE_SHIFT_UNIT)
        work = target_shift * unit
        best, best_error = h.clone(), float('inf')
        for _ in range(FIXED_DOSE_CALIBRATION_STEPS):
            candidate = torch.as_tensor(cast((h+work).numpy()))
            residual = target_shift - (read(candidate)-before)
            error = float(residual.abs())
            if error < best_error:
                best, best_error = candidate.clone(), error
            work += residual * unit
        return best.numpy(), dict(condition=f'fixed_{multiplier:+g}', dose_multiplier=multiplier,
            target_torque=float(before)+target_shift, planned_probe_shift=target_shift,
            residual=best_error, achieved=best_error <= max(.15, .03*abs(target_shift)))
    return apply

def control_edit(scene_id, kind, seed, actual_oracle_norm):
    def apply(pre, cast):
        # Random is isotropic in raw residual space; orthogonal is formed in standardized space.
        scale = PROBE_SCALE if kind == 'orthogonal' else np.ones_like(PROBE_SCALE)
        direction = deterministic_direction(pre.shape, f'{SEED}:{scene_id}:{kind}:{seed}',
            scale, PROBE_COEF, orthogonal=kind == 'orthogonal')
        candidate, audit = match_native_norm(pre, direction, actual_oracle_norm, cast, NORM_TOLERANCE)
        leakage = decode(candidate) - decode(pre)
        audit.update(condition=kind, random_seed=seed, oracle_actual_norm=actual_oracle_norm,
            orthogonal_readout_leakage=leakage if kind == 'orthogonal' else None,
            orthogonal_preserved=abs(leakage) <= READOUT_TOLERANCE if kind == 'orthogonal' else None)
        return candidate, audit
    return apply

# %% 6. 只用开发样本检查写入与零剂量重放
cell_stage('Checking implementation on fixed development examples, before test capture')
dev_examples = sorted((e for e in fit_examples if e.split == 'probe_dev'),
    key=lambda e:hashlib.sha256(f'dev-write:{e.example_id}'.encode()).hexdigest())[:8]
assert len(dev_examples) == 8
dev_rows = []
for example in tqdm(dev_examples, desc='Development write-in checks'):
    plain, _ = forward_one(example)
    zero, _ = forward_one(example, (SELECTED_LAYER,), zero_edit)
    assert max(abs(plain['candidate_logprobs'][k]-zero['candidate_logprobs'][k])
               for k in plain['candidate_logprobs']) < PARITY_TOLERANCE
    assert zero['actual_delta_norm'] == 0
    oracle, _ = forward_one(example, (SELECTED_LAYER,), oracle_edit(example.net_torque))
    assert np.isfinite(oracle['decoded_post']) and np.isfinite(oracle['actual_delta_norm'])
    dev_rows.append(oracle)
    for kind in ('random', 'orthogonal'):
        control, _ = forward_one(example, (SELECTED_LAYER,), control_edit(example.scene_id, kind, 0, oracle['actual_delta_norm']))
        dev_rows.append(control)
write_json(OUTPUT_ROOT / 'development_write_audit.json', dev_rows)
display(pd.DataFrame(dev_rows)[['scene_id','condition','actual_delta_norm','decoded_pre','decoded_post','achieved','norm_matched']])
dev_success = [r['achieved'] for r in dev_rows if r['condition'] == 'oracle']
print('Oracle tolerance attained:', sum(dev_success), '/', len(dev_success))
assert any(dev_success), ('None of the fixed development examples reached the numerical tolerance. '
    'Inspect development_write_audit.json before running test interventions; do not loosen the threshold on test data.')
# Numerical failures are reported, not grounds for choosing a new tolerance using test outcomes.
NUMERIC_LOCK = lock_json(OUTPUT_ROOT / 'numerical_protocol_lock.json', dict(probe_sha=PROBE_SHA,
    tolerance=READOUT_TOLERANCE, steps=CALIBRATION_STEPS, control_norm_tolerance=NORM_TOLERANCE,
    dev_example_ids=[e.example_id for e in dev_examples], dev_oracle_successes=sum(dev_success)))

# %% 7. 锁定原 1,007 场景、304 场景与 72 个对照场景
cell_stage('Reconstructing the exact historical evaluation cohorts')
all_examples = tuple(load_behavior_examples(example_config(('probe_test',))))
test_by_scene = defaultdict(list)
for example in all_examples:
    test_by_scene[str(example.scene_id)].append(example)
assert len(test_by_scene) == 1007 and len(all_examples) == 6042
assert set(test_by_scene).isdisjoint(fit_ids)
assert all(len(v) == 6 and len({e.variant for e in v}) == 6 for v in test_by_scene.values())
scene_by_id = {str(s['scene_id']):s for s in read_jsonl(DATASET_DIR/'scenes.jsonl')}
def stable_key(value):
    return hashlib.sha256(f'{SEED}:{value}'.encode()).hexdigest()
def balanced_take(scenes, cap):
    groups = defaultdict(list)
    for scene in scenes:
        groups[str(scene['derived']['outcome'])].append(scene)
    for values in groups.values():
        values.sort(key=lambda scene:stable_key(scene['scene_id']))
    selected, cursor = [], {k:0 for k in groups}
    while len(selected) < min(cap, len(scenes)):
        for key in sorted(groups):
            if cursor[key] < len(groups[key]) and len(selected) < cap:
                selected.append(groups[key][cursor[key]])
                cursor[key] += 1
    return selected
selected_scenes = []
for problem_type, cap in SCENE_CAPS.items():
    selected_scenes += balanced_take([scene_by_id[s] for s in test_by_scene
                                     if scene_by_id[s]['problem_type'] == problem_type], cap)
selected_examples = []
for scene in sorted(selected_scenes, key=lambda s:s['scene_id']):
    sid = str(scene['scene_id'])
    variant = f"permutation_{int(stable_key('variant:'+sid)[:8],16)%6}"
    selected_examples.append(next(e for e in test_by_scene[sid] if e.variant == variant))
groups = defaultdict(list)
for example in selected_examples:
    groups[example.problem_type].append(example)
for values in groups.values():
    values.sort(key=lambda e:stable_key('control:'+e.scene_id))
control_examples, cursor = [], {k:0 for k in groups}
while len(control_examples) < 72:
    for key in sorted(groups):
        if cursor[key] < len(groups[key]) and len(control_examples) < 72:
            control_examples.append(groups[key][cursor[key]])
            cursor[key] += 1
control_ids = {str(e.scene_id) for e in control_examples}
selection_manifest = [dict(schema_version='causal-selection-1.0', example_id=e.example_id,
    scene_id=str(e.scene_id), variant=e.variant, problem_type=e.problem_type,
    gold_outcome=str(e.gold_semantic), net_torque=int(e.net_torque),
    orthogonal_control_subset=str(e.scene_id) in control_ids) for e in selected_examples]
assert len(selection_manifest) == 304 and len(control_ids) == 72
assert Counter(e.gold_semantic for e in selected_examples) == {'left':132, 'right':131, 'balanced':41}
for spec in MODEL_SPECS.values():
    old_manifest = PROJECT_ROOT/'results'/spec[1]/'causal_torque_intervention_v2'/'selection_manifest.jsonl'
    if old_manifest.is_file():
        assert read_jsonl(old_manifest) == selection_manifest, f'Historical cohort mismatch: {old_manifest}'
COHORT_SHA = lock_json(SHARED_ROOT/'shared_304_manifest.json', selection_manifest)
lock_json(SHARED_ROOT/'shared_1007_scene_ids.json', sorted(test_by_scene))
input_identities = {e.example_id:example_identity(e) for e in tqdm(all_examples, desc='Fingerprinting exact test inputs')}
INPUT_SHA = lock_json(OUTPUT_ROOT/'test_inputs.json', input_identities)
print('Evaluation scenes / selected scenes / control scenes:', len(test_by_scene), len(selected_examples), len(control_ids))

# %% 8. 核验旧测试状态；不兼容时自动重新采集
# EMBED_ARCHIVE_REUSE

# %% 9. 304 场景的修准与对照，保留失败记录并按场景恢复
cell_stage('Running paired zero/oracle controls and preserving the original five doses')
# Original fixed-dose records are supplementary to the fresh zero/oracle pair.
legacy_fixed = {}
legacy_fixed_files = None
old_config_path = CAUSAL_ROOT/'capture_config.json'
old_manifest_path = CAUSAL_ROOT/'selection_manifest.jsonl'
old_shards = sorted((CAUSAL_ROOT/'shards').glob('batch_*.json'))
if not old_shards:
    old_shards = sorted((CAUSAL_ROOT/'intervention_shards').glob('batch_*.json'))
if old_config_path.is_file() and old_manifest_path.is_file() and old_shards:
    identity = read_json(old_config_path)['identity']
    assert identity['model_id'] == MODEL_ID and identity['resolved_revision'] == MODEL_REVISION
    assert identity['dtype'] == MODEL_DTYPE and identity['selected_layer_index'] == SELECTED_LAYER
    assert identity['dataset_fingerprint'] == runtime_identity['dataset']
    assert identity['discovery_selection_sha256'] == sha256_file(discovery_selection_path)
    assert identity['frozen_alpha'] == FROZEN_ALPHA
    assert abs(identity['torque_shift_unit']-TORQUE_SHIFT_UNIT) < 1e-6
    assert identity['discovery_activations_sha256'] == sha256_file(discovery_activation_path)
    assert identity['selection_manifest_sha256'] == sha256_file(old_manifest_path)
    assert read_jsonl(old_manifest_path) == selection_manifest
    for path in old_shards:
        shard = read_json(path)
        assert shard['capture_identity_sha256'] == sha256_file(old_config_path)
        for r in shard['records']:
            if r['condition_family'] != 'torque_direction':
                continue
            key = (r['example_id'], float(r['dose_multiplier']))
            assert key not in legacy_fixed
            legacy_fixed[key] = r
    legacy_fixed_files = {str(p):sha256_file(p) for p in [old_config_path,old_manifest_path,*old_shards]}
FIXED_REUSE_SHA = lock_json(OUTPUT_ROOT/'fixed_dose_reuse_identity.json',dict(files=legacy_fixed_files))
all_records = []
for example in tqdm(selected_examples, desc='Numerical correction', unit='scene'):
    sid = str(example.scene_id)
    def run_scene(example=example, sid=sid):
        zero, _ = forward_one(example, (SELECTED_LAYER,), zero_edit)
        oracle, _ = forward_one(example, (SELECTED_LAYER,), oracle_edit(example.net_torque))
        rows = [zero,oracle]
        if sid in control_ids:
            for kind in ('random','orthogonal'):
                for seed in RANDOM_SEEDS:
                    row, _ = forward_one(example, (SELECTED_LAYER,), control_edit(sid,kind,seed,oracle['actual_delta_norm']))
                    rows.append(row)
        complete_legacy = all((example.example_id,m) in legacy_fixed for m in DOSE_MULTIPLIERS)
        for multiplier in DOSE_MULTIPLIERS:
            if complete_legacy:
                r = legacy_fixed[(example.example_id,multiplier)]
                old_zero = legacy_fixed[(example.example_id,0.)]
                row = dict(example_id=example.example_id,scene_id=sid,variant=example.variant,
                    split='probe_test',problem_type=example.problem_type,net_torque=int(example.net_torque),
                    gold_semantic=str(example.gold_semantic),predicted_semantic=r['predicted_semantic'],correct=r['correct'],
                    condition=f'fixed_{multiplier:+g}',dose_multiplier=multiplier,
                    semantic_logprobs={k:r['logprob_'+k] for k in ('left','right','balanced')},
                    decoded_pre=r['decoded_torque_pre'],decoded_post=r['decoded_torque_post'],
                    actual_delta_norm=r['realized_delta_norm'], provenance='historical_v2',
                    fixed_baseline_correct=old_zero['correct'])
            else:
                row, _ = forward_one(example,(SELECTED_LAYER,),fixed_edit(multiplier))
                row.update(provenance='new_v2_rule',fixed_baseline_correct=zero['correct'])
            rows.append(row)
        for r in rows:
            r.setdefault('provenance','new_capture')
            r['original_model_correct'] = zero['correct']
            r['original_readout_correct'] = probe_outcome(zero['decoded_pre']) == str(example.gold_semantic)
            r['cohort_sha'] = COHORT_SHA
            r['model_id'],r['revision'],r['layer'] = MODEL_ID,MODEL_REVISION,SELECTED_LAYER
        return rows
    rows = cached_scene('intervention_scene_shards',sid,
        dict(probe=PROBE_SHA, input=input_identities[example.example_id], cohort=COHORT_SHA,
             numerical=NUMERIC_LOCK, fixed_reuse=FIXED_REUSE_SHA),run_scene)
    assert len(rows) == (15 if sid in control_ids else 7)
    all_records.extend(rows)
assert len(all_records) == 304*7+72*8
write_json(ANALYSIS_DIR/'intervention_records.json',all_records)
record_frame = pd.DataFrame(all_records)
record_frame.to_csv(ANALYSIS_DIR/'intervention_records.csv',index=False)

# %% 10. 汇总读出、答案、修正与新增错误，报告区间
cell_stage('Writing readable numerical and behavioral summaries with separate denominators')
record_frame['probe_correct'] = [probe_outcome(float(r['decoded_post']))==r['gold_semantic'] for r in all_records]
record_frame['absolute_readout_error'] = (record_frame['decoded_post']-record_frame['net_torque']).abs()
record_frame['model_change'] = record_frame['correct'].astype(float)-record_frame['original_model_correct'].astype(float)
record_frame['correct_side_margin'] = [
    (r['semantic_logprobs']['left']-r['semantic_logprobs']['right']) * (1 if r['net_torque']>0 else -1)
    if r['net_torque'] != 0 else np.nan for r in all_records]
summary_rows = []
for group_name, mask in [('all_304',np.ones(len(record_frame),dtype=bool)),
                         ('tilt_263',record_frame['net_torque'].ne(0)),
                         ('balance_41',record_frame['net_torque'].eq(0))]:
    for condition, frame in record_frame.loc[mask].groupby('condition'):
        # Seeds are repeated measurements: average inside scene, never inflate n.
        per_scene = frame.groupby('scene_id')[['correct','probe_correct','absolute_readout_error','model_change']].mean()
        accuracy_ci = scene_bootstrap(per_scene['correct'].to_numpy())
        change_ci = scene_bootstrap(per_scene['model_change'].to_numpy())
        if condition.startswith('fixed_'):
            fixed_change = frame['correct'].astype(float)-frame['fixed_baseline_correct'].astype(float)
            change_ci = scene_bootstrap(fixed_change.to_numpy())
        summary_rows.append(dict(group=group_name,condition=condition,scenes=len(per_scene),
            readout_mae=float(per_scene['absolute_readout_error'].mean()),
            readout_accuracy=float(per_scene['probe_correct'].mean()),
            model_accuracy=accuracy_ci['mean'],accuracy_lower=accuracy_ci['lower'],accuracy_upper=accuracy_ci['upper'],
            model_change=change_ci['mean'],change_lower=change_ci['lower'],change_upper=change_ci['upper']))
summary = pd.DataFrame(summary_rows)
summary.to_csv(ANALYSIS_DIR/'summary_by_cohort.csv',index=False)
zero = record_frame[record_frame['condition']=='zero'].copy()
oracle = record_frame[record_frame['condition']=='oracle'].copy()
assert zero['scene_id'].is_unique and oracle['scene_id'].is_unique
assert set(zero['scene_id']) == set(oracle['scene_id'])
achieved = oracle['achieved'].eq(True)
paired_effect = scene_bootstrap(oracle['model_change'].to_numpy())
repaired = int((~oracle['original_model_correct'].astype(bool)&oracle['correct'].astype(bool)).sum())
harmed = int((oracle['original_model_correct'].astype(bool)&~oracle['correct'].astype(bool)).sum())
original_errors = int((~oracle['original_model_correct'].astype(bool)).sum())
original_correct = len(oracle)-original_errors
technical = dict(oracle_achieved=int(achieved.sum()),oracle_total=len(oracle),
    oracle_failed=int((~achieved).sum()), zero_or_tiny=int((oracle['actual_delta_norm']<=1e-10).sum()),
    repaired=repaired,original_errors=original_errors,harmed=harmed,original_correct=original_correct,
    readout_correct_model_wrong=int((zero['original_readout_correct'].astype(bool)&~zero['correct'].astype(bool)).sum()),
    all_scene_accuracy_change=paired_effect,
    achieved_subset_accuracy_change=scene_bootstrap(oracle.loc[achieved,'model_change'].to_numpy()) if achieved.any() else None)
oracle.loc[~achieved].to_csv(ANALYSIS_DIR/'oracle_write_failures.csv',index=False)
controls = record_frame[record_frame['condition'].isin(['random','orthogonal'])].copy()
controls['valid_control'] = controls['norm_matched'].eq(True) & (
    controls['condition'].ne('orthogonal') | controls['orthogonal_preserved'].eq(True))
controls.loc[~controls['valid_control']].to_csv(ANALYSIS_DIR/'control_match_failures.csv',index=False)
control_contrasts = []
for kind, frame in controls.groupby('condition'):
    counts = frame.groupby('scene_id').size()
    assert len(counts) == 72 and set(counts) == {4}
    per_scene = frame.groupby('scene_id')['model_change'].mean().reset_index()
    matched_oracle = oracle[oracle['scene_id'].isin(per_scene['scene_id'])]
    difference = paired_on_ids(matched_oracle,per_scene,'scene_id','model_change')
    control_contrasts.append(dict(control=kind,scope='all_72_attempted',
        **scene_bootstrap(difference),invalid_records=int((~frame['valid_control']).sum())))
    valid_ids = set(frame.groupby('scene_id')['valid_control'].all().loc[lambda x:x].index)
    valid_ids &= set(oracle.loc[achieved,'scene_id'])
    if valid_ids:
        difference = paired_on_ids(matched_oracle[matched_oracle['scene_id'].isin(valid_ids)],
            per_scene[per_scene['scene_id'].isin(valid_ids)],'scene_id','model_change')
        control_contrasts.append(dict(control=kind,scope='oracle_achieved_all_four_controls_valid',**scene_bootstrap(difference)))
pd.DataFrame(control_contrasts).to_csv(ANALYSIS_DIR/'paired_control_accuracy.csv',index=False)
# Margin contrasts are defined only for the original 263 nonbalanced scenes.
tilt_zero = zero[zero['net_torque'].ne(0)][['scene_id','correct_side_margin']]
margin_rows = []
for condition, frame in record_frame[record_frame['net_torque'].ne(0)&~record_frame['condition'].str.startswith('fixed_')].groupby('condition'):
    per_scene = frame.groupby('scene_id')['correct_side_margin'].mean().reset_index()
    diff = paired_on_ids(per_scene,tilt_zero[tilt_zero['scene_id'].isin(per_scene['scene_id'])],'scene_id','correct_side_margin')
    margin_rows.append(dict(condition=condition,**scene_bootstrap(diff)))
pd.DataFrame(margin_rows).to_csv(ANALYSIS_DIR/'tilt_margin_changes.csv',index=False)
for key, frame in [('readout_correct',oracle[oracle['original_readout_correct'].eq(True)]),
                    ('readout_wrong',oracle[oracle['original_readout_correct'].eq(False)])]:
    if len(frame):
        technical[key] = dict(scenes=len(frame),**{'accuracy_change':scene_bootstrap(frame['model_change'].to_numpy())})
evaluation_summary = dict(scenes=len(eval_frame),
    six_map_scene_mean_mae=float(np.abs(eval_frame['scene_mean_prediction']-eval_frame['net_torque']).mean()),
    six_map_scene_mean_readout_accuracy=float(eval_frame['scene_mean_readout_correct'].mean()),
    six_map_model_accuracy=float(eval_frame['six_map_model_accuracy'].mean()),
    single_mapping_readout_accuracy=float(eval_frame['fixed_readout_correct'].mean()),
    single_mapping_model_accuracy=float(eval_frame['fixed_model_correct'].mean()),
    single_mapping_readout_correct_model_wrong=int((eval_frame['fixed_readout_correct']&~eval_frame['fixed_model_correct']).sum()))
write_json(ANALYSIS_DIR/'result_summary.json',dict(model_key=MODEL_KEY,model_id=MODEL_ID,revision=MODEL_REVISION,
    layer=SELECTED_LAYER,probe_sha=PROBE_SHA,cohort_sha=COHORT_SHA,technical=technical,evaluation=evaluation_summary,
    interpretation='Oracle correction edits an external readout; it does not establish repair of native physical computation.'))
display(summary[summary['condition'].isin(['zero','oracle'])])
display(pd.DataFrame([technical]))
lines = ['# 净力矩读出与数值修准实验', '', f'模型：{MODEL_ID}；读出层：{SELECTED_LAYER}。', '',
    f'304 个场景中，原读出正确但模型答错的有 {technical["readout_correct_model_wrong"]} 个。',
    f'数值写入达到预定误差要求的有 {int(achieved.sum())}/304 个；其余记录单独保留。',
    f'在全部 304 个场景上，修准操作修正了 {repaired}/{original_errors} 个原有错误，并使 {harmed}/{original_correct} 个原来正确的答案变错。',
    f'模型准确率变化：{paired_effect["mean"]*100:.2f} 个百分点；95% 场景配对区间为 [{paired_effect["lower"]*100:.2f}, {paired_effect["upper"]*100:.2f}]。',
    '', '随机与正交对照按场景平均四个方向；写入失败与范数不匹配记录见单独表格。',
    '本轮使用原来的物理场景，是配对新增诊断；不构成新家族独立复现。',
    '修准外部预测器不等于修复模型原生物理计算。是否支持较小的答案效应，需要结合区间与成功写入比例判断。']
(ANALYSIS_DIR/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

# %% 11. 三模型共同场景比较：不把显著与不显著直接作差异结论
cell_stage('Comparing paired effects only on identical scene manifests')
import itertools
coverage, effects = [], {}
for key, spec in MODEL_SPECS.items():
    directory = SHARED_ROOT/'models'/spec[1]/'analysis'
    summary_path, raw_path = directory/'result_summary.json', directory/'intervention_records.json'
    if not summary_path.is_file() or not raw_path.is_file():
        coverage.append(dict(model_key=key,status='not_complete'))
        continue
    result = read_json(summary_path)
    assert result['cohort_sha'] == COHORT_SHA
    coverage.append(dict(model_key=key,status='complete',**result['evaluation']))
    raw = read_json(raw_path)
    frame = pd.DataFrame([r for r in raw if r['condition']=='oracle'])
    assert len(frame)==304 and frame['scene_id'].is_unique
    frame['model_change'] = frame['correct'].astype(float)-frame['original_model_correct'].astype(float)
    effects[key] = frame[['scene_id','model_change']]
paired_rows = []
for left,right in itertools.combinations(effects,2):
    differences = paired_on_ids(effects[left],effects[right],'scene_id','model_change')
    paired_rows.append(dict(left_model=left,right_model=right,**scene_bootstrap(differences)))
cross_dir = SHARED_ROOT/'cross_model_analysis'
pd.DataFrame(coverage).to_csv(cross_dir/'coverage_and_readout.csv',index=False)
pd.DataFrame(paired_rows).to_csv(cross_dir/'paired_accuracy_differences.csv',index=False)
display(pd.DataFrame(coverage))
display(pd.DataFrame(paired_rows))
print('Outputs:', ANALYSIS_DIR)
print('Pairwise intervals are descriptive; no significance claims from multiple comparisons are auto-generated.')
