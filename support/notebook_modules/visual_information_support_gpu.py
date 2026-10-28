"""Runtime helpers embedded into the information-support Colab notebook."""
import inspect
import re
import shutil
import time
import zipfile
from datetime import datetime, timezone
from dataclasses import asdict

from tqdm.auto import tqdm
from visual_torque_toolkit.behavior.data import BehaviorExample
from visual_torque_toolkit.behavior.schema import PromptCondition


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def image_for(scene_id, condition):
    if condition == 'text_full':
        return None
    return IMAGE_DIR / (scene_id + ('__plain.png' if condition == 'legacy_image' else '__ids.png'))


def make_example(spec):
    scene = SCENES[spec['scene_id']]
    image = image_for(spec['scene_id'], spec['condition'])
    return BehaviorExample(example_id=spec['trial_id'], question_id=spec['trial_id'], scene_id=scene['scene_id'],
        condition=PromptCondition.SYMBOLIC_RULE if image is None else PromptCondition.IMAGE_RULE,
        question_kind='outcome', variant=spec['condition'], split=PHASE, problem_type=spec['stratum'],
        outcome=spec['gold_semantic'], net_torque=scene['features']['net_torque'],
        prompt=prompt_for(scene, spec['condition'], spec['semantics']), image_path=image,
        labels=LABELS, option_text=tuple(SEMANTIC_TEXT[s] for s in spec['semantics']),
        option_semantics=tuple(spec['semantics']), gold_label=spec['gold_label'], gold_semantic=spec['gold_semantic'])


def freeze_inputs():
    global SCENES, SPECS, SPEC_BY_ID, AUDIT_IDS
    SCENES = {s['scene_id']: s for f in FAMILIES for s in f['scenes']}
    SPECS = list(trial_specs(FAMILIES))
    SPEC_BY_ID = {s['trial_id']: s for s in SPECS}
    assert len(SPEC_BY_ID) == len(SPECS)
    AUDIT_IDS = generation_audit_ids(FAMILIES)
    inputs = []
    for spec in SPECS:
        ex = make_example(spec)
        inputs.append(dict(spec, prompt=ex.prompt, prompt_sha256=hashlib.sha256(ex.prompt.encode()).hexdigest(),
                           image_filename=ex.image_path.name if ex.image_path else None,
                           image_sha256=IMAGE_MANIFEST[ex.image_path.name]['sha256'] if ex.image_path else None,
                           generation_audit=spec['trial_id'] in AUDIT_IDS))
    freeze(SHARED_ROOT / 'inputs.json', inputs)
    freeze(SHARED_ROOT / 'generation_audit_plan.json', sorted(AUDIT_IDS))
    print(f'{len(FAMILIES)} physical families; {len(SCENES)} scene presentations; '
          f'{len(SPECS)} scored forwards + {len(AUDIT_IDS)} short direct-generation checks per model.')
    display(pd.DataFrame([dict(stratum=s, families=sum(f['stratum']==s for f in FAMILIES)) for s in STRATA]))
    previews = []
    for stratum in STRATA:
        family = next(f for f in FAMILIES if f['stratum'] == stratum)
        scene = next(s for s in family['scenes'] if s['role']=='base' and not s['mirror'])
        with Image.open(image_for(scene['scene_id'], 'image')) as image:
            previews.append(image.convert('RGB').resize((448, 224)))
    sheet = Image.new('RGB', (896, 3 * 250), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (label, image) in enumerate(zip(STRATA, previews)):
        x, y = i % 2 * 448, i // 2 * 250
        sheet.paste(image, (x, y + 22))
        draw.text((x + 8, y + 4), label, fill='black')
    sheet.save(SHARED_ROOT / 'scene_preview.png')
    display(sheet)
    example = next(s for s in SPECS if s['condition']=='image_full')
    print('Example complete-variable prompt (no torque or correct answer supplied):\n')
    print(make_example(example).prompt)


def tensor_identity(inputs):
    result = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            x = value.detach().cpu().contiguous()
            result[key] = dict(shape=list(x.shape), dtype=str(x.dtype),
                               sha256=hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest())
        else:
            result[key] = str(value)
    return result


def init_runtime():
    global RUNTIME_IDENTITY, RUNTIME_SHA
    import PIL
    import transformers
    assert adapter.metadata['resolved_revision'] == SPEC['revision']
    assert adapter.metadata['dtype'] == 'bfloat16'
    assert not getattr(adapter.model, 'is_quantized', False), 'Use native BF16, not a quantized checkpoint.'
    assert not any(str(v) in ('cpu', 'disk') for v in getattr(adapter.model, 'hf_device_map', {}).values()), 'The full BF16 model must fit on GPU.'
    adapter.model.eval()
    torch.manual_seed(SEED)
    np.random.seed(SEED % (2**32))
    torch.backends.cuda.matmul.allow_tf32 = False
    plans = {}
    for condition in CONDITIONS:
        ex = make_example(next(s for s in SPECS if s['condition'] == condition))
        plan = adapter._candidate_plan(ex)
        assert plan is not None, 'Expected exact single-token candidates after a common prefix.'
        prefix, tokens = plan
        assert set(tokens) == set(LABELS) and len(set(tokens.values())) == 3
        plans[condition] = dict(prefix=list(prefix), token_by_label=tokens)
    assert len({digest(v) for v in plans.values()}) == 1, 'Candidate scoring differs between conditions.'
    template = adapter.processor.chat_template
    tokenizer_template = adapter.processor.tokenizer.chat_template
    processor_config = adapter.processor.to_dict() if hasattr(adapter.processor, 'to_dict') else {}
    processor_config = json.loads(json.dumps(processor_config, default=str))
    image_processor = getattr(adapter.processor, 'image_processor', None)
    image_config = image_processor.to_dict() if image_processor is not None else None
    image_config = json.loads(json.dumps(image_config, default=str))
    RUNTIME_IDENTITY = dict(model_key=MODEL_KEY, model_metadata=adapter.metadata,
        gpu=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        torch=torch.__version__, transformers=transformers.__version__, pillow=PIL.__version__,
        cuda=torch.version.cuda, numpy=np.__version__, pandas=pd.__version__,
        implementation=IMPLEMENTATION_SHA256, dtype='bfloat16', batch_size=1,
        thinking=False, max_new_tokens_generation=16, candidate_plans=plans,
        toolkit={str(p.relative_to(TOOLKIT_ROOT)):file_hash(p) for p in sorted((TOOLKIT_ROOT/'src').rglob('*.py'))},
        protocol_sha256=file_hash(RUN_ROOT/'protocol.json'), phase_protocol_sha256=file_hash(SHARED_ROOT/'phase_protocol.json'),
        support_sha256=file_hash(SHARED_ROOT/'support.json'), inputs_sha256=file_hash(SHARED_ROOT/'inputs.json'),
        processor_config=processor_config, image_processor_config=image_config,
        chat_template=template, tokenizer_chat_template=tokenizer_template)
    RUNTIME_SHA = freeze(PHASE_ROOT/'runtime_identity.json', RUNTIME_IDENTITY)
    print('Frozen candidate plans:', plans['image'])
    print('Direct-answer scoring; hidden reasoning disabled. Results do not measure a reasoning-enabled ceiling.')


def score_one(spec):
    ex = make_example(spec)
    prefix, tokens = adapter._candidate_plan(ex)
    raw_inputs = adapter._template([ex.messages()], add_generation_prompt=True)
    inputs = adapter._append_text_prefix(adapter._to_device(raw_inputs), prefix)
    identity = tensor_identity(inputs)
    present_pixels = any('pixel' in k and torch.is_tensor(v) and v.numel() > 0 for k, v in inputs.items())
    assert present_pixels == (ex.image_path is not None), 'Image/text-only processor input mismatch.'
    started = time.monotonic()
    with torch.inference_mode():
        outputs = adapter.model(**inputs, use_cache=False)
        logits = outputs.logits[0, -1].float()
        logp = torch.log_softmax(logits, dim=-1)
        values = [float(logp[tokens[label]].item()) for label in LABELS]
    assert np.isfinite(values).all()
    predicted = LABELS[int(np.argmax(values))]
    semantic = ex.label_to_semantic[predicted]
    correct = int(predicted == ex.gold_label)
    gold_index = LABELS.index(ex.gold_label)
    margin = values[gold_index] - max(v for i, v in enumerate(values) if i != gold_index)
    sorted_scores = sorted(values, reverse=True)
    row = dict(spec, predicted_label=predicted, predicted_semantic=semantic, correct=correct,
        candidate_logprobs=dict(zip(LABELS, values)), gold_margin=margin,
        predicted_balance=int(semantic == 'balanced'), wrong_follows_weight=int(not correct and semantic == spec['weight_outcome']),
        top_score_tie=sorted_scores[0] == sorted_scores[1],
        prompt_sha256=hashlib.sha256(ex.prompt.encode()).hexdigest(),
        input_sha256=digest(identity), prepared_inputs=identity,
        input_token_ids=inputs['input_ids'][0].detach().cpu().tolist(),
        input_token_count=int(inputs['attention_mask'].sum()),
        scoring_mode='single_token_next_token', candidate_prefix=list(prefix), candidate_tokens=tokens,
        scoring_seconds=time.monotonic()-started, generation_audited=spec['trial_id'] in AUDIT_IDS,
        generated_label=None, raw_response=None, parsed_response=None, generated_correct=None,
        recorded_at_utc=now_utc())
    del outputs, logits, logp, inputs, raw_inputs
    if row['generation_audited']:
        generated = adapter.generate_batch([ex])[0]
        label = strict_label(generated.parsed_content)
        row.update(generated_label=label, raw_response=generated.raw_response,
                   parsed_response=generated.parsed_content,
                   generated_correct=int(label == ex.gold_label) if label is not None else None)
    return row


def run_scoring():
    folder = PHASE_ROOT/'families'
    folder.mkdir(parents=True, exist_ok=True)
    grouped = {f['family_id']: [s for s in SPECS if s['family_id'] == f['family_id']] for f in FAMILIES}
    all_rows = []
    bar = tqdm(total=len(SPECS), desc=f'{MODEL_KEY} / {PHASE}: paired inputs', unit='forward')
    try:
        for family in FAMILIES:
            fid = family['family_id']
            path = folder/(fid+'.json')
            expected = {s['trial_id']:s for s in grouped[fid]}
            rows = []
            identity = digest([RUNTIME_SHA, fid])
            if path.exists():
                old = read_json(path)
                assert old['identity'] == identity, 'Different runtime or input definition; use a new RUN_ID.'
                rows = old['rows']
                assert old['rows_sha256'] == digest(rows), 'Cached result rows changed: ' + str(path)
                assert len({r['trial_id'] for r in rows}) == len(rows)
                for row in rows:
                    assert row['trial_id'] in expected
                    assert all(row[k] == v for k, v in expected[row['trial_id']].items())
                bar.update(len(rows))
            done = {r['trial_id'] for r in rows}
            # Fixed pseudo-random ordering prevents condition/order from being confounded.
            order = sorted(expected, key=lambda key: digest(['trial_order', SEED, key]))
            try:
                for tid in order:
                    if tid in done:
                        continue
                    row = score_one(expected[tid])
                    rows.append(row)
                    bar.update(1)
                    bar.set_postfix_str(expected[tid]['condition'])
                    if len(rows) % 6 == 0:
                        save_json(path, dict(identity=identity, family_id=fid, rows=rows, rows_sha256=digest(rows)))
            finally:
                # Save completed inputs even when an error or interruption occurs.
                save_json(path, dict(identity=identity, family_id=fid, rows=rows, rows_sha256=digest(rows)))
            assert len(rows) == len(expected)
            all_rows.extend(rows)
    finally:
        bar.close()
    return all_rows


def complete_analysis(rows):
    summary, tests = analyze_rows(rows, FAMILIES, ANALYSIS_DIR, MODEL_KEY, PHASE)
    display(summary[['scope','condition','n_families','mean','lower','upper']].round(4))
    display(tests.round(4))
    agreement = pd.read_csv(ANALYSIS_DIR/'generation_scoring_agreement.csv')
    display(agreement)
    assert sum(r['generation_audited'] for r in rows) == len(AUDIT_IDS)
    save_json(PHASE_ROOT/'completion.json', dict(status='complete', phase=PHASE, model=MODEL_KEY,
        n_families=len(FAMILIES), n_scored=len(rows), n_generation_checks=len(AUDIT_IDS),
        implementation=IMPLEMENTATION_SHA256, support_sha256=file_hash(SHARED_ROOT/'support.json'),
        runtime_sha256=file_hash(PHASE_ROOT/'runtime_identity.json'),
        family_files={p.name:file_hash(p) for p in sorted((PHASE_ROOT/'families').glob('*.json'))},
        scope='Candidate-scored direct answers; generation agreement is reported separately.', completed_at_utc=now_utc()))
    if PHASE == 'development':
        print('Development completed. Its estimates are exploratory; run the other model before freezing validation.')
    else:
        print('Validation complete. Cross-model primary correction requires both models.')
    return summary, tests


def combine_models():
    frames = []
    for key in MODEL_SPECS:
        root = RUN_ROOT/'models'/key/PHASE
        if not (root/'completion.json').exists():
            print('Cross-model comparison pending:', key)
            return
        complete = read_json(root/'completion.json')
        assert complete['status'] == 'complete' and complete['implementation'] == IMPLEMENTATION_SHA256
        assert complete['support_sha256'] == file_hash(SHARED_ROOT/'support.json')
        frame = pd.read_csv(root/'analysis/primary_comparisons.csv')
        assert len(frame) == 7
        frames.append(frame)
    both = pd.concat(frames, ignore_index=True)
    both['holm_across_models_14_tests'] = holm(both.p_value)
    both['inference_status'] = 'development_descriptive_only' if PHASE=='development' else 'joint_confirmatory_14_tests'
    both.to_csv(ANALYSIS_DIR/'cross_model_primary_comparisons.csv', index=False)
    save_json(SHARED_ROOT/'both_models_complete.json', dict(phase=PHASE, implementation=IMPLEMENTATION_SHA256,
        support_sha256=file_hash(SHARED_ROOT/'support.json'), n_families=len(FAMILIES)))
    display(both.round(4))


def export_support():
    """Works without model weights after the setup/definition cells have run."""
    completion_path = PHASE_ROOT/'completion.json'
    complete = completion_path.exists() and read_json(completion_path).get('status') == 'complete'
    sources = {}
    for name in ('protocol.json', 'historical_exclusion.json'):
        p = RUN_ROOT/name
        if p.exists():
            sources[name] = p
    for p in SHARED_ROOT.rglob('*'):
        if p.is_file() and not p.name.endswith('.tmp'):
            sources['shared/'+PHASE+'/'+p.relative_to(SHARED_ROOT).as_posix()] = p
    for p in PHASE_ROOT.rglob('*'):
        if p.is_file() and not p.name.endswith('.tmp'):
            sources['models/'+MODEL_KEY+'/'+PHASE+'/'+p.relative_to(PHASE_ROOT).as_posix()] = p
    manifest = dict(schema=SCHEMA, model=MODEL_KEY, phase=PHASE,
        status='complete' if complete else 'partial_not_complete',
        created_at_utc=now_utc(), files={name:file_hash(p) for name,p in sources.items()})
    output = RUN_ROOT/f'VisualTorque_VisualInformationSupport_{MODEL_KEY}_{PHASE}_Output.zip'
    tmp = output.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, path in sources.items():
            archive.write(path, name)
        archive.writestr('export_manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(tmp) as archive:
        assert archive.testzip() is None
    tmp.replace(output)
    local = Path('/content')/output.name if Path('/content').is_dir() else output
    if local.resolve() != output.resolve():
        shutil.copyfile(output, local)
    print('Export status:', manifest['status'], '\nDrive:', output, '\nDownload:', local)
    try:
        from google.colab import files
        files.download(str(local))
    except ImportError:
        pass
    return str(output)
