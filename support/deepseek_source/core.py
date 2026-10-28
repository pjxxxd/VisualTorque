"""Protocol orchestration; established numerical kernels are vendored by build_bundle.py."""
import argparse
import ast
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

HERE = Path(__file__).resolve().parent
VENDOR = HERE / 'kernels'


def load_kernels(namespace, names):
    for name in names:
        path = VENDOR / name
        exec(compile(path.read_text(encoding='utf-8'), str(path), 'exec'), namespace)


def base_namespace(project, run_id):
    sys.path.insert(0, str(HERE / 'toolkit' / 'src'))
    n = dict(__name__='deepseek_study')
    load_kernels(n, ['followup_core.py', 'intervention_prediction_core.py',
                     'intervention_prediction_io.py', 'readout_edit_math.py',
                     'aligned_readout_core.py', 'independent_core.py'])
    from adapter import MODEL_ID, MODEL_REVISION, SOURCE_REVISION, LAYER_PATH, source_identity
    from tqdm.auto import tqdm
    import torch
    n.update(PROJECT_ROOT=Path(project).resolve(), TOOLKIT_ROOT=HERE/'toolkit',
        MODEL_ID=MODEL_ID, MODEL_REVISION=MODEL_REVISION, MODEL_KEY='deepseek16b',
        MODEL_DTYPE='bfloat16', LAYER_PATH=LAYER_PATH, EXPECTED_LAYER_COUNT=27,
        RANDOM_SEEDS=(0,1,2,3), SEED=2026091401, RUN_ID=run_id,
        torch=torch, tqdm=tqdm, display=lambda x: print(x.to_string(index=False) if hasattr(x,'to_string') else x),
        cell_stage=print, SPEC=dict(content_layers=(20,26), physical_layers=(20,26)),
        CAPTURE_LAYERS=(20,26), NORM_RELATIVE_TOLERANCE=.02)
    assert run_id and all(c.isalnum() or c in '_-' for c in run_id)
    sources = source_identity(HERE)
    n['IMPLEMENTATION_SHA256'] = n['digest'](sources)
    n['ROOT'] = n['PROJECT_ROOT']/'results/deepseek_vl2_small'/run_id
    n['ROOT'].mkdir(parents=True, exist_ok=True)
    n['freeze'](n['ROOT']/'suite_identity.json', dict(model=MODEL_ID, revision=MODEL_REVISION,
        source_revision=SOURCE_REVISION, implementation=n['IMPLEMENTATION_SHA256'],
        source_files=sources, layer_index='zero_based',dtype='bfloat16',batch_size=1,
        prediction_layers=[20,26], readout_layers=list(range(27)),
        readout_alphas=[.01,.1,1.,10.,100.,1000.], repair_upstream_layers=[10,13,16,19],
        repair_semantic_layer=20, repair_symbol_layer=23,repair_dose_factors=[.5,1.],
        repair_candidate_rule='maximum development specificity; save eligibility separately; no test reselection'))
    return n


def runtime(n):
    import torch, transformers, numpy, pandas, scipy, sklearn, platform
    from adapter import SOURCE_REVISION
    return dict(torch=torch.__version__, transformers=transformers.__version__,
        numpy=numpy.__version__,pandas=pandas.__version__,scipy=scipy.__version__,
        sklearn=sklearn.__version__,python=platform.python_version(),cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),gpu_capability=list(torch.cuda.get_device_capability(0)),
        source_revision=SOURCE_REVISION,checkpoint=n['MODEL_REVISION'],
        layer_path=n['LAYER_PATH'],dtype='bfloat16',batch_size=1,
        packages={k:importlib.metadata.version(k) for k in ['timm','xformers','accelerate','Pillow']},
        tf32=False,cudnn_benchmark=False,implementation=n['IMPLEMENTATION_SHA256'])


def load_model(n):
    from adapter import DeepSeekAdapter, output_hidden_tensor, replace_output_hidden, final_mask, score_records
    torch=n['torch']
    torch.manual_seed(n['SEED']); n['np'].random.seed(n['SEED'] % 2**32)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    adapter=DeepSeekAdapter()
    n.update(adapter=adapter, DECODER_LAYERS=adapter.layers,
        output_hidden_tensor=output_hidden_tensor,replace_output_hidden=replace_output_hidden,
        final_mask=final_mask,score_records=score_records,prepare_examples=adapter.prepare_examples)
    load_kernels(n, ['followup_gpu.py','intervention_prediction_gpu.py'])
    n['current_runtime_identity']=lambda: runtime(n)
    n['RUNTIME_IDENTITY']=runtime(n)
    n['freeze'](n['ROOT']/'runtime_identity.json', n['RUNTIME_IDENTITY'])
    return adapter


def verify_scoring_path(n, example, destination):
    """A fixed technical check, not a selection experiment or a scientific result."""
    np=n['np']; previous=n['CAPTURE_LAYERS'];n['CAPTURE_LAYERS']=(20,26)
    try:
        clean,h=n['predictive_forward'](example)
        noop,hn=n['predictive_forward'](example,layer=20,payload=np.zeros_like(h[20]),
            expected_pre=h[20],expected_input=clean['input_sha256'])
        assert np.array_equal(h[20],hn[20])
        error=float(np.max(np.abs(np.array(clean['candidate_logprobs'])-noop['candidate_logprobs'])))
        assert error<=1e-6,'Zero-edit replay differs; do not collect paired experimental results in this runtime.'
        tail=n['adapter'].tail_scores(h[26],example.labels)
        audit=dict(status='technical_check_passed_not_scientific_evidence',example=n['example_stamp'](example),
            noop_score_error=error,tail_max_abs_error=float(np.max(np.abs(tail-n['center'](clean['candidate_logprobs'])))),
            plans={','.join(k):dict(prefix=list(v[0]),tokens=v[1]) for k,v in n['adapter'].plans.items()},
            prepared_input_sha256=clean['input_sha256'])
        n['save_json'](destination,audit)
        print('Scoring/no-op check:',audit['status'],'tail error:',audit['tail_max_abs_error'])
    finally:n['CAPTURE_LAYERS']=previous


def dataset(n):
    project=n['PROJECT_ROOT']
    candidates=[project/'data/dataset_v1',project/'dataset_v1']
    found=next((p for p in candidates if (p/'scenes.jsonl').is_file()),None)
    if found is None:
        archive=project/'data/dataset_v1.zip'
        assert archive.is_file(), f'Missing Dataset v1 folder or archive: {archive}'
        dest=n['ROOT']/'dataset'
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                target=(dest/member.filename).resolve()
                assert target.is_relative_to(dest.resolve())
                if not target.is_file():z.extract(member,dest)
        founds=list(dest.rglob('scenes.jsonl'))
        assert len(founds)==1
        found=founds[0].parent
    for name in ['scenes.jsonl','questions.jsonl','splits.json']:
        assert (found/name).is_file(), found/name
    n['DATASET_DIR']=found
    return found


def fit_examples(n):
    from collections import Counter,defaultdict
    from visual_torque_toolkit.behavior import BehaviorRunConfig,PromptCondition,load_behavior_examples
    ds=dataset(n)
    manifest=n['PROJECT_ROOT']/'results/behavior_comparison_v3/activation_pilot_manifest.jsonl'
    rows=n['read_jsonl'](manifest)
    assert len(rows)==254 and Counter(r['split'] for r in rows)=={'probe_train':235,'probe_dev':19}
    ids={str(r['scene_id']):r['split'] for r in rows}
    cfg=BehaviorRunConfig(dataset_dir=ds,output_dir=n['ROOT']/'unused',model_id=n['MODEL_ID'],
        model_revision=n['MODEL_REVISION'],conditions=(PromptCondition.IMAGE_RULE,),question_kinds=('outcome',),
        splits=('probe_train','probe_dev','probe_test'),batch_size=1,generation_every=0,seed=20260820)
    groups=defaultdict(list)
    for ex in load_behavior_examples(cfg):groups[str(ex.scene_id)].append(ex)
    assert all(len(groups[s])==6 and {e.split for e in groups[s]}=={ids[s]} for s in ids)
    n.update(FIT_SPLITS=ids,FIT_EXAMPLES={s:sorted(groups[s],key=lambda e:e.variant) for s in sorted(ids)},
        ALL_EXAMPLES=groups,TRAIN_IDS=sorted(s for s in ids if ids[s]=='probe_train'),
        DEV_IDS=sorted(s for s in ids if ids[s]=='probe_dev'))
    return manifest


def preflight(n):
    manifest=fit_examples(n)
    required=[manifest,n['PROJECT_ROOT']/'results/cross_model_readout_correction_v2/shared_304_manifest.json',
        n['PROJECT_ROOT']/'results/cross_model_readout_correction_v2/shared_1007_scene_ids.json']
    report=dict(model=n['MODEL_ID'], full_model_run=False, dataset=str(n['DATASET_DIR']),
        files=[dict(path=str(p),exists=p.is_file()) for p in required],
        cuda_available=n['torch'].cuda.is_available(), training_scenes=235,development_scenes=19)
    n['save_json'](n['ROOT']/'preflight.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    assert all(r['exists'] for r in report['files']), 'Restore the listed historical shared manifests first.'


def export(n):
    root=n['ROOT']; path=root.parent/(root.name+'_results.zip')
    files=[p for p in root.rglob('*') if p.is_file() and 'dataset' not in p.relative_to(root).parts and '.tmp' not in p.name]
    with zipfile.ZipFile(path.with_suffix('.tmp.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(root))
    path.with_suffix('.tmp.zip').replace(path)
    print('Saved results (may be partial; check completion markers):',path)
    return path
