"""Validation-specific inference glue; parent scoring and checkpoint logic are reused."""
import gc
import shutil
import time
from datetime import datetime, timezone
from tqdm.auto import tqdm
from visual_torque_toolkit.behavior.data import BehaviorExample
from visual_torque_toolkit.behavior.schema import PromptCondition


def make_example(spec):
    inp=INPUT_BY_ID[spec['trial_id']];scene=SCENES[inp['scene_id']]
    path=IMAGE_DIR/inp['image_filename'] if inp['image_filename'] else None
    options=tuple('balanced (neither side tips)' if s=='balanced' and inp['wording']=='neutral_question'
                  else SEMANTIC_TEXT[s] for s in inp['semantics'])
    return BehaviorExample(example_id=inp['trial_id'],question_id=inp['trial_id'],scene_id=inp['scene_id'],
        condition=PromptCondition.SYMBOLIC_RULE if path is None else PromptCondition.IMAGE_RULE,
        question_kind='outcome',variant=inp['condition']+'__'+inp['wording'],split=PHASE,
        problem_type=inp['stratum'],outcome=inp['gold_semantic'],net_torque=scene['features']['net_torque'],
        prompt=inp['prompt'],image_path=path,labels=LABELS,option_text=options,option_semantics=tuple(inp['semantics']),
        gold_label=inp['gold_label'],gold_semantic=inp['gold_semantic'])


def init_validation_runtime():
    init_runtime()
    prior=SOURCE_SNAPSHOT['evidence']['diagnostic'][MODEL_KEY]['runtime']
    fields=('model_metadata','gpu','torch','transformers','pillow','cuda','numpy','pandas',
        'dtype','batch_size','thinking','toolkit','processor_config','image_processor_config',
        'chat_template','tokenizer_chat_template')
    differences={name:dict(identical=RUNTIME_IDENTITY.get(name)==prior.get(name)) for name in fields}
    freeze(PHASE_ROOT/'source_environment_comparison.json',differences)
    changed=[k for k,v in differences.items() if not v['identical']]
    print('Environment changes relative to the diagnostic archive:',changed or 'none')
    critical={'transformers','pillow','dtype','batch_size','thinking','toolkit','processor_config',
              'image_processor_config','chat_template','tokenizer_chat_template'}
    assert not critical.intersection(changed), 'Restore the recorded preprocessing/scoring configuration before validation: '+str(sorted(critical.intersection(changed)))


def show_validation_inputs():
    examples=[]
    for group in STRATA:
        f=next(f for f in FAMILIES if f['stratum']==group)
        s=next(s for s in f['scenes'] if s['role']=='base' and not s['mirror'])
        with Image.open(IMAGE_DIR/(s['scene_id']+'__ids.png')) as im:
            examples.append((group,im.convert('RGB').resize((448,224))))
    sheet=Image.new('RGB',(896,750),'white');draw=ImageDraw.Draw(sheet)
    for i,(label,im) in enumerate(examples):
        x,y=(i%2)*448,(i//2)*250
        sheet.paste(im,(x,y+22));draw.text((x+8,y+4),label,fill='black')
    sheet.save(SHARED_ROOT/'scene_preview.png');display(sheet)
    print('Frozen: 96 physical families; 240 scene presentations; 8,640 scores and 240 short generation checks per model.')
    print('Known excluded mirror-canonical geometries:',len(EXCLUDED))
    print('Original wording is primary. Neutral wording is a fixed parallel comparison, not selected by validation accuracy.')
    print('Family-by-family checkpointing is enabled. Re-run from the top after an interruption.')


def finish_validation(rows):
    summary,primary,secondary=analyze_validation(rows,INPUTS,FAMILIES,ANALYSIS_DIR,MODEL_KEY)
    assert len(rows)==8640 and sum(r['generation_audited'] for r in rows)==240
    display(summary[summary.scope=='all'][['condition','wording','n_families','mean','lower','upper','answer_left','answer_balanced']].round(4))
    display(primary.round(4));display(secondary.round(4))
    agreement=pd.read_csv(ANALYSIS_DIR/'generation_agreement.csv');display(agreement)
    save_json(ANALYSIS_DIR/'measurement_checks.json',dict(
        invalid_generations=int((agreement.n-agreement.valid).sum()),
        disagreements_among_valid=int((agreement.valid-agreement.agrees).sum()),
        checked_generations=int(agreement.n.sum()),
        meaning='Checks describe scoring/short-generation agreement; no post-hoc input exclusions or protocol changes.'))
    save_json(PHASE_ROOT/'completion.json',dict(status='complete',phase=PHASE,model=MODEL_KEY,n_families=96,
        n_scored=8640,n_generation_checks=240,implementation=IMPLEMENTATION_SHA256,
        support_sha256=file_hash(SHARED_ROOT/'support.json'),inputs_sha256=file_hash(SHARED_ROOT/'inputs.json'),
        protocol_sha256=file_hash(RUN_ROOT/'protocol.json'),runtime_sha256=file_hash(PHASE_ROOT/'runtime_identity.json'),
        family_files={p.name:file_hash(p) for p in sorted((PHASE_ROOT/'families').glob('*.json'))},
        analysis_files={p.name:file_hash(p) for p in sorted(ANALYSIS_DIR.glob('*')) if p.is_file() and not p.name.startswith('cross_model')},
        completed_at_utc=now_utc(),scope='Family-disjoint validation under the frozen, previously examined direct-answer protocol.'))
    print('This model is complete. Keep the planned sample size and protocol unchanged; run the other model for joint correction.')
    return summary


def combine_validation_models():
    primary=[];secondary=[];summaries=[]
    for key in MODEL_SPECS:
        root=RUN_ROOT/'models'/key/PHASE
        if not (root/'completion.json').exists():
            print('Joint analysis pending:',key)
            return
        complete=read_json(root/'completion.json')
        assert complete['status']=='complete' and complete['implementation']==IMPLEMENTATION_SHA256
        assert complete['inputs_sha256']==file_hash(SHARED_ROOT/'inputs.json')
        assert complete['protocol_sha256']==file_hash(RUN_ROOT/'protocol.json')
        for name,h in complete['family_files'].items():
            assert file_hash(root/'families'/name)==h,'Recorded family results changed.'
        for name,h in complete['analysis_files'].items():
            assert file_hash(root/'analysis'/name)==h,'Recorded analysis changed.'
        a=pd.read_csv(root/'analysis/primary_original_comparisons.csv')
        b=pd.read_csv(root/'analysis/secondary_wording_comparisons.csv')
        assert len(a)==7 and len(b)==4
        primary.append(a);secondary.append(b)
        summaries.append(pd.read_csv(root/'analysis/accuracy_and_answer_bias.csv'))
    p=pd.concat(primary,ignore_index=True);s=pd.concat(secondary,ignore_index=True)
    p['holm_primary_across_models_14']=base.holm(p.p_value)
    p['inference_status']='primary_family_holm_14_tests'
    s['holm_secondary_across_models_8']=base.holm(s.p_value)
    s['inference_status']='secondary_family_holm_8_tests_separate_from_primary'
    p.to_csv(ANALYSIS_DIR/'cross_model_primary_original.csv',index=False)
    s.to_csv(ANALYSIS_DIR/'cross_model_secondary_wording.csv',index=False)
    pd.concat(summaries,ignore_index=True).to_csv(ANALYSIS_DIR/'cross_model_accuracy_and_answer_bias.csv',index=False)
    save_json(SHARED_ROOT/'both_models_complete.json',dict(implementation=IMPLEMENTATION_SHA256,
        inputs_sha256=file_hash(SHARED_ROOT/'inputs.json'),primary_family_tests=14,secondary_family_tests=8,
        note='The two multiplicity families are separate; there is no claimed single global 22-test error guarantee.'))
    display(p.round(4));display(s.round(4))
    print('Both models completed. Joint tables are included in this model’s export; neither model is selected by its outcome.')
