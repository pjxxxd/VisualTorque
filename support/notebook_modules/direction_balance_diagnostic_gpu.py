"""Diagnostic-specific glue. Scoring/resume helpers are embedded from the parent notebook."""
import gc
import time
import shutil
from datetime import datetime, timezone
from tqdm.auto import tqdm
from visual_torque_toolkit.behavior.data import BehaviorExample
from visual_torque_toolkit.behavior.schema import PromptCondition


def make_example(spec):
    inp=INPUT_BY_ID[spec['trial_id']]
    scene=SCENES[inp['scene_id']]
    path=IMAGE_DIR/inp['image_filename'] if inp['image_filename'] else None
    option_text=[('balanced (neither side tips)' if semantic=='balanced' and inp['wording']!='original'
                  else SEMANTIC_TEXT[semantic]) for semantic in inp['semantics']]
    return BehaviorExample(example_id=inp['trial_id'],question_id=inp['trial_id'],scene_id=inp['scene_id'],
        condition=PromptCondition.SYMBOLIC_RULE if path is None else PromptCondition.IMAGE_RULE,
        question_kind='outcome',variant=inp['condition']+'__'+inp['wording'],split=PHASE,
        problem_type=inp['stratum'],outcome=inp['gold_semantic'],net_torque=scene['features']['net_torque'],
        prompt=inp['prompt'],image_path=path,labels=LABELS,option_text=tuple(option_text),
        option_semantics=tuple(inp['semantics']),gold_label=inp['gold_label'],gold_semantic=inp['gold_semantic'])


def init_diagnostic_runtime():
    init_runtime()
    prior=SOURCE_SNAPSHOTS[MODEL_KEY]['runtime']
    differences={}
    for name in ('model_metadata','gpu','torch','transformers','pillow','cuda','numpy','pandas',
                 'dtype','batch_size','thinking','toolkit','processor_config','image_processor_config',
                 'chat_template','tokenizer_chat_template'):
        differences[name]=dict(identical=RUNTIME_IDENTITY.get(name)==prior.get(name))
    freeze(PHASE_ROOT/'source_environment_comparison.json',differences)
    changed=[name for name,value in differences.items() if not value['identical']]
    print('Environment fields differing from the development archive:',changed or 'none')
    print('Archive replay will record input equality, score drift and choice agreement; no arbitrary score-parity assertion.')


def show_inputs():
    examples=[]
    for group in GROUPS:
        f=next(f for f in FAMILIES if f['stratum']==group)
        for mirror in (False,True):
            s=next(s for s in f['scenes'] if s['mirror']==mirror)
            filename=s['scene_id']+'__ids.png'
            with Image.open(IMAGE_DIR/filename) as im:
                examples.append((f'{group} / mirror={mirror}',im.convert('RGB').resize((448,224))))
    sheet=Image.new('RGB',(896,4*250),'white')
    draw=ImageDraw.Draw(sheet)
    for i,(name,im) in enumerate(examples):
        x,y=(i%2)*448,(i//2)*250
        sheet.paste(im,(x,y+22));draw.text((x+8,y+4),name,fill='black')
    sheet.save(SHARED_ROOT/'control_preview.png')
    display(sheet)
    print('600 scored inputs = 576 simple controls + 24 exact archived-input replays.')
    print('120 fixed direct-generation checks; no validation data or old results are modified.')
    example=next(x for x in INPUTS if x['condition']=='text_full' and x['wording']=='neutral_question')
    print(example['prompt'])


def finish_diagnostic(rows):
    summary=summarize_diagnostic(rows,INPUTS,SOURCE_SNAPSHOTS,MODEL_KEY,ANALYSIS_DIR)
    assert len(rows)==600 and sum(r['generation_audited'] for r in rows)==120
    display(summary[['scope','stratum','condition','wording','n_families','correct','answer_left','answer_balanced']].round(4))
    agreement=pd.read_csv(ANALYSIS_DIR/'generation_agreement.csv')
    display(agreement)
    replay=pd.read_csv(ANALYSIS_DIR/'archived_replay_comparison.csv')
    display(replay[['condition','prepared_inputs_identical','same_choice','max_centered_score_difference']])
    flags=dict(replay_input_mismatches=int((~replay.prepared_inputs_identical).sum()),
        replay_candidate_plan_mismatches=int((~replay.candidate_plan_identical).sum()),
        replay_changed_choices=int((~replay.same_choice).sum()),
        generation_invalid=int((agreement.n-agreement.valid).sum()),
        generation_disagreements=int((agreement.valid-agreement.agrees).sum()),
        inference_status='descriptive_diagnostic_only',validation_authorized_by_software=False)
    save_json(ANALYSIS_DIR/'measurement_checks.json',flags)
    save_json(PHASE_ROOT/'completion.json',dict(status='complete',model=MODEL_KEY,phase=PHASE,
        n_scored=600,n_generation_checks=120,implementation=IMPLEMENTATION_SHA256,
        support_sha256=file_hash(SHARED_ROOT/'support.json'),inputs_sha256=file_hash(SHARED_ROOT/'inputs.json'),
        runtime_sha256=file_hash(PHASE_ROOT/'runtime_identity.json'),
        family_files={p.name:file_hash(p) for p in sorted((PHASE_ROOT/'families').glob('*.json'))},
        scope='Simple controls and intentional archive replays; not held-out validation.',completed_at_utc=now_utc()))
    print('Diagnostic complete. Review the simple controls, wording changes and replay checks before deciding on validation.')
    return summary


def combine_diagnostics():
    frames=[]
    for key in MODEL_SPECS:
        root=RUN_ROOT/'models'/key/PHASE
        complete=root/'completion.json'
        if not complete.is_file():
            print('Combined diagnostic table pending:',key)
            return
        old=read_json(complete)
        assert old['status']=='complete' and old['implementation']==IMPLEMENTATION_SHA256
        assert old['inputs_sha256']==file_hash(SHARED_ROOT/'inputs.json')
        frames.append(pd.read_csv(root/'analysis/diagnostic_summary.csv').assign(model=key))
    combined=pd.concat(frames,ignore_index=True)
    combined.to_csv(ANALYSIS_DIR/'cross_model_diagnostic_summary.csv',index=False)
    print('Both models complete; the combined descriptive table is included in this model’s ZIP.')
