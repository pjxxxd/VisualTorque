"""CPU protocol, narrow response grammar and analysis for archived Gemma inputs."""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile
import numpy as np
import pandas as pd
from visual_information_support_core import digest, file_hash, read_json, save_json, freeze, paired_stats, holm

SCHEMA = 'gemma-generation-recheck-v1'
MODEL_ID = 'google/gemma-4-12B-it'
REVISION = '707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7'
SOURCE_NAME = 'VisualTorque_Frozen_Visual_Validation_gemma12b_Output.zip'
SOURCE_SHA = 'fcb26b5ef9d55ab1f79565bb4905f4a3f42d67a847d6c81d874edbfa0c47dffd'
SOURCE_IMPLEMENTATION = 'e84045fd42cd1f058593cdbd45e07115e0339b491ea91c315ffdee99be1e75d4'
PREFIX = 'models/gemma12b/frozen_validation/'
WORDINGS = ('original', 'neutral_question')
STRATA = ('congruent', 'balanced', 'equal_weight_distance', 'weight_conflict', 'binding_swap')
LABELS = ('A', 'B', 'C')
PARSER_VERSION = 'complete-turn-single-letter-grammar-v1'
PARSER_RULES = {
    'strict': 'Completed turn containing exactly A, B or C after whitespace trimming.',
    'normalized': 'Strict reply, or exactly thought + closing channel + letter, closing channel + letter, or a complete thought span followed by a single letter. All require <turn|> and no trailing content.',
    'boundary': 'Channel-wrapped outputs are explicitly marked; recovery defines a revised measurement and is not proof of a normal final channel or disabled internal reasoning.',
    'invalid': 'No completed turn, multiple letters, extra answer prose, unclosed/unknown/nested channel markup, and any ambiguous continuation remain invalid. Never search for the last letter.',
}


def parse_response(raw):
    """Full-string grammar; do not erase arbitrary markup or search answer prose."""
    text = raw.strip()
    result = dict(strict_label=None, normalized_label=None, response_form='invalid',
                  reason=None, completed_turn=False, had_channel_wrapper=False)
    if text.count('<turn|>') != 1 or not text.endswith('<turn|>'):
        result['reason'] = 'missing_or_ambiguous_turn_end'
        return result
    body = text[:-len('<turn|>')].strip()
    result['completed_turn'] = True
    if body in LABELS:
        return dict(result, strict_label=body, normalized_label=body, response_form='direct_letter')
    observed = re.fullmatch(r'(thought\s*)?<channel\|>\s*([ABC])', body)
    if observed:
        return dict(result, normalized_label=observed.group(2), had_channel_wrapper=True,
            response_form='observed_thought_wrapper' if observed.group(1) else 'observed_closing_wrapper')
    structured = re.fullmatch(r'<\|channel>thought\s+(.*?)<channel\|>\s*([ABC])', body, flags=re.S)
    if structured and '<' not in structured.group(1) and '>' not in structured.group(1):
        return dict(result, normalized_label=structured.group(2), had_channel_wrapper=True,
                    response_form='closed_thought_span')
    result['reason'] = 'not_a_single_letter_under_fixed_grammar'
    return result


def choose_source(project_root, explicit=None):
    candidates = [Path(explicit)] if explicit else [
        Path(project_root)/'results/frozen_visual_validation_v1/main'/SOURCE_NAME,
        Path('/content')/SOURCE_NAME,
        Path(project_root)/SOURCE_NAME,
    ]
    for path in candidates:
        if path.is_file():
            assert file_hash(path) == SOURCE_SHA, f'Wrong source ZIP: {path}. Use the reviewed Gemma frozen-validation export.'
            return path
    raise FileNotFoundError('Gemma source ZIP not found. Set SOURCE_ZIP to its exact Drive path, or upload '+SOURCE_NAME+' to /content. No recursive search is performed.')


def prepare_source(path, shared_root):
    """One named ZIP, verified in memory; no broad extraction or history scan."""
    raw = Path(path).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA
    root = Path(shared_root); root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        manifest = json.loads(z.read('export_manifest.json'))
        assert manifest['model'] == 'gemma12b' and manifest['status'] == 'complete'
        assert len(z.namelist()) == len(set(z.namelist()))
        assert set(z.namelist()) == set(manifest['files']) | {'export_manifest.json'}
        for name, h in manifest['files'].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == h, name
        runtime = json.loads(z.read(PREFIX+'runtime_identity.json'))
        complete = json.loads(z.read(PREFIX+'completion.json'))
        support = json.loads(z.read('shared/frozen_validation/support.json'))
        inputs = [x for x in json.loads(z.read('shared/frozen_validation/inputs.json')) if x['condition']=='image']
        assert runtime['implementation'] == SOURCE_IMPLEMENTATION
        assert runtime['model_metadata']['resolved_revision'] == REVISION
        assert complete['n_scored'] == 8640 and len(support['families']) == 96
        assert len(inputs) == len({x['trial_id'] for x in inputs}) == 1440
        assert all(sum(x['wording']==w for x in inputs)==720 for w in WORDINGS)
        selected = {x['trial_id'] for x in inputs}; old = {}
        for name, h in complete['family_files'].items():
            payload_raw = z.read(PREFIX+'families/'+name)
            assert hashlib.sha256(payload_raw).hexdigest() == h
            payload = json.loads(payload_raw)
            assert digest(payload['rows']) == payload['rows_sha256']
            for row in payload['rows']:
                if row['trial_id'] in selected:
                    assert row['trial_id'] not in old
                    old[row['trial_id']] = row
        assert set(old) == selected
        image_hashes={}
        for x in inputs:
            assert hashlib.sha256(x['prompt'].encode()).hexdigest() == x['prompt_sha256']
            assert old[x['trial_id']]['prompt_sha256'] == x['prompt_sha256']
            name=x['image_filename']; assert Path(name).name == name
            assert name not in image_hashes or image_hashes[name]==x['image_sha256']
            image_hashes[name]=x['image_sha256']
        for name,image_hash in image_hashes.items():
            image_raw=z.read('shared/frozen_validation/images/'+name)
            assert hashlib.sha256(image_raw).hexdigest() == image_hash
            dest=root/'images'/name; dest.parent.mkdir(exist_ok=True)
            if dest.exists():
                assert file_hash(dest)==image_hash
            else:
                dest.write_bytes(image_raw)
        preflight=[x['trial_id'] for x in inputs if x['generation_audit']]
        assert len(preflight)==40
        snapshot=dict(source_zip_name=SOURCE_NAME,source_zip_sha256=SOURCE_SHA,
            source_runtime=runtime,source_completion=complete,
            source_protocol=json.loads(z.read('protocol.json')),
            scope='Post-hoc measurement recheck of 96 already observed physical families, not new held-out evidence.')
        for name,obj in [('source_snapshot.json',snapshot),('support.json',support),('inputs.json',inputs),
                         ('archived_image_rows.json',old),('preflight_ids.json',preflight)]:
            freeze(root/name,obj)
    return snapshot,support,inputs,old,preflight


def validate_row(row, inp, runtime_sha):
    assert row['runtime_sha256']==runtime_sha and row['trial_id']==inp['trial_id']
    assert row['prompt_sha256']==inp['prompt_sha256']
    assert row['input_definition_sha256']==digest(inp)
    assert row['parsed']==parse_response(row['raw_response'])
    assert len(row['generated_ids'])==len(row['generated_token_pieces'])==len(row['generation_steps'])
    for field in ('family_id','scene_id','stratum','role','mirror','mapping_index','wording','gold_label','gold_semantic','semantics'):
        assert row[field]==inp[field]


def load_rows(folder, inputs_by_id, runtime_sha):
    result={}
    for path in sorted(Path(folder).glob('*.json')):
        payload=read_json(path)
        assert payload['runtime_sha256']==runtime_sha
        assert payload['rows_sha256']==digest(payload['rows'])
        for row in payload['rows']:
            assert row['family_id']==payload['family_id']
            assert row['trial_id'] not in result
            validate_row(row,inputs_by_id[row['trial_id']],runtime_sha)
            result[row['trial_id']]=row
    return result


def checkpoint(folder, family_id, rows, runtime_sha):
    assert family_id and all(c.isalnum() or c in '_-' for c in family_id)
    selected=sorted((r for r in rows.values() if r['family_id']==family_id),key=lambda r:r['trial_id'])
    save_json(Path(folder)/(family_id+'.json'),dict(family_id=family_id,runtime_sha256=runtime_sha,
        rows=selected,rows_sha256=digest(selected)))


def preflight_report(rows, ids, old, outdir):
    assert set(ids)<=set(rows)
    data=[];traces=[]
    for tid in ids:
        r=rows[tid];a=old[tid];p=r['parsed']
        before=np.array([a['candidate_logprobs'][l] for l in LABELS])
        after=np.array([r['replayed_candidate_logprobs'][l] for l in LABELS])
        data.append(dict(trial_id=tid,wording=r['wording'],stratum=r['stratum'],
            raw_response=r['raw_response'],response_form=p['response_form'],normalized_label=p['normalized_label'],
            old_raw_response=a['raw_response'],raw_exact_replay=r['raw_response']==a['raw_response'],
            strict_label=p['strict_label'],old_candidate_label=a['predicted_label'],
            input_identity_equal=r['prepared_inputs']==a['prepared_inputs'],
            centered_candidate_error=float(np.max(np.abs((after-after.mean())-(before-before.mean())))),
            first_step_candidate_mass=r['generation_steps'][0]['abc_probability_mass'],
            truncated=r['hit_token_limit_without_turn_end']))
        for index,(token,piece,step) in enumerate(zip(r['generated_ids'],r['generated_token_pieces'],r['generation_steps'])):
            traces.append(dict(trial_id=tid,wording=r['wording'],step=index,token_id=token,
                token_piece=piece,abc_probability_mass=step['abc_probability_mass']))
    df=pd.DataFrame(data);Path(outdir).mkdir(parents=True,exist_ok=True)
    df.to_csv(Path(outdir)/'preflight_comparison.csv',index=False)
    pd.DataFrame(traces).to_csv(Path(outdir)/'preflight_token_trace.csv',index=False)
    # Gate checks technical interpretability only, never physical accuracy.
    ready=bool(df.input_identity_equal.all() and df.normalized_label.notna().all() and not df.truncated.any())
    report=dict(ready=ready,n=40,normalized_valid=int(df.normalized_label.notna().sum()),
        strict_valid=int(df.strict_label.notna().sum()),raw_exact_replays=int(df.raw_exact_replay.sum()),
        largest_centered_score_difference=float(df.centered_candidate_error.max()),
        rows_with_score_difference_above_015=int((df.centered_candidate_error>.15).sum()),
        gate='Exact input identity and complete, unambiguous response under the fixed grammar. No gate on physical correctness or agreement with archived candidate choices.',
        interpretation='Wrapper recovery is reported separately from strict direct letters; it does not establish normal channel behavior.')
    save_json(Path(outdir)/'preflight_report.json',report)
    return report,df


def analyze(rows, inputs, old, outdir, replicates=10000):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    assert len(rows)==len(inputs)==1440 and set(rows)=={x['trial_id'] for x in inputs}
    records=[]
    for inp in inputs:
        r=rows[inp['trial_id']];p=r['parsed']
        record={k:inp[k] for k in ('trial_id','family_id','stratum','scene_id','role','mirror','mapping_index','wording','gold_label','gold_semantic')}
        for metric,label in [('normalized',p['normalized_label']),('strict',p['strict_label'])]:
            sem=inp['semantics'][LABELS.index(label)] if label in LABELS else None
            record.update({metric+'_valid':int(label is not None),metric+'_correct':int(label==inp['gold_label']),
                metric+'_balance':int(sem=='balanced'),metric+'_left':int(sem=='left'),metric+'_right':int(sem=='right')})
        record.update(response_form=p['response_form'],normalized_label=p['normalized_label'],strict_label=p['strict_label'],
            raw_response=r['raw_response'],wrapper=int(p['had_channel_wrapper']),
            truncated=int(r['hit_token_limit_without_turn_end']),archived_scored_correct=old[r['trial_id']]['correct'],
            archived_scored_label=old[r['trial_id']]['predicted_label'],
            generated_matches_archived_score=int(p['normalized_label']==old[r['trial_id']]['predicted_label']))
        records.append(record)
    df=pd.DataFrame(records);df.to_csv(outdir/'generation_trials.csv',index=False)
    metrics=[c for c in df if c.startswith(('normalized_','strict_')) and c not in ('normalized_label','strict_label')]
    metrics+=['wrapper','truncated','archived_scored_correct','generated_matches_archived_score']
    fam=df.groupby(['family_id','stratum','wording'])[metrics].mean().reset_index()
    fam.to_csv(outdir/'family_metrics.csv',index=False)
    summary=[]
    for scope in ('all',*STRATA):
        g=fam if scope=='all' else fam[fam.stratum==scope]
        for wording,group in g.groupby('wording'):
            for metric in metrics:
                # Intervals are descriptive; do not test every table cell.
                stats=paired_stats(group[metric].to_numpy(),[SCHEMA,scope,wording,metric],replicates)
                stats.pop('p_value')
                summary.append(dict(scope=scope,wording=wording,metric=metric,**stats))
    pd.DataFrame(summary).to_csv(outdir/'accuracy_and_format_summary.csv',index=False)
    comparisons=[]
    for name,scope,metric in [('overall_accuracy','all','normalized_correct'),
                              ('true_balance_accuracy','balanced','normalized_correct'),
                              ('false_balance_response','nonbalanced','normalized_balance')]:
        g=fam if scope=='all' else fam[fam.stratum=='balanced'] if scope=='balanced' else fam[fam.stratum!='balanced']
        pivot=g.pivot(index='family_id',columns='wording',values=metric)
        comparisons.append(dict(comparison=name,scope=scope,metric=metric,
            **paired_stats((pivot.neutral_question-pivot.original).to_numpy(),[SCHEMA,name],replicates)))
    comp=pd.DataFrame(comparisons);comp['holm_recheck_3']=holm(comp.p_value)
    comp['status']='post_hoc_measurement_recheck_same_families_not_fresh_confirmation'
    comp.to_csv(outdir/'paired_wording_comparisons.csv',index=False)
    forms=df.groupby(['wording','response_form']).size().reset_index(name='n')
    forms.to_csv(outdir/'response_forms.csv',index=False)
    events=[]
    for scope in ('all',*STRATA):
        g=df if scope=='all' else df[df.stratum==scope]
        pivot=g.pivot(index=['family_id','scene_id','mapping_index'],columns='wording',values='normalized_correct')
        errors=int((pivot.original==0).sum());correct=int((pivot.original==1).sum())
        repair=int(((pivot.original==0)&(pivot.neutral_question==1)).sum())
        harm=int(((pivot.original==1)&(pivot.neutral_question==0)).sum())
        events.append(dict(scope=scope,n=len(pivot),repairs=repair,harms=harm,original_errors=errors,original_correct=correct,
                           repair_rate=repair/errors if errors else None,harm_rate=harm/correct if correct else None))
    pd.DataFrame(events).to_csv(outdir/'repairs_and_harms.csv',index=False)
    save_json(outdir/'measurement_status.json',dict(n_trials=len(df),n_families=96,
        normalized_invalid=int((1-df.normalized_valid).sum()),strict_invalid=int((1-df.strict_valid).sum()),
        incomplete_turns=int(df.truncated.sum()),
        normalized_interpretation='Single-letter generation after predefined response normalization; anomalous wrappers remain explicitly identified.',
        denominator='All 1440 trials; invalid replies count as unsuccessful, never silently removed. Family macro averaging.',
        multiplicity='Three revised Gemma wording comparisons form a separate post-hoc family; old 14/8-test results are preserved, not overwritten.'))
    return pd.DataFrame(summary),comp


def export_run(run_root, status):
    root=Path(run_root);target=root/'VisualTorque_Gemma_Generation_Recheck_Output.zip'
    files={}
    for folder in ('shared','run'):
        for p in sorted((root/folder).rglob('*')):
            if p.is_file() and p.suffix!='.tmp':files[p.relative_to(root).as_posix()]=file_hash(p)
    for name in ('protocol.json',):
        if (root/name).exists():files[name]=file_hash(root/name)
    manifest=dict(schema=SCHEMA,status=status,model='gemma12b',source_zip_sha256=SOURCE_SHA,files=files)
    temp=target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for name in files:z.write(root/name,name)
        z.writestr('export_manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    with zipfile.ZipFile(temp) as z:
        assert z.testzip() is None
        assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in files.items())
    temp.replace(target)
    return target
