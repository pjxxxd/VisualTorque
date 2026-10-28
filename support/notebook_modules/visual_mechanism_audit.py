"""CPU verification of exported visual-mechanism evidence; no model loading."""
import zipfile
from visual_mechanism_core import *


def audit_one(path,outdir):
    a=VerifiedArchive(path)
    try:
        meta=a.manifest;stage=meta['stage'];model=meta['model']
        assert meta['schema']==SCHEMA and stage in CONDITIONS and model in MODEL_SPECS
        # Stream verification avoids retaining large activation/JSON files at once.
        for name,expected in meta['files'].items():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            h=hashlib.sha256()
            with a.zip.open(a.prefix+name) as f:
                for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
            assert h.hexdigest()==expected,('Export checksum mismatch',name)
        prefix=f'models/{model}/{stage}/'
        report=dict(model=model,stage=stage,source_name=Path(path).name,source_zip_sha256=file_hash(path),
            manifest_files=len(meta['files']),complete=meta['status']=='complete')
        if not report['complete']:
            report['note']='Partial export verified; primary comparisons not treated as complete.'
            return report
        protocol=a.json(stage+'_protocol.json');runtime=a.json(prefix+'runtime_identity.json')
        completion=a.json(prefix+'completion.json');seal=a.json(prefix+'prediction_seal.json')
        sealed=a.json(prefix+'predictions_before_intervention.json');frozen=a.json(prefix+'frozen_predictors.json')
        source=a.json(prefix+'source_identity.json');support=a.json('shared/prediction_support.json')
        pre_manifest=a.json(prefix+'pre_intervention_manifest.json')
        assert protocol==protocol_for(stage,protocol['implementation'])
        assert completion['protocol_sha256']==a.hashes[stage+'_protocol.json']==runtime['protocol_sha256']
        assert completion['support_sha256']==a.hashes['shared/prediction_support.json']
        assert completion['runtime_sha256']==digest(runtime)==seal['runtime_sha256']
        assert seal['predictions_sha256']==a.hashes[prefix+'predictions_before_intervention.json']==completion['prediction_sha256']
        assert seal['frozen_models_sha256']==a.hashes[prefix+'frozen_predictors.json']
        assert seal['pre_manifest_sha256']==a.hashes[prefix+'pre_intervention_manifest.json']
        assert sealed['payload']['identity']=={k:v for k,v in seal.items() if k!='predictions_sha256'}
        expected_frozen=source['consumed_sha256'][f'models/{model}/frozen_predictors.json']
        assert a.hashes[prefix+'frozen_predictors.json']==expected_frozen,'Imported predictor bytes changed'
        pre_rows=a.json(prefix+'pre_intervention_rows.json')
        assert sealed['payload']['rows_sha256']==digest(pre_rows)
        families=[f for f in support['families'] if f['split']=='validation' and (stage=='bridge' or f['module']=='content')]
        expected={r['trial_id'] for f in families for c in CONDITIONS[stage] for r in trial_specs(f,MODEL_SPECS[model],c)}
        assert set(sealed['payload']['predictions'])==expected=={r['trial_id'] for r in pre_rows}
        reconstructed=[];baseline={};saved_interventions={}
        for name,sha in pre_manifest.items():
            obj=a.json(prefix+'pre_intervention/'+name)
            assert a.hashes[prefix+'pre_intervention/'+name]==sha
            archive_name=prefix+'pre_intervention/'+name[:-5]+'.npz'
            assert meta['files'][archive_name]==obj['arrays_sha256']
            reconstructed.extend(obj['rows'])
            for r in obj['rows']:baseline[r['trial_id']]=obj['baseline_records'][r['recipient_key']]
            batch=a.json(prefix+'interventions/'+name)
            expected_identity=digest([completion['runtime_sha256'],digest(seal),name,obj['arrays_sha256']])
            assert batch['identity']==expected_identity
            for k,v in batch['records'].items():
                assert k not in saved_interventions;saved_interventions[k]=v
        # JSON object keys are sorted on serialization; compare by trial ID, not file iteration order.
        assert {r['trial_id']:r for r in reconstructed}=={r['trial_id']:r for r in pre_rows}
        max_error=0.
        for r in pre_rows:
            for name,fit in frozen['operations'][operation_key(r)]['models'].items():
                saved=np.asarray(sealed['payload']['predictions'][r['trial_id']][name])
                err=float(np.max(np.abs(saved-predict_one(r,fit))));max_error=max(max_error,err)
        assert max_error<1e-10,'Saved forecasts do not match frozen predictors'
        targets=a.json(prefix+'target_rows.json');controls=a.json(prefix+'random_rows.json')
        assert len(targets)==len(expected)==completion['n_target_interventions']
        assert {r['trial_id'] for r in targets}==expected
        expected_controls={(r['trial_id'],seed) for r in pre_rows if r['random_control_family'] for seed in RANDOM_SEEDS}
        assert len(controls)==len(expected_controls) and {(r['trial_id'],r['seed']) for r in controls}==expected_controls
        for r in targets:
            assert r['pre_record']==baseline[r['trial_id']]
            np.testing.assert_array_equal(np.asarray(r['actual_delta']),center(r['post_record']['candidate_logprobs'])-np.asarray(r['s0']))
            saved=saved_interventions[r['trial_id']+'__target']
            assert saved['record']==r['post_record'] and saved['seal']==digest(seal)
            assert saved['created_at_utc']>=sealed['created_at_utc'],'Intervention predates forecast seal'
            rec=r['post_record'];g=rec['generation']
            assert parse_generation(g['raw'],model,g['completed'])==g['parsed']
            assert g['token_ids']==[s['token_id'] for s in g['steps']]
            assert all(s['token_id']==s['processed_argmax'] for s in g['steps'])
            assert rec['predicted_label']==LABELS[int(np.argmax(rec['candidate_logprobs']))]
            np.testing.assert_array_equal(rec['candidate_logprobs'],g['steps'][0]['candidate_logprobs'])
        folder=Path(outdir)/'models'/model/stage;folder.mkdir(parents=True,exist_ok=True)
        analysis=folder/'analysis'
        summary,tests=compare_primary(targets,sealed,stage,model,analysis)
        archived=pd.read_csv(io.BytesIO(a.raw(prefix+'analysis/primary_comparisons.csv')),keep_default_na=False)
        key=['model','stage','module','layer','condition','preferred','comparator']
        left=tests.sort_values(key).reset_index(drop=True);right=archived.sort_values(key).reset_index(drop=True)
        assert left[key].equals(right[key])
        for c in ('mean','lower','upper','p_value','holm_p_planned_scope'):
            np.testing.assert_allclose(left[c],right[c],rtol=1e-10,atol=1e-12)
        if stage=='visual':
            panel=a.json('shared/panel_support.json');rows=[];reads=[]
            assert completion['panel_support_sha256']==a.hashes['shared/panel_support.json']
            for f in panel['families']:
                for c in CONDITIONS['visual']:
                    obj=a.json(prefix+'panel/'+f['family_id']+'__'+c+'.json')
                    rows.extend(obj['answers']);reads.extend(obj['readings'])
            assert len(rows)==2160 and len(reads)==720
            for r in reads:
                g=r['record']['generation'];assert parse_generation(g['raw'],model,g['completed'],True)==g['parsed']
            panel_analysis(rows,reads,panel['families'],analysis)
        checked_completion=dict(completion,primary_comparisons_sha256=file_hash(analysis/'primary_comparisons.csv'))
        save_json(folder/'completion.json',checked_completion)
        save_json(Path(outdir)/(stage+'_protocol.json'),protocol)
        report.update(n_families=len(families),n_target_interventions=len(targets),n_controls=len(controls),
            frozen_prediction_max_error=max_error,primary_tables_reproduced=True,
            measurement_status=a.json(prefix+'analysis/measurement_status.json'))
        return report
    finally:a.close()


def audit_outputs(paths,outdir,result_root=''):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    if not paths:
        roots=[Path('/content')]+([Path(result_root)] if result_root else [])
        paths=[]
        for stage in CONDITIONS:
            for model in MODEL_SPECS:
                name=f'VisualTorque_VisualMechanism_{model}_{stage}_Output.zip'
                candidate=next((root/name for root in roots if (root/name).is_file()),None)
                if candidate:paths.append(candidate)
    assert paths,'No results found. Upload the output ZIPs or fill OUTPUT_ZIPS / RESULT_ROOT.'
    reports=[];seen=set()
    for path in paths:
        print('Verifying:',Path(path).name)
        r=audit_one(path,outdir);key=(r['stage'],r['model'])
        assert key not in seen,'Choose one export per stage/model';seen.add(key);reports.append(r)
    complete={(r['stage'],r['model']) for r in reports if r['complete']}
    combined=[]
    for stage in CONDITIONS:
        # Only combine models that were audited in THIS invocation, not stale output files.
        if all((stage,m) in complete for m in MODEL_SPECS):
            table=combined_tests(outdir,stage);assert table is not None;combined.append(stage)
    pending=[dict(stage=s,model=m) for s in CONDITIONS for m in MODEL_SPECS if (s,m) not in complete]
    report=dict(schema='visual-mechanism-audit-v1',reports=reports,combined_stages=combined,pending=pending,
        all_four_complete=not pending,created_at_utc=utc_now(),scope='CPU evidence/forecast verification, not GPU replication.')
    save_json(outdir/'audit_report.json',report)
    (outdir/'README_ZH.md').write_text('# 视觉机制实验核验\n\n'+
        ('四个计划运行均已核验。' if not pending else '仍有未完成或未提供的运行，详见 audit_report.json。')+
        '\n\n第一位置候选评分与实际生成分别呈现；measurement_status 标记不能忽略。'+
        '\n共享家族与多种映射不增加独立样本数；两阶段分别校正。\n',encoding='utf-8')
    zip_path=outdir.parent/'VisualTorque_VisualMechanism_Audit_Output.zip'
    with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(outdir.rglob('*')):
            if p.is_file():z.write(p,str(p.relative_to(outdir)).replace('\\','/'))
    return report,zip_path
