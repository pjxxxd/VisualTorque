"""Data/provenance and analysis for frozen-predictor format transfer (no fitting)."""
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor

FORMAT_SYMBOLS = {'ABC': ('A', 'B', 'C'), 'XYZ': ('X', 'Y', 'Z'), 'DIGITS': ('1', '2', '3')}
TRANSFER_SCHEMA = 'frozen-format-transfer-v1'


class PriorResults:
    """Read a previous validation ZIP or original run directory without modifying it."""
    def __init__(self, path, model_key):
        self.path = Path(path)
        self.model_key = model_key
        self.prefix = f'models/{model_key}/'
        self.is_zip = self.path.is_file() and self.path.suffix.lower() == '.zip'
        self.archive_prefix = ''
        if self.is_zip:
            with zipfile.ZipFile(self.path) as z:
                candidates = [n[:-len('support.json')] for n in z.namelist()
                              if n == 'support.json' or n.endswith('/support.json')]
                assert len(candidates) == 1, 'ZIP must contain exactly one prior support.json.'
                self.archive_prefix = candidates[0]
        else:
            assert (self.path/'support.json').is_file(), 'SOURCE_ROOT must be the original run root containing support.json.'
        self.manifest = self.json('export_manifest.json') if self.exists('export_manifest.json') else None
        self.hashes = {}

    def exists(self, name):
        if self.is_zip:
            with zipfile.ZipFile(self.path) as z:
                return self.archive_prefix+name in z.namelist()
        return (self.path/name).is_file()

    def raw(self, name):
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        if self.is_zip:
            with zipfile.ZipFile(self.path) as z:
                data = z.read(self.archive_prefix+name)
        else:
            data = (self.path/name).read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if getattr(self, 'manifest', None) is not None and name != 'export_manifest.json':
            assert self.manifest['files'].get(name) == sha, 'Prior ZIP member checksum mismatch: '+name
        if hasattr(self, 'hashes'):
            self.hashes[name] = sha
        return data

    def json(self, name):
        return json.loads(self.raw(name))

    def validate(self):
        self.support = self.json('support.json')
        self.protocol = self.json('protocol.json')
        self.runtime = self.json(self.prefix+'runtime_identity.json')
        self.frozen = self.json(self.prefix+'frozen_predictors.json')
        self.completion = self.json(self.prefix+'validation/completion.json')
        self.pre_manifest = self.json(self.prefix+'validation/pre_intervention_manifest.json')
        self.image_manifest = self.json('image_manifest.json')
        self.old_rows = self.json(self.prefix+'validation/target_rows.json')
        self.old_sealed = self.json(self.prefix+'validation/predictions_before_intervention.json')
        self.old_seal = self.json(self.prefix+'validation/prediction_seal.json')
        ident = self.frozen['identity']
        assert ident['model_key'] == self.model_key
        for field, name in [('support_sha256','support.json'),
                            ('runtime_sha256',self.prefix+'runtime_identity.json')]:
            assert ident[field] == self.hashes[name], 'Prior identity mismatch: '+field
        # The original protocol identity is a canonical-object digest; the two
        # other identities above are byte hashes. Do not confuse the contracts.
        assert ident['protocol_sha256'] == digest(self.protocol)
        assert self.support['protocol_sha256'] == digest(self.protocol)
        assert self.completion['status'] == 'heldout_validation_complete'
        assert self.completion['prediction_sha256'] == self.hashes[self.prefix+'validation/predictions_before_intervention.json']
        for key, name in [('prediction_sha256','validation/predictions_before_intervention.json'),
                          ('pre_manifest_sha256','validation/pre_intervention_manifest.json'),
                          ('frozen_models_sha256','frozen_predictors.json')]:
            assert self.old_seal[key] == self.hashes[self.prefix+name]
        self.families = [f for f in self.support['families'] if f['module']=='content' and f['split']=='validation']
        assert len(self.families) == 96 and len({f['family_id'] for f in self.families}) == 96
        calibration_ids = {f['family_id'] for f in self.support['families'] if f['split']=='calibration'}
        assert not calibration_ids & {f['family_id'] for f in self.families}
        self.old_rows = [r for r in self.old_rows if r['module']=='content']
        assert len(self.old_rows)==1152 and len({r['trial_id'] for r in self.old_rows})==1152
        expected_ids = {f['family_id'] for f in self.families}
        assert {r['family_id'] for r in self.old_rows} == expected_ids
        assert all(r['split']=='validation' for r in self.old_rows)
        self.operations = {k:v for k,v in self.frozen['operations'].items() if k.startswith('content:')}
        assert len(self.operations)==2
        assert all(not expected_ids & set(v['calibration_families']) for v in self.operations.values())
        return self

    def cache_archives(self, directory):
        """Only the 96 used families: no recursive history scan and no extraction of unrelated files."""
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        def one(family):
            fid = family['family_id']; name = self.prefix+'validation/pre_intervention/'+fid+'.json'
            raw = self.raw(name); obj = json.loads(raw)
            assert hashlib.sha256(raw).hexdigest() == self.pre_manifest[fid+'.json']
            assert obj['family']==family
            archive = directory/(fid+'.npz')
            if not archive.is_file() or file_hash(archive)!=obj['arrays_sha256']:
                data = self.raw(name[:-5]+'.npz')
                assert hashlib.sha256(data).hexdigest()==obj['arrays_sha256']
                tmp = archive.with_suffix('.tmp'); tmp.write_bytes(data); tmp.replace(archive)
            self.hashes[name[:-5]+'.npz'] = obj['arrays_sha256']
            (directory/(fid+'.json')).write_bytes(raw)
            return fid
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(tqdm(pool.map(one,self.families),total=len(self.families),desc='Read 96 saved families'))
        return directory


def canonical_mapping(mapping, codebook):
    symbols = FORMAT_SYMBOLS[codebook]
    return [dict(x, label=symbols[LABELS.index(x['label'])]) for x in mapping]


def prediction_metric_rows(rows, sealed):
    pred = {r['trial_id']:r for r in sealed['payload']['records']}
    assert len(pred)==len(rows) and set(pred)=={r['trial_id'] for r in rows}
    metrics=[]
    for row in rows:
        entry=pred[row['trial_id']]
        assert entry['input_sha256']==digest({k:v for k,v in row.items() if k not in ('gold','cf_gold','actual_delta','post_record')})
        s0=np.asarray(row['s0']); actual=np.asarray(row['actual_delta'])
        before=int(np.argmax(s0)); after=int(np.argmax(s0+actual)); gold=LABELS.index(row['gold'])
        for name, delta in entry['predicted_deltas'].items():
            delta=np.asarray(delta); predicted=int(np.argmax(s0+delta))
            metrics.append(dict(family_id=row['family_id'],trial_id=row['trial_id'],
                base_trial_id=row['base_trial_id'],codebook=row['codebook'],operation=operation_key(row),
                predictor=name,mapping=row['mapping_name'],task=row['task'],
                mse=float(np.mean((delta-actual)**2)),post_answer_match=int(predicted==after),
                actual_change=int(after!=before),predicted_change=int(predicted!=before),
                baseline_correct=int(before==gold),post_correct=int(after==gold),predicted_correct=int(predicted==gold),
                actual_transition=transition(before,after,gold),predicted_transition=transition(before,predicted,gold)))
    return pd.DataFrame(metrics)


def analyze_transfer(rows, sealed, frozen, outdir, model_key, replicates=10000):
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    df=prediction_metric_rows(rows,sealed)
    df.to_csv(outdir/'trial_prediction_metrics.csv',index=False)
    keys=['codebook','operation','predictor']
    fam=df.groupby(keys+['family_id'])[['mse','post_answer_match','actual_change','baseline_correct','post_correct','predicted_correct']].mean().reset_index()
    fam.to_csv(outdir/'family_prediction_metrics.csv',index=False)
    summary=[]
    for values,g in df.groupby(keys):
        cb,op,predictor=values; actual=g.actual_change.eq(1)
        fg=fam[(fam.codebook==cb)&(fam.operation==op)&(fam.predictor==predictor)]
        err=paired_interval(fg.mse,[model_key,values,'mse'],replicates)
        summary.append(dict(zip(keys,values),model=model_key,n_families=g.family_id.nunique(),n_trials=len(g),
            mse=err['mean'],mse_lower=err['lower'],mse_upper=err['upper'],
            correct_choices=int(g.post_answer_match.sum()),choice_accuracy=float(g.post_answer_match.mean()),
            actual_changed=int(actual.sum()),correct_when_changed=int(g.loc[actual,'post_answer_match'].sum()),
            accuracy_when_changed=float(g.loc[actual,'post_answer_match'].mean()) if actual.any() else None,
            actual_gain=float((g.post_correct-g.baseline_correct).mean()),
            predicted_gain=float((g.predicted_correct-g.baseline_correct).mean()),
            frozen_selected=predictor==frozen[op]['selected_baseline']))
    summary=pd.DataFrame(summary); summary.to_csv(outdir/'prediction_summary.csv',index=False)
    tests=[]
    # Four comparisons at each of two sites and two NEW codebooks = 16/model.
    for (cb,op),g in fam[fam.codebook.ne('ABC')].groupby(['codebook','operation']):
        pivot=g.pivot(index='family_id',columns='predictor',values='mse')
        selected=frozen[op]['selected_baseline']
        comparisons=[('source_semantic','source_letter'),('source_letter','source_position'),
                     (selected,'null'),(selected,'confidence')]
        for index,(a,b) in enumerate(comparisons):
            stats=paired_interval((pivot[b]-pivot[a]).values,[model_key,cb,op,index,a,b],replicates)
            tests.append(dict(model=model_key,codebook=cb,operation=op,better_if_positive=a,comparator=b,**stats))
    for t,q in zip(tests,holm([t['p_value'] for t in tests])): t['holm_p_within_model']=float(q)
    pd.DataFrame(tests).to_csv(outdir/'primary_comparisons.csv',index=False)
    transfer=[]
    for (op,predictor),g in fam.groupby(['operation','predictor']):
        pivot=g.pivot(index='family_id',columns='codebook',values='mse')
        assert set(pivot)==set(FORMAT_SYMBOLS) and not pivot.isna().any().any()
        for cb in ('XYZ','DIGITS'):
            stats=paired_interval((pivot[cb]-pivot.ABC).values,[model_key,op,predictor,cb,'degradation'],replicates)
            transfer.append(dict(model=model_key,operation=op,predictor=predictor,codebook=cb,
                                 positive_means='larger_error_than_fresh_ABC',**stats))
    pd.DataFrame(transfer).to_csv(outdir/'format_error_change_descriptive.csv',index=False)
    # This is not an equivalence test: a CI containing zero does NOT prove invariance.
    events=[]
    for values,g in df.groupby(keys):
        for event in ('repair','harm','wrong_to_other_wrong','unchanged'):
            actual=g.actual_transition.eq(event); predicted=g.predicted_transition.eq(event)
            nt=int(actual.sum()); npred=int(predicted.sum()); tp=int((actual&predicted).sum())
            events.append(dict(zip(keys,values),event=event,n_actual=nt,n_predicted=npred,n_true_positive=tp,
                recall=tp/nt if nt else None,precision=tp/npred if npred else None))
    pd.DataFrame(events).to_csv(outdir/'transition_events.csv',index=False)
    mapping=df.groupby(keys+['mapping','family_id'])[['baseline_correct','post_correct','predicted_correct']].mean().reset_index()
    mapping['actual_gain']=mapping.post_correct-mapping.baseline_correct
    mapping['predicted_gain']=mapping.predicted_correct-mapping.baseline_correct
    mapping.to_csv(outdir/'mapping_family_gains.csv',index=False)
    return summary,pd.DataFrame(tests)
