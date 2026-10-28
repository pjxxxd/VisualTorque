"""GPU runner shared by both notebooks. New conditions never fit predictors."""
import gc
import time
import shutil
import torch
from tqdm.auto import tqdm
from visual_torque_toolkit.behavior.data import BehaviorExample
from visual_torque_toolkit.behavior.schema import PromptCondition
from visual_mechanism_core import *


def hidden_tensor(output):
    if torch.is_tensor(output): return output
    if isinstance(output,tuple): return output[0]
    raise TypeError('Unsupported decoder output: '+str(type(output)))


def with_hidden(output,hidden):
    return hidden if torch.is_tensor(output) else (hidden,*output[1:])


class PrefillHook:
    """Edit the final prompt state exactly once, never the generated token states."""
    def __init__(self,index,width,captured,audits,payload=None,expected_pre=None,random_key=None):
        self.index=index; self.width=width; self.captured=captured; self.audits=audits
        self.payload=payload; self.expected_pre=expected_pre; self.random_key=random_key; self.calls=0

    def __call__(self,module,inputs,output):
        self.calls+=1
        if self.calls!=1: return None
        hidden=hidden_tensor(output)
        assert hidden.shape[0]==1 and hidden.shape[1]==self.width, 'Expected full prompt on first cached-generation forward'
        assert hidden.dtype==torch.bfloat16
        pre=hidden[0,-1:]; before=pre.detach().float().cpu().numpy().copy()
        if self.expected_pre is not None:
            error=float(np.max(np.abs(before-np.asarray(self.expected_pre))))
            assert error==0., ('Untreated replay differs within this run',error,
                'Preserve output, check model/runtime/input identity; do not loosen this equality.')
        if self.payload is None:
            self.captured[self.index]=before; return None
        delta=torch.as_tensor(np.asarray(self.payload).copy(),device=pre.device,dtype=pre.dtype).reshape(pre.shape)
        target=pre+delta; desired=float((target.float()-pre.float()).norm())
        if self.random_key is None:
            post=target; audit=dict(matched=True,target_norm=desired,relative_error=0.)
        else:
            post_np,audit=native_norm_edit(before,random_unit(before.shape,self.random_key),desired,bf16)
            post=torch.as_tensor(post_np,device=pre.device,dtype=pre.dtype)
        audit.update(actual_norm=float((post.float()-pre.float()).norm()),pre_sha256=state_hash(before),
            payload_sha256=state_hash(self.payload),site='last_prompt_token_prefill_only')
        self.audits[str(self.index)]=audit
        self.captured[self.index]=post.detach().float().cpu().numpy().copy()
        modified=hidden.clone();modified[0,-1:]=post
        return with_hidden(output,modified)


def tensor_info(inputs):
    result={}
    for k,value in inputs.items():
        if torch.is_tensor(value):
            x=value.detach().cpu().contiguous()
            result[k]=dict(shape=list(x.shape),dtype=str(x.dtype),sha256=hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest())
        else: result[k]=str(value)
    return result


def get_layers(model):
    for path in ('model.language_model.layers','model.text_model.layers','language_model.model.layers',
                 'model.model.layers','model.layers','language_model.layers'):
        try:
            value=model
            for part in path.split('.'): value=getattr(value,part)
        except AttributeError: continue
        if isinstance(value,torch.nn.ModuleList) and len(value)>=8: return path,value
    raise RuntimeError('Decoder layers were not found in this pinned model.')


class VisualMechanismRun:
    def __init__(self,project,toolkit,model,stage,run_id,implementation,prediction_zip='',visual_zip=''):
        assert model in MODEL_SPECS and stage in CONDITIONS
        assert run_id and all(c.isalnum() or c in '_-' for c in run_id)
        self.project=Path(project);self.toolkit=Path(toolkit);self.model_key=model;self.stage=stage
        self.spec=MODEL_SPECS[model];self.implementation=implementation
        self.root=self.project/'results/visual_mechanism_frozen_v1'/run_id
        self.folder=self.root/'models'/model/stage;self.folder.mkdir(parents=True,exist_ok=True)
        self.analysis=self.folder/'analysis';self.analysis.mkdir(exist_ok=True)
        source=locate_source(project,model,'prediction',prediction_zip)
        print('Reading the named prediction ZIP:',source)
        self.prior=read_prediction_source(source,model)
        self.families=[f for f in self.prior['families'] if stage=='bridge' or f['module']=='content']
        self.by_id={f['family_id']:f for f in self.families}
        self.protocol=protocol_for(stage,implementation)
        freeze(self.root/(stage+'_protocol.json'),self.protocol)
        freeze(self.root/'shared/prediction_support.json',self.prior['support'])
        freeze(self.folder/'source_identity.json',self.prior['source_identity'])
        freeze(self.folder/'archived_runtime_identity.json',self.prior['runtime'])
        frozen_path=self.folder/'frozen_predictors.json'
        if frozen_path.exists(): assert frozen_path.read_bytes()==self.prior['frozen_bytes']
        else: frozen_path.write_bytes(self.prior['frozen_bytes'])
        self.fitted=self.prior['frozen']['operations'];self.panel=None
        if stage=='visual':
            source=locate_source(project,model,'visual',visual_zip)
            print('Reading the named visual-validation ZIP:',source)
            self.panel=read_panel_source(source,model)
            freeze(self.root/'shared/panel_support.json',self.panel['support'])
            freeze(self.folder/'panel_source_identity.json',self.panel['source_identity'])
        # PNGs are deterministic and cheap to regenerate. Avoid thousands of Drive writes.
        self.images=Path('/content/visualtorque_visual_mechanism_images')/digest([
            self.prior['source_identity']['protocol_identity']['support_sha256'],stage,implementation])[:20]
        self.images.mkdir(parents=True,exist_ok=True)
        self.local_edits=Path('/content/visualtorque_visual_mechanism_edits')/digest([str(self.root),model,stage,implementation])[:20]
        self.local_edits.mkdir(parents=True,exist_ok=True)
        self.layers_to_capture=sorted({l for f in self.families for l in self.spec[f['module']+'_layers']})
        self.runtime=None;self.adapter=None

    def image_path(self,role,condition):
        c='image' if condition=='image_full' else condition
        return self.images/(role['scene_id']+'__'+c+'.png')

    def prepare_images(self):
        roles={r['scene_id']:r for f in self.families for r in f['roles'].values()}
        if self.panel:
            roles.update({s['scene_id']:s for f in self.panel['support']['families'] for s in f['scenes']})
        styles=('image',) if self.stage=='bridge' else ('image','outline','layout')
        manifest={}
        for role in tqdm(list(roles.values()),desc='Render fixed visual conditions',unit='scene'):
            for c in styles:
                p=self.image_path(role,c);manifest[p.name]=render_variant(role,c,p)
        freeze(self.folder/'image_manifest.json',manifest)
        self.image_manifest=manifest
        counts=[]
        for c in CONDITIONS[self.stage]:
            for module in self.protocol['modules']:
                fs=[f for f in self.families if f['module']==module]
                n=sum(len(list(trial_specs(f,self.spec,c))) for f in fs)
                controls=sum(len(list(trial_specs(f,self.spec,c)))*len(RANDOM_SEEDS) for f in fs if f['random_control_family'])
                counts.append(dict(condition=c,module=module,families=len(fs),target_generations=n,random_generations=controls))
        pd.DataFrame(counts).to_csv(self.folder/'planned_counts.csv',index=False)
        print(pd.DataFrame(counts).to_string(index=False))
        print('Counts exclude untreated/source generations, reading calls and the visual behavior panel.')
        from PIL import Image,ImageDraw
        preview=Image.new('RGB',(896*len(styles),500),'white');draw=ImageDraw.Draw(preview)
        role=next(iter(roles.values()))
        for i,c in enumerate(styles):
            with Image.open(self.image_path(role,c)) as im: preview.paste(im,(896*i,28))
            draw.text((896*i+15,6),c,fill='black')
        preview.save(self.folder/'scene_preview.png')
        return self.folder/'scene_preview.png'

    def load_model(self):
        from visual_torque_toolkit.behavior.adapter import HuggingFaceMultimodalAdapter
        assert torch.cuda.is_available(),'Select a GPU runtime. Qwen-27B requires enough memory for native BF16.'
        self.adapter=HuggingFaceMultimodalAdapter(self.spec['model_id'],revision=self.spec['revision'],dtype='bfloat16',device_map='auto')
        assert self.adapter.metadata['resolved_revision']==self.spec['revision']
        assert not getattr(self.adapter.model,'is_quantized',False),'Native BF16 is required'
        assert not any(str(v) in ('cpu','disk') for v in getattr(self.adapter.model,'hf_device_map',{}).values()),'Model must fit on GPU without CPU/disk offload'
        self.adapter.model.eval();self.layer_path,self.layers=get_layers(self.adapter.model)
        assert len(self.layers)==self.spec['expected_layers']
        torch.manual_seed(self.protocol['seed']);np.random.seed(self.protocol['seed']);torch.backends.cuda.matmul.allow_tf32=False
        import transformers,PIL
        serial=lambda obj:json.loads(json.dumps(obj,default=str))
        proc=self.adapter.processor
        toolkit_hashes={str(p.relative_to(self.toolkit)):file_hash(p) for p in sorted((self.toolkit/'src').rglob('*.py'))}
        prior=self.prior['runtime'];assert transformers.__version__==prior['transformers']
        for suffix in ('behavior/adapter.py','behavior/data.py','behavior/schema.py','render.py','world.py'):
            canonical='src/visual_torque_toolkit/'+suffix
            old={k.replace('\\','/'):v for k,v in prior['toolkit'].items()}
            current={k.replace('\\','/'):v for k,v in toolkit_hashes.items()}
            assert canonical in old,'Missing old toolkit identity: '+canonical
            assert current.get(canonical)==old[canonical],'Restore the archived toolkit source: '+canonical
        self.runtime=dict(schema=SCHEMA,implementation=self.implementation,model_key=self.model_key,
            adapter_metadata=self.adapter.metadata,torch=torch.__version__,transformers=transformers.__version__,
            pillow=PIL.__version__,numpy=np.__version__,pandas=pd.__version__,cuda=torch.version.cuda,
            gpu=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],toolkit=toolkit_hashes,
            layer_path=self.layer_path,processor=serial(proc.to_dict()),image_processor=serial(proc.image_processor.to_dict()),
            chat_template=proc.chat_template,tokenizer_chat_template=proc.tokenizer.chat_template,
            generation_config=serial(self.adapter.model.generation_config.to_dict()),
            generation_overrides=dict(do_sample=False,num_beams=1,num_return_sequences=1,use_cache=True,max_choice_tokens=16,max_reading_tokens=128),
            protocol_sha256=file_hash(self.root/(self.stage+'_protocol.json')),
            source_identity_sha256=file_hash(self.folder/'source_identity.json'),
            image_manifest_sha256=file_hash(self.folder/'image_manifest.json'))
        self.runtime_sha=freeze(self.folder/'runtime_identity.json',self.runtime)
        for v in (np.array([1.00390625,-3.57,60.125],np.float32),):
            assert np.array_equal(bf16(v),torch.tensor(v).bfloat16().float().numpy())
        print('Model loaded; cached generation captures raw prefill logits and all generated tokens.')

    def example(self,family,role_name,task,mapping,condition,reading=False):
        role=family['roles'][role_name]
        semantics=[next(x['semantic'] for x in mapping if x['label']==l) for l in LABELS]
        gold=question_gold(role,task);eid='__'.join([family['family_id'],role_name,task,condition,digest(mapping)[:12],'reading' if reading else 'answer'])
        prompt=reading_prompt() if reading else prompt_for(role,task,mapping,condition)
        return BehaviorExample(example_id=eid,question_id=eid,scene_id=role['scene_id'],condition=PromptCondition.IMAGE_RULE,
            question_kind=task,variant=SCHEMA,split='validation',problem_type=family.get('stratum','controlled'),
            outcome=gold,net_torque=int(role['features']['net_torque']),prompt=prompt,image_path=self.image_path(role,condition),
            labels=LABELS,option_text=tuple(WORDS[s] for s in semantics),option_semantics=tuple(semantics),
            gold_label=LABELS[semantics.index(gold)],gold_semantic=gold)

    def forward(self,ex,layer=None,payload=None,expected_pre=None,expected_input=None,random_key=None,reading=False,parity=False):
        assert self.runtime is not None
        plan=self.adapter._candidate_plan(ex)
        assert plan is not None and not plan[0],'This protocol requires direct single-token A/B/C candidates with no inserted prefix'
        tokens=plan[1]
        inputs=self.adapter._to_device(self.adapter._template([ex.messages()],add_generation_prompt=True))
        identity=tensor_info(inputs);fingerprint=digest(dict(inputs=identity,tokens=tokens))
        if expected_input is not None: assert fingerprint==expected_input,'Prepared inputs changed'
        width=int(inputs['input_ids'].shape[1]);assert inputs['input_ids'].shape[0]==1
        standalone=None
        if parity:
            with torch.inference_mode():
                result=self.adapter.model(**dict(inputs),use_cache=False)
                lp=torch.log_softmax(result.logits[0,-1].float(),dim=-1)
                standalone=[float(lp[tokens[l]].item()) for l in LABELS]
            del result,lp
        captured={};audits={};hooks=[];handles=[]
        active=[] if reading else (self.layers_to_capture if layer is None else [int(layer)])
        for index in active:
            h=PrefillHook(index,width,captured,audits,payload,expected_pre,random_key)
            hooks.append(h);handles.append(self.layers[index].register_forward_hook(h))
        started=time.monotonic()
        try:
            with torch.inference_mode():
                output=self.adapter.model.generate(**dict(inputs),do_sample=False,num_beams=1,num_return_sequences=1,use_cache=True,
                    max_new_tokens=128 if reading else 16,pad_token_id=self.adapter.processor.tokenizer.pad_token_id,
                    return_dict_in_generate=True,output_scores=True,output_logits=True)
            assert all(h.calls>=1 for h in hooks) and set(captured)==set(active)
        finally:
            for handle in handles:handle.remove()
        ids=output.sequences[0,width:].detach().cpu().tolist()
        assert ids and len(ids)==len(output.scores)==len(output.logits)
        raw=self.adapter.processor.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False).strip()
        eos=self.adapter.model.generation_config.eos_token_id
        eos=[eos] if isinstance(eos,int) else list(eos or [])
        completed=ids[-1] in eos;parsed=parse_generation(raw,self.model_key,completed,reading)
        logp=torch.log_softmax(output.logits[0][0].float(),dim=-1)
        scores=[float(logp[tokens[l]].item()) for l in LABELS]
        assert np.isfinite(scores).all()
        first_mass=float(torch.exp(logp[[tokens[l] for l in LABELS]]).sum().item())
        chosen=int(np.argmax(scores));steps=[]
        for i,(token,raw_logits,processed) in enumerate(zip(ids,output.logits,output.scores)):
            lp=torch.log_softmax(raw_logits[0].float(),dim=-1)
            steps.append(dict(index=i,token_id=int(token),raw_argmax=int(raw_logits[0].argmax().item()),
                processed_argmax=int(processed[0].argmax().item()),
                candidate_logprobs=[float(lp[tokens[l]].item()) for l in LABELS],
                abc_mass=float(torch.exp(lp[[tokens[l] for l in LABELS]]).sum().item())))
        assert all(x['token_id']==x['processed_argmax'] for x in steps),'Generated token differs from greedy processed score'
        record=dict(candidate_labels=list(LABELS),candidate_semantics=list(ex.option_semantics),candidate_logprobs=scores,
            predicted_label=LABELS[chosen],predicted_semantic=ex.option_semantics[chosen],gold_label=ex.gold_label,gold_semantic=ex.gold_semantic,
            input_sha256=fingerprint,prepared_inputs=identity,input_token_ids=inputs['input_ids'][0].detach().cpu().tolist(),
            prompt=ex.prompt,prompt_sha256=hashlib.sha256(ex.prompt.encode()).hexdigest(),
            image_sha256=file_hash(ex.image_path),write_audits=audits,
            generation=dict(raw=raw,token_ids=ids,steps=steps,completed=completed,parsed=parsed,first_candidate_mass=first_mass,
                candidate_tokens=tokens,first_position_is_answer=parsed['label'] is not None and ids[0]==tokens[parsed['label']],
                generation_agreement=parsed['label']==LABELS[chosen]),
            standalone_logprobs=standalone,elapsed_seconds=time.monotonic()-started,created_at_utc=utc_now())
        del output,inputs,logp
        return record,captured

    def preflight(self):
        folder=self.folder/'preflight';folder.mkdir(exist_ok=True);rows=[]
        for module in self.protocol['modules']:
            family=next(f for f in self.families if f['module']==module)
            for c in CONDITIONS[self.stage]:
                spec=next(trial_specs(family,self.spec,c));mapping=spec['recipient_mapping']
                ex=self.example(family,spec['recipient'],'outcome',mapping,c)
                path=folder/(module+'__'+c+'.json')
                if path.exists():
                    saved=read_json(path);assert saved['runtime_sha256']==self.runtime_sha
                else:
                    rec,hidden=self.forward(ex,parity=True)
                    null,post=self.forward(ex,layer=spec['layer'],payload=np.zeros_like(hidden[spec['layer']]),
                        expected_pre=hidden[spec['layer']],expected_input=rec['input_sha256'])
                    saved=dict(runtime_sha256=self.runtime_sha,baseline=rec,zero_edit=null)
                    save_json(path,saved)
                baseline=saved['baseline'];null=saved['zero_edit']
                # A zero write must replay exactly in this fixed generation path.
                assert baseline['candidate_logprobs']==null['candidate_logprobs']
                assert baseline['generation']['token_ids']==null['generation']['token_ids']
                gap=float(np.max(np.abs(center(baseline['candidate_logprobs'])-center(baseline['standalone_logprobs']))))
                rows.append(dict(module=module,condition=c,standalone_vs_generation_centered_max=gap,
                    generation_valid=baseline['generation']['parsed']['valid'],
                    first_candidate_mass=baseline['generation']['first_candidate_mass']))
        table=pd.DataFrame(rows);table.to_csv(self.analysis/'prefill_generation_checks.csv',index=False)
        freeze(self.folder/'preflight_pass.json',dict(runtime_sha256=self.runtime_sha,zero_write_replay=True,
            source_metric='Raw prefill candidate log probabilities from generation; standalone differences reported, not silently used to refit.'))
        print(table.to_string(index=False))
        print('Responses with unknown grammar remain invalid and are retained; they are not filtered by accuracy.')

    def capture_family(self,family,c):
        folder=self.folder/'pre_intervention';folder.mkdir(exist_ok=True)
        name=family['family_id']+'__'+c;path=folder/(name+'.json');archive=folder/(name+'.npz')
        identity=digest([self.runtime_sha,family,c])
        if path.exists():
            obj=read_json(path);assert obj['identity']==identity and file_hash(archive)==obj['arrays_sha256'];return obj
        arrays={};records={};examples={};rows=[]
        def capture(role,task,mapping):
            ex=self.example(family,role,task,mapping,c);key=digest(ex.example_id)[:24]
            if key not in records:
                record,hidden=self.forward(ex);records[key]=record
                examples[key]=dict(role=role,task=task,mapping=mapping)
                for layer,h in hidden.items():arrays[key+f'__L{layer}']=h
            return key
        for spec in trial_specs(family,self.spec,c):
            sb=capture(spec['source_before'],spec['task'],spec['source_mapping'])
            sa=capture(spec['source_after'],spec['task'],spec['source_mapping'])
            rb=capture(spec['recipient'],'outcome',spec['recipient_mapping']);layer=spec['layer']
            h0=arrays[rb+f'__L{layer}'];delta=bf16(arrays[sa+f'__L{layer}']-arrays[sb+f'__L{layer}'])
            norm=float(np.linalg.norm(bf16(h0+delta)-h0));pkey=spec['trial_id']+'__delta';arrays[pkey]=delta
            row=dict(spec,family_id=family['family_id'],module=family['module'],stratum=family['stratum'],
                gold=records[rb]['gold_label'],source_key_before=sb,source_key_after=sa,recipient_key=rb,
                payload_key=pkey,pre_key=rb+f'__L{layer}',pre_sha256=state_hash(h0),payload_sha256=state_hash(delta),
                source_delta=center(np.asarray(records[sa]['candidate_logprobs'])-records[sb]['candidate_logprobs']).tolist(),
                source_before_argmax=int(np.argmax(records[sb]['candidate_logprobs'])),
                source_after_argmax=int(np.argmax(records[sa]['candidate_logprobs'])),
                s0=center(records[rb]['candidate_logprobs']).tolist(),planned_norm=norm,
                relative_norm=norm/max(float(np.linalg.norm(h0)),1e-12),random_control_family=family['random_control_family'])
            if family['module']=='physical':
                f=family['roles'][spec['recipient']]['features'];cf=capture(spec['cf_role'],'outcome',spec['recipient_mapping'])
                row.update(dw=family['dw'],target_side=family['target_side'],
                    baseline_physics=[f['net_torque'],f['normalized_margin'],f['weight_score'],f['distance_score']],
                    cf_gold=records[cf]['gold_label'],natural_input_key=cf,
                    natural_input_delta=(center(records[cf]['candidate_logprobs'])-np.asarray(row['s0'])).tolist())
            rows.append(row)
        temp=archive.with_suffix('.tmp.npz');np.savez_compressed(temp,**arrays);temp.replace(archive)
        obj=dict(identity=identity,family_id=family['family_id'],condition=c,rows=rows,baseline_records=records,examples=examples,
            arrays_sha256=file_hash(archive),created_at_utc=utc_now());save_json(path,obj);return obj

    def capture_all(self):
        assert read_json(self.folder/'preflight_pass.json')['runtime_sha256']==self.runtime_sha
        rows=[];manifest={}
        for family in tqdm(self.families,desc='Capture sources and untreated recipients',unit='family'):
            for c in CONDITIONS[self.stage]:
                obj=self.capture_family(family,c);rows.extend(obj['rows'])
                p=self.folder/'pre_intervention'/(family['family_id']+'__'+c+'.json');manifest[p.name]=file_hash(p)
        freeze(self.folder/'pre_intervention_manifest.json',manifest)
        freeze(self.folder/'pre_intervention_rows.json',rows)
        return rows

    def seal(self,rows):
        identity=dict(runtime_sha256=self.runtime_sha,frozen_models_sha256=file_hash(self.folder/'frozen_predictors.json'),
            pre_manifest_sha256=file_hash(self.folder/'pre_intervention_manifest.json'),
            protocol_sha256=file_hash(self.root/(self.stage+'_protocol.json')))
        path=self.folder/'predictions_before_intervention.json'
        assert path.exists() or not any((self.folder/'interventions').glob('*.json')),'Cannot seal predictions after targets already exist'
        seal_payload(path,rows,self.fitted,identity)
        freeze(self.folder/'prediction_seal.json',dict(**identity,predictions_sha256=file_hash(path)))
        print('All predictions saved before interventions:',len(rows))

    def verify_seal(self):
        seal=read_json(self.folder/'prediction_seal.json')
        assert seal['runtime_sha256']==self.runtime_sha
        for key,name in [('predictions_sha256','predictions_before_intervention.json'),
                         ('frozen_models_sha256','frozen_predictors.json'),('pre_manifest_sha256','pre_intervention_manifest.json')]:
            assert seal[key]==file_hash(self.folder/name)
        return digest(seal)

    def execute(self):
        seal=self.verify_seal();manifest=read_json(self.folder/'pre_intervention_manifest.json')
        folder=self.folder/'interventions';folder.mkdir(exist_ok=True);targets=[];controls=[]
        for name,sha in tqdm(manifest.items(),desc='Run frozen interventions',unit='family-condition'):
            p=self.folder/'pre_intervention'/name;assert file_hash(p)==sha
            obj=read_json(p);archive=p.with_suffix('.npz');assert file_hash(archive)==obj['arrays_sha256']
            family=self.by_id[obj['family_id']]
            checkpoint_identity=digest([self.runtime_sha,seal,name,obj['arrays_sha256']])
            durable=folder/name;local=self.local_edits/name
            if durable.exists():batch=read_json(durable)
            elif local.exists():batch=read_json(local)
            else:batch=dict(identity=checkpoint_identity,records={})
            assert batch['identity']==checkpoint_identity,'Intervention family checkpoint differs'
            with np.load(archive,allow_pickle=False) as arrays:
                for row in tqdm(obj['rows'],desc=name[:-5],leave=False,unit='edit'):
                    pre=obj['baseline_records'][row['recipient_key']]
                    identity=digest([self.runtime_sha,seal,row,obj['arrays_sha256']])
                    ex=self.example(family,row['recipient'],'outcome',row['recipient_mapping'],row['condition'])
                    kwargs=dict(layer=row['layer'],payload=arrays[row['payload_key']],expected_pre=arrays[row['pre_key']],expected_input=pre['input_sha256'])
                    for seed in [None]+(list(RANDOM_SEEDS) if row['random_control_family'] else []):
                        suffix='target' if seed is None else f'random{seed}'
                        record_key=row['trial_id']+'__'+suffix
                        if record_key in batch['records']:
                            saved=batch['records'][record_key];assert saved['identity']==identity and saved['seed']==seed
                        else:
                            key=None if seed is None else [self.protocol['seed'],row['trial_id'],seed]
                            record,_=self.forward(ex,random_key=key,**kwargs)
                            saved=dict(identity=identity,seed=seed,record=record,seal=seal,created_at_utc=utc_now())
                            batch['records'][record_key]=saved
                            save_json(local,batch)
                        record=saved['record'];actual=(center(record['candidate_logprobs'])-np.asarray(row['s0'])).tolist()
                        if seed is None:
                            targets.append(dict(row,actual_delta=actual,post_record=record,pre_record=pre,
                                natural_record=obj['baseline_records'].get(row.get('natural_input_key'))))
                        else:
                            controls.append(dict(trial_id=row['trial_id'],family_id=row['family_id'],module=row['module'],layer=row['layer'],
                                condition=row['condition'],seed=seed,audit=record['write_audits'][str(row['layer'])],
                                actual_delta=actual,pre_record=pre,post_record=record))
            expected_keys={r['trial_id']+'__'+suffix for r in obj['rows'] for suffix in
                ['target']+([f'random{s}' for s in RANDOM_SEEDS] if r['random_control_family'] else [])}
            assert set(batch['records'])==expected_keys
            if not durable.exists():save_json(durable,batch)
        expected=sum(len(list(trial_specs(f,self.spec,c))) for f in self.families for c in CONDITIONS[self.stage])
        assert len(targets)==len({r['trial_id'] for r in targets})==expected
        save_json(self.folder/'target_rows.json',targets);save_json(self.folder/'random_rows.json',controls)
        return targets,controls

    def run_panel(self):
        assert self.panel is not None
        folder=self.folder/'panel';folder.mkdir(exist_ok=True);outcomes=[];readings=[]
        for f in tqdm(self.panel['support']['families'],desc='Visual behavior and variable reading',unit='family'):
            family=dict(f,roles={s['scene_id']:s for s in f['scenes']})
            for c in CONDITIONS['visual']:
                path=folder/(f['family_id']+'__'+c+'.json');identity=digest([self.runtime_sha,f,c,'panel'])
                if path.exists():
                    obj=read_json(path);assert obj['identity']==identity
                else:
                    answers=[];reads=[]
                    for scene in f['scenes']:
                        ref=mapping_reference(f['family_id'])
                        mappings=[ref,mapping_conflict(ref,'left'),mapping_conflict(ref,'left',True)]
                        for mi,mapping in enumerate(mappings):
                            ex=self.example(family,scene['scene_id'],'outcome',mapping,c)
                            record,_=self.forward(ex)
                            answers.append(dict(trial_id=ex.example_id,family_id=f['family_id'],scene_id=scene['scene_id'],
                                role=scene['role'],mirror=scene['mirror'],stratum=f['stratum'],mapping_index=mi,condition=c,
                                semantics=list(ex.option_semantics),gold_semantic=ex.gold_semantic,record=record))
                        ex=self.example(family,scene['scene_id'],'outcome',ref,c,reading=True)
                        record,_=self.forward(ex,reading=True)
                        reads.append(dict(family_id=f['family_id'],scene_id=scene['scene_id'],condition=c,record=record))
                    obj=dict(identity=identity,answers=answers,readings=reads);save_json(path,obj)
                outcomes.extend(obj['answers']);readings.extend(obj['readings'])
        assert len(outcomes)==2160 and len(readings)==720
        panel_analysis(outcomes,readings,self.panel['support']['families'],self.analysis)
        return outcomes,readings

    def analyze(self,targets,controls):
        self.verify_seal()
        summary,tests=compare_primary(targets,read_json(self.folder/'predictions_before_intervention.json'),self.stage,self.model_key,self.analysis)
        controls_table=[]
        targets_by_id={r['trial_id']:r for r in targets}
        for r in controls:
            t=targets_by_id[r['trial_id']];pre=r['pre_record'];post=r['post_record']
            controls_table.append(dict(trial_id=r['trial_id'],family_id=r['family_id'],module=r['module'],layer=r['layer'],
                condition=r['condition'],seed=r['seed'],**r['audit'],
                random_gain=int(post['predicted_label']==t['gold'])-int(pre['predicted_label']==t['gold']),
                target_gain=int(t['post_record']['predicted_label']==t['gold'])-int(pre['predicted_label']==t['gold'])))
        cd=pd.DataFrame(controls_table);cd.to_csv(self.analysis/'random_norm_checks.csv',index=False)
        if len(cd):
            cd['target_minus_random_gain']=cd.target_gain-cd.random_gain
            cd.groupby(['module','layer','condition','family_id'])[['target_minus_random_gain','matched']].mean().groupby(
                ['module','layer','condition']).mean().to_csv(self.analysis/'random_comparison_descriptive.csv')
        natural=[]
        for r in targets:
            if r['module']!='physical':continue
            rec=r['natural_record'];post=r['post_record'];pre=r['pre_record']
            natural.append(dict(family_id=r['family_id'],condition=r['condition'],layer=r['layer'],mapping=r['mapping_name'],
                source_distance=r['source_distance'],recipient_distance=r['recipient_distance'],
                natural_scored_correct=rec['predicted_label']==r['cf_gold'],
                natural_generated_correct=rec['generation']['parsed']['label']==r['cf_gold'],
                transferred_scored_cf_correct=post['predicted_label']==r['cf_gold'],
                transferred_generated_cf_correct=post['generation']['parsed']['label']==r['cf_gold'],
                untreated_scored_cf_correct=pre['predicted_label']==r['cf_gold']))
        if natural:
            nf=pd.DataFrame(natural);nf.to_csv(self.analysis/'natural_counterfactual_trials.csv',index=False)
            metrics=[c for c in nf if c.endswith('correct')]
            nf.groupby(['condition','layer','family_id'])[metrics].mean().groupby(['condition','layer']).mean().to_csv(self.analysis/'natural_counterfactual_summary.csv')
        # Inventory every actual choice generation, including baseline/source and random edits.
        measurement=[]
        for p in sorted((self.folder/'pre_intervention').glob('*.json')):
            obj=read_json(p)
            for key,rec in obj['baseline_records'].items():measurement.append(self.measurement_row(obj['condition'],'untreated',key,rec))
        for r in targets:measurement.append(self.measurement_row(r['condition'],'target',r['trial_id'],r['post_record']))
        for r in controls:measurement.append(self.measurement_row(r['condition'],'random',r['trial_id']+str(r['seed']),r['post_record']))
        reading_invalid=0
        if self.stage=='visual':
            for path in sorted((self.folder/'panel').glob('*.json')):
                obj=read_json(path)
                for r in obj['answers']:measurement.append(self.measurement_row(r['condition'],'panel',r['trial_id'],r['record']))
                reading_invalid+=sum(not r['record']['generation']['parsed']['valid'] for r in obj['readings'])
        mf=pd.DataFrame(measurement);mf.to_csv(self.analysis/'measurement_trials.csv',index=False)
        mf.groupby(['condition','kind'])[['valid','direct','first_position_is_answer','agreement','candidate_mass']].mean().to_csv(self.analysis/'measurement_summary.csv')
        status=dict(acquisition_complete=True,measurement_needs_review=bool((~mf.valid).any() or (~mf.first_position_is_answer).any()),
            n_choice_generations=len(mf),invalid=int((~mf.valid).sum()),non_answer_first_position=int((~mf.first_position_is_answer).sum()),
            score_generation_disagreements=int((~mf.agreement).sum()),random_unmatched=int((~cd.matched).sum()) if len(cd) else 0,
            reading_invalid=reading_invalid,
            explanation='Score forecasts target first-position raw candidate logits. Actual generated answers are separate; invalid replies stay in denominators.')
        save_json(self.analysis/'measurement_status.json',status)
        return summary,tests,status

    @staticmethod
    def measurement_row(condition,kind,key,record):
        g=record['generation'];p=g['parsed']
        return dict(condition=condition,kind=kind,key=key,valid=p['valid'],direct=p['strict_valid'],
            first_position_is_answer=g['first_position_is_answer'],agreement=g['generation_agreement'],candidate_mass=g['first_candidate_mass'])

    def complete(self,targets,status):
        if self.stage=='visual':
            assert len(list((self.folder/'panel').glob('*.json')))==288,'The behavior/reading panel is unfinished'
        save_json(self.folder/'completion.json',dict(**status,n_families=len(self.families),n_target_interventions=len(targets),
            protocol_sha256=file_hash(self.root/(self.stage+'_protocol.json')),
            support_sha256=file_hash(self.root/'shared/prediction_support.json'),
            panel_support_sha256=file_hash(self.root/'shared/panel_support.json') if self.stage=='visual' else None,
            primary_comparisons_sha256=file_hash(self.analysis/'primary_comparisons.csv'),runtime_sha256=self.runtime_sha,
            prediction_sha256=file_hash(self.folder/'predictions_before_intervention.json'),completed_at_utc=utc_now()))
        return combined_tests(self.root,self.stage)

    def export(self):
        # Include only this stage/model, shared input assets, and any combined tables.
        paths=[self.root/(self.stage+'_protocol.json')]
        paths+=list((self.root/'shared').rglob('*'))+list(self.folder.rglob('*'))
        paths+=list((self.root/'analysis').glob(self.stage+'_*'))
        paths=sorted({p for p in paths if p.is_file() and '.tmp' not in p.name})
        members={str(p.relative_to(self.root)).replace('\\','/'):p for p in paths}
        members.update({'shared/images/'+p.name:p for p in self.images.glob('*.png')})
        # Include a partial current family from local disk when the user exports after interruption.
        for p in self.local_edits.glob('*.json'):
            name=f'models/{self.model_key}/{self.stage}/interventions/'+p.name
            if name not in members:members[name]=p
        files={name:file_hash(p) for name,p in members.items()}
        complete=read_json(self.folder/'completion.json') if (self.folder/'completion.json').exists() else None
        manifest=dict(schema=SCHEMA,stage=self.stage,model=self.model_key,status='complete' if complete else 'partial',
            measurement_needs_review=complete.get('measurement_needs_review') if complete else None,
            created_at_utc=utc_now(),files=files)
        name=f'VisualTorque_VisualMechanism_{self.model_key}_{self.stage}_Output.zip'
        local=Path('/content')/name;tmp=local.with_suffix('.tmp.zip')
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
            for name,p in tqdm(members.items(),desc='Export evidence',unit='file'):z.write(p,name)
            z.writestr('export_manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        tmp.replace(local);dest=self.root/name;shutil.copy2(local,dest)
        print('Drive ZIP:',dest);print('Local ZIP:',local)
        try:
            from IPython.display import FileLink,display
            display(FileLink(str(local)))
        except ImportError:pass
        return local
