"""Resumable generation runner. All inference is batch-size one; no activation edits."""
from __future__ import annotations

import gc
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from relation_use_core import *


class RelationUseRun:
    def __init__(self,project,toolkit,model,run_id,implementation,source='',extra_history=(),local_root='/content'):
        assert model in MODEL_SPECS, 'Use gemma12b or qwen27b'
        assert run_id and all(c.isalnum() or c in '_-' for c in run_id)
        self.project=Path(project);self.toolkit=Path(toolkit);self.model=model
        self.spec=MODEL_SPECS[model];self.implementation=implementation
        self.root=self.project/'results/relation_use_v1'/run_id
        self.shared=self.root/'shared';self.folder=self.root/'models'/model
        self.folder.mkdir(parents=True,exist_ok=True)
        self.config=protocol(implementation)
        freeze(self.shared/'protocol.json',self.config)
        selected=find_history_source(self.project,source)
        print('Reading the named prior support:',selected)
        excluded,snapshot,provenance=read_history(selected,extra_history)
        freeze(self.folder/'history_source.json',provenance)
        freeze(self.shared/'historical_exclusion.json',snapshot)
        support_path=self.shared/'support.json'
        if support_path.exists():
            support=read_json(support_path)
            assert support['implementation']==implementation and support['history_sha256']==digest(snapshot)
            validate_families(support['families'],excluded)
        else:
            support=make_support(snapshot,implementation)
            freeze(support_path,support)
        self.support=support;self.families=support['families']
        self.family_by_id={f['family_id']:f for f in self.families}
        self.scenes={s['scene_id']:s for f in self.families for s in f['scenes']}
        self.local=Path(local_root)/'visualtorque_relation_use'/digest([str(self.root),implementation])[:20]
        self.images=self.local/'images';self.local.mkdir(parents=True,exist_ok=True)
        self.adapter=None;self.runtime_sha=None
        print(f"Frozen {len(excluded)} archived geometry exclusions; 24 development + 96 validation families.")
        plans=[]
        for phase in COUNTS:
            fs=[f for f in self.families if f['phase']==phase]
            ts=[t for f in fs for t in trials(f)]
            plans.append(dict(phase=phase,families=len(fs),scenes=sum(len(f['scenes']) for f in fs),
                judgment_generations=sum(t['task']=='judgment' for t in ts),
                component_generations=sum(t['task']!='judgment' for t in ts),total_generations=len(ts)))
        pd.DataFrame(plans).to_csv(self.folder/'planned_counts.csv',index=False)
        print(pd.DataFrame(plans).to_string(index=False))

    def prepare_images(self):
        manifest=render_images(self.families,self.images)
        freeze(self.shared/'image_manifest.json',manifest)
        image_zip=self.shared/'images.zip'
        expected_identity=dict(image_manifest_sha256=digest(manifest),files=len(manifest))
        if image_zip.exists():
            old=read_json(self.shared/'images_identity.json')
            assert old['manifest']==expected_identity and old['sha256']==file_hash(image_zip)
        else:
            localzip=self.local/'images.zip'
            with zipfile.ZipFile(localzip,'w',zipfile.ZIP_DEFLATED) as z:
                for name in sorted(manifest): z.write(self.images/name,name)
            tmp=image_zip.with_suffix('.tmp.zip');shutil.copyfile(localzip,tmp);tmp.replace(image_zip)
            freeze(self.shared/'images_identity.json',dict(manifest=expected_identity,sha256=file_hash(image_zip)))
        self.image_manifest=manifest
        from PIL import Image,ImageDraw
        family=next(f for f in self.families if f['phase']=='development' and f['stratum']=='swap_flip')
        scenes=[s for s in family['scenes'] if not s['mirror']]
        canvas=Image.new('RGB',(896*2,490),'white');draw=ImageDraw.Draw(canvas)
        for i,s in enumerate(scenes):
            with Image.open(self.images/(s['scene_id']+'__ids.png')) as im: canvas.paste(im,(i*896,35))
            draw.text((i*896+15,10),f"{s['role']} | left torque {s['features']['left_torque']}, right torque {s['features']['right_torque']}",fill='black')
        canvas.save(self.folder/'scene_preview.png')
        # Preview annotations are never shown to the model.
        ex=next(t for t in trials(family) if t['level']=='facts' and t['mode']=='semantic')
        print('Example prompt (actual model input):\n'+prompt_for(self.scenes[ex['scene_id']],ex))
        return self.folder/'scene_preview.png'

    def load_model(self):
        import torch
        import transformers
        import PIL
        from visual_torque_toolkit.behavior.adapter import HuggingFaceMultimodalAdapter
        assert torch.cuda.is_available(), 'Select a GPU runtime'
        assert transformers.__version__=='5.14.0' and PIL.__version__=='11.3.0'
        total=sum(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count()))/2**30
        minimum=32 if self.model=='gemma12b' else 70
        assert total>=minimum, f'{self.model}: use at least {minimum} GiB aggregate GPU memory for native BF16; no quantization/offload'
        torch.manual_seed(20261014);np.random.seed(20261014);torch.backends.cuda.matmul.allow_tf32=False
        self.adapter=HuggingFaceMultimodalAdapter(self.spec['model_id'],revision=self.spec['revision'],dtype='bfloat16',device_map='auto')
        assert self.adapter.metadata['resolved_revision']==self.spec['revision']
        assert not getattr(self.adapter.model,'is_quantized',False)
        assert not any(str(x) in ('cpu','disk') for x in getattr(self.adapter.model,'hf_device_map',{}).values()), 'Native model must fit on GPU'
        self.adapter.model.eval()
        serial=lambda v:json.loads(json.dumps(v,default=str))
        proc=self.adapter.processor
        self.runtime=dict(schema=SCHEMA,implementation=self.implementation,model_key=self.model,
            adapter=self.adapter.metadata,torch=torch.__version__,transformers=transformers.__version__,
            pillow=PIL.__version__,numpy=np.__version__,pandas=pd.__version__,cuda=torch.version.cuda,
            gpu=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            processor=serial(proc.to_dict()),image_processor=serial(proc.image_processor.to_dict()),
            chat_template=proc.chat_template,tokenizer_chat_template=proc.tokenizer.chat_template,
            generation_config=serial(self.adapter.model.generation_config.to_dict()),
            generation_overrides=self.config['generation'],
            toolkit={str(p.relative_to(self.toolkit)).replace('\\','/'):file_hash(p)
                     for p in sorted((self.toolkit/'src/visual_torque_toolkit').rglob('*.py'))},
            support_sha256=file_hash(self.shared/'support.json'),protocol_sha256=file_hash(self.shared/'protocol.json'),
            image_manifest_sha256=file_hash(self.shared/'image_manifest.json'))
        self.runtime_sha=freeze(self.folder/'runtime_identity.json',self.runtime)
        print('Model ready. Actual generated completions are the sole answer measure.')

    def forward(self,t):
        import torch
        scene=self.scenes[t['scene_id']];prompt=prompt_for(scene,t)
        content=[]
        image_hash=None
        if not t['text_only']:
            name=scene['scene_id']+'__ids.png'
            content.append(dict(type='image',image=str(self.images/name)))
            image_hash=self.image_manifest[name]['sha256']
        content.append(dict(type='text',text=prompt))
        inputs=self.adapter._to_device(self.adapter._template([[dict(role='user',content=content)]],add_generation_prompt=True))
        width=int(inputs['input_ids'].shape[1])
        fingerprint={}
        for k,v in inputs.items():
            if torch.is_tensor(v):
                a=v.detach().cpu().contiguous()
                fingerprint[k]=dict(shape=list(a.shape),dtype=str(a.dtype),
                    sha256=hashlib.sha256(a.view(torch.uint8).numpy().tobytes()).hexdigest())
        budget=self.config['generation'][('component' if t['task']!='judgment' else 'structured' if t['mode']=='structured' else 'direct')+'_tokens']
        start=time.monotonic()
        with torch.inference_mode():
            out=self.adapter.model.generate(**dict(inputs),do_sample=False,num_beams=1,num_return_sequences=1,
                use_cache=True,max_new_tokens=budget,pad_token_id=self.adapter.processor.tokenizer.pad_token_id,
                return_dict_in_generate=True,output_scores=False,output_logits=False)
        ids=out.sequences[0,width:].detach().cpu().tolist()
        eos=self.adapter.model.generation_config.eos_token_id
        eos=[eos] if isinstance(eos,int) else list(eos or [])
        raw=self.adapter.processor.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False).strip()
        record=dict(trial_id=t['trial_id'],raw=raw,completed=bool(ids and ids[-1] in eos),
            token_ids=ids,max_new_tokens=budget,eos_token_ids=eos,prompt=prompt,
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),image_sha256=image_hash,
            prepared_inputs=fingerprint,input_sha256=digest(fingerprint),
            input_token_ids=inputs['input_ids'][0].detach().cpu().tolist(),
            elapsed_seconds=time.monotonic()-start,runtime_sha256=self.runtime_sha)
        del out,inputs
        return record

    def run_phase(self,phase):
        assert phase in COUNTS and self.runtime_sha is not None
        if phase=='validation':
            assert (self.folder/'development/completion.json').is_file(), 'Run development cell first; no prompt selection occurs automatically'
        fs=[f for f in self.families if f['phase']==phase]
        planned=[t for f in fs for t in trials(f)]
        chunks={}
        for t in planned:
            key=t['scene_id'] if t['task']=='judgment' else t['family_id']+'__components'
            chunks.setdefault(key,[]).append(t)
        phase_dir=self.folder/phase;store_dir=phase_dir/'blocks';store_dir.mkdir(parents=True,exist_ok=True)
        cache=self.local/self.model/self.runtime_sha/phase;cache.mkdir(parents=True,exist_ok=True)
        all_records=[];invalid=0
        bar=tqdm(total=len(planned),desc=f'{self.model} {phase}',unit='answer')
        try:
            for key,items in chunks.items():
                path=store_dir/(key+'.json');local=cache/(key+'.json')
                identity=digest([self.runtime_sha,phase,items])
                existing=None
                for p in (path,local):
                    if p.exists():
                        existing=read_json(p)
                        assert existing['identity']==identity, 'Saved block identity mismatch'
                        break
                records=[] if existing is None else existing['records']
                assert len({r['trial_id'] for r in records})==len(records)
                by_id={r['trial_id']:r for r in records}
                assert set(by_id)<=set(t['trial_id'] for t in items)
                if path.exists(): assert len(records)==len(items), 'Persistent block must be complete'
                for t in items:
                    if t['trial_id'] not in by_id:
                        r=self.forward(t);records.append(r);by_id[t['trial_id']]=r
                        save_json(local,dict(identity=identity,records=records))
                    else:
                        r=by_id[t['trial_id']]
                    assert r['runtime_sha256']==self.runtime_sha
                    scored=score_record(r,t,self.scenes[t['scene_id']],self.model)
                    invalid+=int(not scored['valid'])
                    bar.update(1);bar.set_postfix(invalid=invalid,block=key[-14:])
                if not path.exists(): save_json(path,dict(identity=identity,records=records))
                all_records.extend(records)
        finally:
            bar.close()
        summary,tests=analyze(all_records,fs,self.model,phase_dir/'analysis')
        completion=dict(status='complete',phase=phase,n_trials=len(planned),runtime_sha256=self.runtime_sha,
            support_sha256=file_hash(self.shared/'support.json'),protocol_sha256=file_hash(self.shared/'protocol.json'),
            record_blocks={p.name:file_hash(p) for p in sorted(store_dir.glob('*.json'))},
            invalid_responses=invalid)
        freeze(phase_dir/'completion.json',completion)
        print(f'{phase}: {len(all_records)} completed generations; {invalid} unrecognized/truncated responses retained as incorrect.')
        quality=pd.read_csv(phase_dir/'analysis/generation_quality.csv')
        if (quality.valid<.95).any():
            print('Measurement review needed: inspect saved raw responses before interpreting accuracy. Do not silently change the parser for validation.')
            print(quality[quality.valid<.95].to_string(index=False))
        return summary,tests

    def export(self):
        path=self.root/f'VisualTorque_Relation_Use_{self.model}_Output.zip'
        export_archive(self.root,self.model,path,self.shared/'images.zip' if (self.shared/'images.zip').exists() else None)
        local=Path('/content')/path.name if Path('/content').exists() else self.local/path.name
        shutil.copyfile(path,local)
        print('Drive output:',path)
        print('Download copy:',local)
        from IPython.display import FileLink,display
        display(FileLink(str(local)))
        return path

    def release(self):
        if self.adapter is not None:
            import torch
            self.adapter=None;gc.collect();torch.cuda.empty_cache()
