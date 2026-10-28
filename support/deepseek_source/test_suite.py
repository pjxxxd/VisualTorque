"""Small synthetic software checks only. Never interpret these as experimental evidence."""
import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from core import HERE,load_kernels
from adapter import DeepSeekAdapter,output_hidden_tensor,replace_output_hidden,final_mask,score_records
import prediction
import readout
import repair
torch.set_num_threads(1)


def namespace():
    sys.path.insert(0,str(HERE/'toolkit/src'))
    n=dict(__name__='test_only')
    load_kernels(n,['followup_core.py','intervention_prediction_core.py','readout_edit_math.py',
        'aligned_readout_core.py','independent_core.py','followup_gpu.py','intervention_prediction_gpu.py'])
    n.update(torch=torch,np=np,output_hidden_tensor=output_hidden_tensor,replace_output_hidden=replace_output_hidden,
        final_mask=final_mask,score_records=score_records,CAPTURE_LAYERS=(20,26),RANDOM_SEEDS=(0,1,2,3))
    return n


def example(labels=('A','B','C')):
    return SimpleNamespace(example_id='fixture',scene_id='synthetic_scene',variant='fixture',split='dev',
        net_torque=2,problem_type='software_fixture',gold_label=labels[0],gold_semantic='left',
        labels=labels,option_semantics=('left','balanced','right'),
        label_to_semantic=dict(zip(labels,('left','balanced','right'))))


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__();self.linear=torch.nn.Linear(8,8,bias=False)
    def forward(self,h):return (h+0.05*self.linear(h),)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();torch.manual_seed(19)
        self.layers=torch.nn.ModuleList([Block() for _ in range(27)])
        self.embed=torch.nn.Embedding(20,8);self.norm=torch.nn.LayerNorm(8)
        self.head=torch.nn.Linear(8,10,bias=False);self.bfloat16();self.eval()
    def forward(self,input_ids,attention_mask,use_cache=False):
        h=self.embed(input_ids)
        for block in self.layers:h=block(h)[0]
        return SimpleNamespace(logits=self.head(self.norm(h)).float())


def mock_namespace():
    n=namespace();model=Model()
    def prepare(examples):
        assert len(examples)==1
        return dict(inputs=dict(input_ids=torch.tensor([[2,4,8]]),attention_mask=torch.ones(1,3,dtype=torch.long)),
                    examples=examples,token_by_label=dict(zip(examples[0].labels,[1,2,3])))
    n.update(adapter=SimpleNamespace(model=model),DECODER_LAYERS=model.layers,prepare_examples=prepare)
    return n


class CriticalTests(unittest.TestCase):
    def test_candidate_prefix_image_mask_and_symbol_alignment(self):
        a=DeepSeekAdapter.__new__(DeepSeekAdapter);a.device='cpu';a.plans={}
        a.model=SimpleNamespace(config=SimpleNamespace(language_config=SimpleNamespace(max_position_embeddings=100)))
        tokens={'A':4,'B':5,'C':6,'X':7,'Y':8,'Z':9,'1':10,'2':11,'3':12}
        inverse={v:k for k,v in tokens.items()}
        a.tokenizer=SimpleNamespace(decode=lambda ids:''.join(inverse.get(i,' ') for i in ids))
        def raw(ex,answer=''):
            ids=[1,2,3]+([0,tokens[answer]] if answer else [])
            return dict(input_ids=torch.tensor([ids]),attention_mask=torch.ones(1,len(ids),dtype=torch.long),
                images_seq_mask=torch.tensor([[True]+[False]*(len(ids)-1)]),images=torch.zeros(1,1,3,2,2),
                images_spatial_crop=torch.ones(1,1,2,dtype=torch.long))
        a.raw=raw
        for labels in [('A','B','C'),('X','Y','Z'),('1','2','3')]:
            ex=example(labels);p=a.prepare_examples([ex])
            self.assertEqual(p['inputs']['input_ids'].tolist(),[[1,2,3,0]])
            self.assertEqual(p['inputs']['images_seq_mask'].tolist(),[[True,False,False,False]])
            self.assertEqual(p['token_by_label'],{k:tokens[k] for k in labels})
        self.assertEqual(len({p[0] for p in a.plans.values()}),1)

    def test_native_hook_noop_replay_and_cleanup_on_failure(self):
        n=mock_namespace();ex=example()
        base,h=n['predictive_forward'](ex)
        row,post=n['predictive_forward'](ex,layer=20,payload=np.zeros_like(h[20]),
            expected_pre=h[20],expected_input=base['input_sha256'])
        self.assertEqual(base['candidate_logprobs'],row['candidate_logprobs'])
        np.testing.assert_array_equal(h[20],post[20])
        with self.assertRaises(AssertionError):
            n['predictive_forward'](ex,layer=20,payload=np.ones_like(h[20]),expected_pre=h[20]+1)
        self.assertTrue(all(not m._forward_hooks for m in n['DECODER_LAYERS']))
        row,post=n['predictive_forward'](ex,layer=20,payload=np.ones_like(h[20])*.2,expected_pre=h[20])
        self.assertGreater(row['write_audits']['20']['actual_norm'],0)
        self.assertEqual(post[20].shape,(1,8))

    def test_readout_correction_preserves_pairing_and_native_precision(self):
        n=mock_namespace();ex=example()
        probe=dict(layer=10,alpha=1.,mean=[0.]*8,scale=[1.]*8,coef=[.2]*8,intercept=0.)
        readout.install_readout(n,probe);n['PROBE_SHA']='fixture'
        rec,hidden=readout.forward(n,ex)
        rows,arrays=n['run_aligned_scene'](ex,rec,hidden['pre'],True)
        self.assertEqual(len(rows),6)
        self.assertTrue(rows[1]['achieved'])
        self.assertLessEqual(abs(rows[1]['decoded_post']-ex.net_torque),.05)
        self.assertEqual(arrays['oracle_post'].dtype,np.float32)
        self.assertTrue(all(not m._forward_hooks for m in n['DECODER_LAYERS']))

    def test_generic_three_path_removal_has_complete_grid_and_actual_norm_checks(self):
        n=mock_namespace();ex=example();layers=(10,20,23)
        base,h=n['final_forward'](ex,layers)
        u=np.arange(1,9,dtype=np.float32)/10;a=np.arange(8,0,-1,dtype=np.float32)/10;code=np.ones(8,np.float32)/5
        ops=repair.conditions(layers,1,u,a,code,h,['fixture'])
        self.assertEqual(len(ops),35)
        self.assertTrue(all(set(o)<={10,20,23} for _,_,o in ops))
        rows=[]
        for name,seed,operation in ops:
            rec=base if name=='zero' else n['final_forward'](ex,operations=operation)[0]
            rows.append(dict(condition=name,random_seed=seed,record=rec,valid=True))
        checked=repair.audit_rows(n,rows,layers)
        self.assertTrue(all(r['valid'] for r in checked if r['condition'].startswith('self_')))
        self.assertEqual(sum('paired_norm_error' in r for r in checked),20)
        # Projection of the induced change is restored to the untreated coordinate, not zeroed absolutely.
        h0=np.array([7.,3.]);he=np.array([9.,4.]);axis=np.array([1.,0.])
        post,_=n['coordinate_clamp'](he,h0,axis,lambda x:x)
        np.testing.assert_array_equal(post,[7.,4.])

    def test_seal_tampering_and_cache_identity_are_rejected(self):
        n=namespace()
        with tempfile.TemporaryDirectory(dir=HERE,prefix='software_test_') as folder:
            assert Path(folder).resolve().parent==HERE.resolve()
            path=Path(folder)/'frozen.json';n['freeze'](path,dict(dose=1))
            with self.assertRaises(AssertionError):n['freeze'](path,dict(dose=2))
            n['cached_json'](Path(folder)/'cache','x',dict(model='a'),lambda:[1])
            with self.assertRaises(AssertionError):n['cached_json'](Path(folder)/'cache','x',dict(model='b'),lambda:[2])
            # A real predictive record with known source scores, independent of gold.
            mapping=[dict(label=a,semantic=b,position=i) for i,(a,b) in enumerate(zip('ABC',['left','balanced','right']))]
            rows=[dict(module='content',layer=20,family_id=f'f{i}',trial_id=f't{i}',split='calibration',
                source_mapping=mapping,recipient_mapping=mapping,source_delta=[1.,0.,-1.],
                source_before_argmax=0,source_after_argmax=0,s0=[.2,-.3,.1],
                planned_norm=1.,relative_norm=.01,actual_delta=[.4,0.,-.4]) for i in range(10)]
            fitted=n['fit_all'](rows)
            test=[dict(rows[0],trial_id='validation',family_id='v',split='validation')]
            test[0].pop('actual_delta')
            sha=n['seal_predictions'](Path(folder)/'pred.json',test,fitted,dict(model='fixture'))
            self.assertEqual(sha,n['file_hash'](Path(folder)/'pred.json'))
            changed=[dict(test[0],source_delta=[-1.,0.,1.])]
            with self.assertRaises(AssertionError):n['seal_predictions'](Path(folder)/'pred.json',changed,fitted,dict(model='fixture'))

    def test_readout_selection_uses_training_and_development_only(self):
        n=namespace();rng=np.random.default_rng(9)
        ids=[str(i) for i in range(35)];values=rng.normal(size=(35,5));targets={s:float(values[i,0]*3) for i,s in enumerate(ids)}
        means={s:{'0':values[i].astype(np.float32),'1':rng.normal(size=5).astype(np.float32)} for i,s in enumerate(ids)}
        best,rows=readout.select_probe(n,means,targets,ids[:25],ids[25:],(0,1))
        self.assertEqual(best['layer'],0);self.assertEqual(len(rows),12)
        with self.assertRaises(AssertionError):readout.select_probe(n,means,targets,ids[:25],ids[24:],(0,1))

    def test_all_notebook_cells_and_sources_compile(self):
        for p in HERE.rglob('*.py'):
            if 'reference' not in p.parts:ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
        out=HERE.parents[1]/'output/notebooks/VisualTorque_DeepSeekVL2_Small'
        # HERE is project/deepseek_vl2; outer workspace is one level above project.
        out=HERE.parent.parent/'output/notebooks/VisualTorque_DeepSeekVL2_Small'
        notebooks=list(out.glob('*.ipynb'));self.assertEqual(len(notebooks),3)
        for p in notebooks:
            nb=json.loads(p.read_text(encoding='utf-8'))
            for c in nb['cells']:
                if c['cell_type']=='code':ast.parse(''.join(c['source']));self.assertFalse(c['outputs'])


if __name__=='__main__':unittest.main(verbosity=2)
