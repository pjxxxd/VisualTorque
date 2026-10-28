"""DeepSeek-specific singleton scoring; no changes to the Qwen/Gemma adapters."""
import collections
import collections.abc
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image

MODEL_ID = 'deepseek-ai/deepseek-vl2-small'
MODEL_REVISION = '6033e16432a1d771cf9fe4a6f894ff5e5e1459af'
SOURCE_REVISION = 'ef9f91e2b6426536b83294c11742c27be66361b1'
LAYER_PATH = 'language.model.layers'


def output_hidden_tensor(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def replace_output_hidden(output, hidden):
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    if isinstance(output, list):
        return [hidden] + output[1:]
    return hidden


def final_mask(prepared):
    mask = prepared['inputs']['attention_mask'][0].bool()
    assert bool(mask[-1]), 'Singleton prompt must end in a real text token.'
    out = torch.zeros_like(mask)
    out[-1] = True
    return out


def score_records(prepared, logits):
    ex = prepared['examples'][0]
    values = torch.log_softmax(logits[0, -1].float(), dim=-1)
    scores = [float(values[prepared['token_by_label'][s]]) for s in ex.labels]
    assert np.isfinite(scores).all()
    index = int(np.argmax(scores))
    return [dict(candidate_logprobs=scores, candidate_labels=list(ex.labels),
        candidate_semantics=list(ex.option_semantics),
        semantic_logprobs=dict(zip(ex.option_semantics, scores)),
        label_to_semantic=ex.label_to_semantic, gold_label=ex.gold_label,
        gold_semantic=ex.gold_semantic, predicted_label=ex.labels[index],
        predicted_semantic=ex.option_semantics[index],
        correct=ex.option_semantics[index] == ex.gold_semantic)]


class DeepSeekAdapter:
    def __init__(self, checkpoint=None):
        # attrdict 2.0.1 predates collections.abc's Python 3.10 migration.
        for name in ('Mapping', 'MutableMapping', 'Sequence'):
            if not hasattr(collections, name):
                setattr(collections, name, getattr(collections.abc, name))
        from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
        from deepseek_vl2.models.modeling_deepseek_vl_v2 import DeepseekVLV2Config
        from huggingface_hub import snapshot_download
        assert torch.cuda.is_available(), 'A CUDA GPU is required for real model experiments.'
        assert torch.cuda.is_bf16_supported(), 'Native BF16 GPU required; no silent quantization fallback.'
        self.device = torch.device('cuda:0')
        self.checkpoint = checkpoint or snapshot_download(MODEL_ID, revision=MODEL_REVISION)
        self.processor = DeepseekVLV2Processor.from_pretrained(self.checkpoint)
        self.tokenizer = self.processor.tokenizer
        config = DeepseekVLV2Config.from_pretrained(self.checkpoint)
        config._attn_implementation = 'eager'
        config.language_config._attn_implementation = 'eager'
        assert config.language_config.num_hidden_layers == 27
        self.model = DeepseekVLV2ForCausalLM.from_pretrained(self.checkpoint,
            config=config, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            attn_implementation='eager').to(self.device).eval()
        self.layers = self.model.language.model.layers
        assert len(self.layers) == 27
        self.plans = {}
        self.metadata = dict(model_id=MODEL_ID, resolved_revision=MODEL_REVISION,
            source_revision=SOURCE_REVISION, layer_path=LAYER_PATH,
            dtype='bfloat16', language_attention='eager', batch_size=1,
            score='single-token log probability after native chat template',
            quantized=False, cpu_offload=False)

    def raw(self, example, answer=''):
        with Image.open(example.image_path) as image:
            picture = image.convert('RGB')
        conversation = [dict(role='<|User|>', content='<image>\n'+example.prompt,
                             images=[str(example.image_path)]),
                        dict(role='<|Assistant|>', content=answer)]
        prepared = self.processor(conversations=conversation, images=[picture],
                                  force_batchify=True, inference_mode=True)
        return {k: getattr(prepared, k) for k in
                ('input_ids', 'attention_mask', 'images', 'images_seq_mask', 'images_spatial_crop')}

    def _candidate_plan(self, example, base=None):
        labels = tuple(example.labels)
        if labels in self.plans:
            return self.plans[labels]
        base = self.raw(example) if base is None else base
        prompt = base['input_ids'][0].tolist()
        suffixes = []
        for label in labels:
            full = self.raw(example, label)['input_ids'][0].tolist()
            assert full[:len(prompt)] == prompt, 'Completed answer changes the prompt prefix.'
            suffix = full[len(prompt):]
            assert suffix and self.tokenizer.decode(suffix).strip() == label
            suffixes.append(suffix)
        prefixes = [tuple(s[:-1]) for s in suffixes]
        assert len(set(prefixes)) == 1, 'Labels need a common prefix and one distinct scored token.'
        tokens = dict(zip(labels, [s[-1] for s in suffixes]))
        assert len(set(tokens.values())) == len(labels)
        self.plans[labels] = (prefixes[0], tokens)
        return self.plans[labels]

    def prepare_examples(self, examples):
        assert len(examples) == 1, 'All stages use singleton forwards.'
        inputs = self.raw(examples[0])
        prefix, tokens = self._candidate_plan(examples[0], inputs)
        if prefix:
            ids = torch.tensor([prefix], dtype=inputs['input_ids'].dtype)
            inputs['input_ids'] = torch.cat([inputs['input_ids'], ids], dim=1)
            inputs['attention_mask'] = torch.cat([inputs['attention_mask'],
                torch.ones_like(ids, dtype=inputs['attention_mask'].dtype)], dim=1)
            inputs['images_seq_mask'] = torch.cat([inputs['images_seq_mask'],
                torch.zeros_like(ids, dtype=torch.bool)], dim=1)
        assert inputs['input_ids'].shape == inputs['images_seq_mask'].shape
        assert inputs['input_ids'].shape[1] <= self.model.config.language_config.max_position_embeddings
        inputs = {k: v.to(self.device, dtype=torch.bfloat16) if k == 'images'
                  else v.to(self.device) for k, v in inputs.items()}
        return dict(inputs=inputs, examples=list(examples), token_by_label=tokens)

    def tail_scores(self, hidden, labels):
        prefix, tokens = self.plans[tuple(labels)]
        h = torch.as_tensor(np.asarray(hidden).copy(), device=self.device,
                            dtype=torch.bfloat16).reshape(1, 1, -1)
        with torch.inference_mode():
            logits = self.model.language.lm_head(self.model.language.model.norm(h)).float()
        scores = logits[0, -1, [tokens[s] for s in labels]].cpu().numpy()
        return scores - scores.mean()


def source_identity(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(root).rglob('*.py')) if not {'__pycache__','reference'} & set(p.parts)
            and not p.name.startswith('test_') and p.name not in ('build_bundle.py','fetch_reference.py')}
