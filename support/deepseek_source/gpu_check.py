"""Run real GPU kernels before checkpoint download. No scientific results or model weights."""
import argparse
import collections
import collections.abc
import importlib.metadata
import json
from pathlib import Path
import platform
import traceback


def validate_device(torch):
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU is available. Select a GPU runtime in Colab.')
    capability=tuple(torch.cuda.get_device_capability(0))
    cuda=tuple(int(x) for x in (torch.version.cuda or '0.0').split('.')[:2])
    if capability[0]>=10 and cuda<(12,8):
        raise RuntimeError(f'Blackwell GPU sm_{capability[0]}{capability[1]} requires the new cu128 environment; '
                           f'found torch {torch.__version__} / CUDA {torch.version.cuda}.')
    if capability[0]<8 or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This BF16 experiment requires native BF16 support (Ampere or newer).')
    return dict(gpu=torch.cuda.get_device_name(0),capability=list(capability),
                compiled_architectures=torch.cuda.get_arch_list(),
                vram_gib=round(torch.cuda.get_device_properties(0).total_memory/2**30,1))


def run_checks(report):
    import torch
    from install import GPU_STACK
    report.update(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda)
    report.update(validate_device(torch))
    for package in ('torch','torchvision','xformers'):
        actual=importlib.metadata.version(package)
        if actual.split('+')[0]!=GPU_STACK[package]:
            raise RuntimeError(f'{package} {actual} differs from pinned {GPU_STACK[package]}. Rerun the installer.')
        report[package]=actual
    if torch.version.cuda!=GPU_STACK['cuda']:
        raise RuntimeError(f"Expected CUDA {GPU_STACK['cuda']}, found {torch.version.cuda}.")
    print('GPU compatibility check:',json.dumps(report,ensure_ascii=False),flush=True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.manual_seed(7)
    report['checks']=[]
    with torch.inference_mode():
        # Reproduce the exact operation that failed in final_mask, then test BF16 GEMM.
        mask=torch.ones(32,device='cuda',dtype=torch.bool)
        selected=torch.zeros_like(mask);selected[-1]=True
        assert int(selected.sum())==1
        matrix=torch.ones((128,128),device='cuda',dtype=torch.bfloat16)
        product=matrix@matrix
        torch.testing.assert_close(product.float(),torch.full_like(product,128).float(),rtol=0,atol=0)
        torch.cuda.synchronize()
        report['checks'].append('bool_mask_and_bf16_matmul')
        print('PASS: bool mask and BF16 matrix multiplication',flush=True)

        for name in ('Mapping','MutableMapping','Sequence'):
            if not hasattr(collections,name):setattr(collections,name,getattr(collections.abc,name))
        from deepseek_vl2.models import DeepseekVLV2Processor,DeepseekVLV2ForCausalLM
        from deepseek_vl2.models.siglip_vit import Attention
        # Match Small's vision width / head dimension and one 384px crop's 27x27 patches.
        # This also executes xFormers' actual BF16 attention dispatch (head_dim=72).
        attention=Attention(dim=1152,num_heads=16,qkv_bias=True).to(device='cuda',dtype=torch.bfloat16).eval()
        tokens=torch.randn((1,729,1152),device='cuda',dtype=torch.bfloat16)
        first=attention(tokens);second=attention(tokens)
        torch.cuda.synchronize()
        assert first.shape==tokens.shape and bool(torch.isfinite(first).all())
        torch.testing.assert_close(first,second,rtol=0,atol=0)
        report['checks'].append('official_vision_attention_bf16_repeat')
        report['vision_input_shape']=list(tokens.shape)
        print('PASS: official vision attention, BF16, 72 dimensions per head',flush=True)
    report['status']='passed_synthetic_gpu_checks_not_full_model'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report',required=True)
    args=parser.parse_args();path=Path(args.report)
    report=dict(status='started',model_weights_loaded=False)
    try:
        run_checks(report)
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc())
        raise
    finally:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('GPU checks passed; model loading may now begin.',flush=True)


if __name__=='__main__':main()
