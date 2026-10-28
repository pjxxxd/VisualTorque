"""Vendor stable existing kernels and build standalone Colab notebooks and a portable source ZIP."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent
OUT=PROJECT.parent/'output/notebooks/VisualTorque_DeepSeekVL2_Small'
KERNELS=['followup_core.py','intervention_prediction_core.py','intervention_prediction_io.py',
    'intervention_prediction_gpu.py','frozen_format_transfer_core.py','frozen_format_transfer_gpu.py',
    'followup_gpu.py','readout_edit_math.py','aligned_readout_core.py','aligned_readout_gpu.py','independent_core.py']


def extract(path,names):
    text=path.read_text(encoding='utf-8');lines=text.splitlines()
    defs={a.name:'\n'.join(lines[a.lineno-1:a.end_lineno]) for a in ast.parse(text).body if isinstance(a,(ast.FunctionDef,ast.ClassDef))}
    return '\n\n'.join(defs[name] for name in names)+'\n'


def source_payload():
    sources={p.name:p.read_bytes() for p in HERE.iterdir() if p.suffix in ('.py','.txt','.md')
             and p.name not in ('fetch_reference.py','test_official_interface.py')}
    for name in KERNELS:sources['kernels/'+name]=(PROJECT/'notebooks'/name).read_bytes()
    repair=extract(PROJECT/'notebooks/followup_repair_cells.py',['repair_prompt','new_example','two_mappings','code_key'])
    sources['kernels/repair_examples.py']=repair.encode()
    for p in (PROJECT/'toolkit').rglob('*'):
        if p.is_file() and p.suffix=='.py' and '__pycache__' not in p.parts:
            sources['toolkit/'+p.relative_to(PROJECT/'toolkit').as_posix()]=p.read_bytes()
    # No original images, historical results, model weights or downloaded reference source in this bundle.
    return sources


def notebook(name,title,stages,encoded):
    cells=[]
    def add(kind,text):
        cell=dict(cell_type=kind,metadata={},source=text.strip().splitlines(True),id=hashlib.sha256((name+text).encode()).hexdigest()[:12])
        if kind=='code':ast.parse(text);cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    add('markdown',title+'\n\n支持 Blackwell 的 cu128 环境；建议约 80GB 或更多显存，全 BF16。按顺序运行。每个实验阶段独立保存，断线后可重跑。'
        '\n自动配置独立的 Python 3.11.11，兼容 Python 3.13 Colab 内核，无需手动切换版本。'
        '\n默认 RUN_ID 为 blackwell_cu128，避免与旧 cu124 环境的记录混合；旧 main 中已冻结的图片与家族会优先复用。'
        '\n模型与环境首次下载较大。模型只在 GPU 运行时下载；此 notebook 不含模型权重。'
        '\n读出与修复实验要求 Drive 中原 Dataset v1、254 场景训练清单及共享测试清单。'
        '\n修改 RUN_ID 会新建一轮结果；不要根据测试效果修改层号、剂量、样本数。')
    add('code',"from google.colab import drive\ndrive.mount('/content/drive')\nfrom pathlib import Path\nPROJECT_ROOT=Path('/content/drive/MyDrive/VisualTorque')  # 改成你的实际项目目录\nRUN_ID='blackwell_cu128'  # 三份 notebook 使用相同值；请勿改回旧环境的 main\nSOURCE_SUPPORT=None  # 可手动指定已有 support.json\nassert PROJECT_ROOT.is_dir(), PROJECT_ROOT\nprevious_support=PROJECT_ROOT/'results/deepseek_vl2_small/main/prediction/support.json'\nif SOURCE_SUPPORT is None and previous_support.is_file():\n    SOURCE_SUPPORT=previous_support\n    print('Reuse frozen families/images from:',SOURCE_SUPPORT)\nprint('Results:',PROJECT_ROOT/'results/deepseek_vl2_small'/RUN_ID)")
    add('code',"import base64,io,zipfile,hashlib\nPACKAGE_ROOT=Path('/content/visualtorque_deepseek_package')\n"
        +"payload=base64.b64decode("+repr(encoded)+")\n"
        +"with zipfile.ZipFile(io.BytesIO(payload)) as z:\n    for member in z.infolist():\n        assert (PACKAGE_ROOT/member.filename).resolve().is_relative_to(PACKAGE_ROOT.resolve())\n        z.extract(member,PACKAGE_ROOT)\nprint('Experiment sources ready:',PACKAGE_ROOT)")
    add('code',"import sys,shutil\nsys.path.insert(0,str(PACKAGE_ROOT))\nfrom notebook_runner import run_logged\nENV=Path('/content/visualtorque_deepseek_py311_cu128')\nLOG_DIR=PROJECT_ROOT/'results/deepseek_vl2_small'/RUN_ID/'logs'\nprint('Colab kernel:',sys.version.split()[0],'; experiments use Python 3.11.11 / CUDA 12.8.')\ntry:\n    run_logged([sys.executable,'-u',str(PACKAGE_ROOT/'install.py'),'--env',str(ENV)],LOG_DIR/'install.log')\nfinally:\n    gpu_report=ENV/'gpu_check.json'\n    if gpu_report.is_file():\n        LOG_DIR.mkdir(parents=True,exist_ok=True)\n        shutil.copy2(gpu_report,LOG_DIR/'gpu_check.json')\nPYTHON=ENV/'bin/python'\ndef run_stage(stage):\n    command=[str(PYTHON),'-u',str(PACKAGE_ROOT/'run.py'),'--project',str(PROJECT_ROOT),'--run-id',RUN_ID,'--stage',stage]\n    if SOURCE_SUPPORT: command += ['--source-support',str(SOURCE_SUPPORT)]\n    run_logged(command,LOG_DIR/(stage+'.log'))\nrun_stage('preflight')")
    for stage in stages:
        add('markdown','运行阶段：`'+stage+'`')
        add('code',f"run_stage('{stage}')")
    add('markdown','即使前面的阶段中断，也可以运行下面一格打包已有结果。缺少完成标记时，报告会显示未完成。')
    add('code',"run_stage('export')\nfrom google.colab import files\nfiles.download(str(PROJECT_ROOT/'results/deepseek_vl2_small'/(RUN_ID+'_results.zip')))")
    return dict(cells=cells,nbformat=4,nbformat_minor=5,metadata=dict(accelerator='GPU',
        kernelspec=dict(name='python3',display_name='Python 3',language='python'),colab=dict(name=name),
        language_info=dict(name='python'),visualtorque=dict(model='deepseek-ai/deepseek-vl2-small',scientific_results_embedded=False)))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    sources=source_payload()
    # Materialize the identical payload locally for tests and a conventional CLI workflow.
    for name,data in sources.items():
        path=HERE/name
        if not path.exists() or path.read_bytes()!=data:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in sorted(sources.items()):z.writestr(name,data)
    encoded=base64.b64encode(buf.getvalue()).decode()
    specs=[('01_DeepSeek_Transfer_4_2_4_3.ipynb','# 01｜DeepSeek：映射收益与迁移预测（4.2–4.3）',['calibration','validation','formats']),
        ('02_DeepSeek_Readout_4_4.ipynb','# 02｜DeepSeek：净力矩读出与数值修准（4.4）',['readout-fit','readout-validation']),
        ('03_DeepSeek_Repair_4_5.ipynb','# 03｜DeepSeek：监督修复与后续方向移除（4.5）',['repair-development','repair-confirmation'])]
    outputs=[]
    for name,title,stages in specs:
        p=OUT/name;p.write_text(json.dumps(notebook(name,title,stages,encoded),ensure_ascii=False,indent=1)+'\n',encoding='utf-8');outputs.append(p)
    for name in ('README_ZH.md','VALIDATION.md'):
        if (HERE/name).exists():shutil.copy2(HERE/name,OUT/name);outputs.append(OUT/name)
    sourcezip=OUT/'experiment_source.zip';sourcezip.write_bytes(buf.getvalue());outputs.append(sourcezip)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs}
    (OUT/'manifest.json').write_text(json.dumps(hashes,indent=2)+'\n');outputs.append(OUT/'manifest.json')
    target=OUT.with_suffix('.zip')
    with zipfile.ZipFile(target.with_suffix('.tmp.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in outputs:z.write(p,p.name)
    target.with_suffix('.tmp.zip').replace(target)
    # Check embedded payload bytes against the separately delivered source archive.
    for name,_,_ in specs:
        nb=json.loads((OUT/name).read_text(encoding='utf-8'))
        cell=''.join(nb['cells'][2]['source'])
        tree=ast.parse(cell)
        encoded_literal=next(a.value.args[0].value for a in tree.body if isinstance(a,ast.Assign)
                             and any(isinstance(t,ast.Name) and t.id=='payload' for t in a.targets))
        assert base64.b64decode(encoded_literal)==sourcezip.read_bytes()
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        assert set(z.namelist())=={p.name for p in outputs}
    print('Bundle:',target)
    print('Standalone notebooks:',*[p.name for p in outputs if p.suffix=='.ipynb'],sep='\n')


if __name__=='__main__':main()
