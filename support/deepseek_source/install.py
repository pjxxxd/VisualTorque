"""Create an isolated Linux CUDA environment without replacing Colab's notebook kernel."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
SOURCE_REVISION='ef9f91e2b6426536b83294c11742c27be66361b1'
EXPERIMENT_PYTHON='3.11.11'
UV_VERSION='0.6.9'
DEFAULT_ENV='/content/visualtorque_deepseek_py311_cu128'
GPU_STACK=dict(torch='2.9.1',torchvision='0.24.1',xformers='0.0.33.post2',cuda='12.8')
WHEEL_INDEX='https://download.pytorch.org/whl/cu128'


def clean_environment():
    """Do not inherit a notebook's Python import paths into another interpreter."""
    result=os.environ.copy()
    for key in ('PYTHONHOME','PYTHONPATH','VIRTUAL_ENV'):
        result.pop(key,None)
    result['PYTHONNOUSERSITE']='1'
    result['PYTHONUNBUFFERED']='1'
    return result


def environment_python(env):
    return Path(env)/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')


def python_version(python):
    return subprocess.check_output([str(python),'-c','import platform; print(platform.python_version())'],
        env=clean_environment(),text=True).strip()


def ensure_python(env):
    """Provision the pinned experiment interpreter even when Colab runs Python 3.13+."""
    env=Path(env).resolve(); python=environment_python(env)
    if python.is_file():
        version=python_version(python)
        if version!=EXPERIMENT_PYTHON:
            raise RuntimeError(f'{env} contains Python {version}; expected {EXPERIMENT_PYTHON}. '
                f'Use the new notebook default ENV ({DEFAULT_ENV}).')
        subprocess.check_call([str(python),'-m','ensurepip','--upgrade'],env=clean_environment())
        return python
    bootstrap=env.parent/'visualtorque_uv_bootstrap'
    subprocess.check_call([sys.executable,'-m','pip','install','--disable-pip-version-check',
        '--target',str(bootstrap),'--upgrade',f'uv=={UV_VERSION}'],env=clean_environment())
    uv_env=clean_environment()
    uv_env.update(PYTHONPATH=str(bootstrap),
        UV_PYTHON_INSTALL_DIR=str(env.parent/'visualtorque_python_runtimes'),
        UV_CACHE_DIR=str(env.parent/'visualtorque_uv_cache'))
    uv=[sys.executable,'-m','uv']
    print(f'Notebook Python: {sys.version.split()[0]}; installing experiment Python {EXPERIMENT_PYTHON}.',flush=True)
    subprocess.check_call(uv+['python','install',EXPERIMENT_PYTHON],env=uv_env)
    subprocess.check_call(uv+['venv','--python',EXPERIMENT_PYTHON,'--seed',str(env)],env=uv_env)
    assert python_version(python)==EXPERIMENT_PYTHON
    return python


def environment_identity():
    return dict(python=EXPERIMENT_PYTHON,bootstrap_uv=UV_VERSION,
        requirements=hashlib.sha256((HERE/'requirements.txt').read_bytes()).hexdigest(),
        gpu_check=hashlib.sha256((HERE/'gpu_check.py').read_bytes()).hexdigest(),
        source=SOURCE_REVISION,**GPU_STACK)


def verify_gpu(python,env):
    # cuda.is_available() only checks device discovery; it does not launch a CUDA kernel.
    subprocess.check_call([str(python),'-u',str(HERE/'gpu_check.py'),
        '--report',str(Path(env)/'gpu_check.json')],env=clean_environment())


def install(env):
    assert sys.platform.startswith('linux'),'Use Colab or a Linux CUDA server for the GPU environment.'
    env=Path(env).resolve(); python=environment_python(env)
    identity=environment_identity()
    marker=env/'visualtorque_environment.json'
    if marker.is_file():
        assert json.loads(marker.read_text())==json.loads(json.dumps(identity)), 'Environment differs; use a fresh /content runtime.'
        assert python.is_file() and python_version(python)==EXPERIMENT_PYTHON
        verify_gpu(python,env)
        return python
    python=ensure_python(env)
    def pip(*args):subprocess.check_call([str(python),'-m','pip',*args],env=clean_environment())
    pip('install','--upgrade','pip==25.0.1','setuptools==75.8.2','wheel==0.45.1')
    pip('install',f"torch=={GPU_STACK['torch']}",f"torchvision=={GPU_STACK['torchvision']}",
        '--index-url',WHEEL_INDEX)
    pip('install',f"xformers=={GPU_STACK['xformers']}",'--no-deps','--index-url',WHEEL_INDEX)
    pip('install','-r',str(HERE/'requirements.txt'))
    # Keep the explicit, matched binary versions above; the official metadata pins torch 2.0.1.
    # Install inspected source without letting its dependency resolver replace this stack.
    pip('install','--no-deps',f'git+https://github.com/deepseek-ai/DeepSeek-VL2.git@{SOURCE_REVISION}')
    verify_gpu(python,env)
    marker.write_text(json.dumps(identity,indent=2)+'\n')
    return python


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--env',default=DEFAULT_ENV)
    args=p.parse_args();print('Experiment Python:',install(args.env))
