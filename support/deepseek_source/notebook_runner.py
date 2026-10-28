"""Make child-process output visible in Colab and retain the actual failure in saved notebooks."""
from collections import deque
from pathlib import Path
import os
import subprocess


def run_logged(command,log_path):
    path=Path(log_path);path.parent.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    for key in ('PYTHONHOME','PYTHONPATH','VIRTUAL_ENV'):
        env.pop(key,None)
    env.update(PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1')
    tail=deque(maxlen=30)
    with path.open('a',encoding='utf-8') as log:
        log.write('\n--- New invocation ---\n');log.flush()
        with subprocess.Popen([str(x) for x in command],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,encoding='utf-8',errors='replace',bufsize=1,env=env) as process:
            for line in process.stdout:
                print(line,end='',flush=True);log.write(line);log.flush();tail.append(line)
            code=process.wait()
    if code:
        raise RuntimeError(f'运行失败（退出码 {code}）。完整日志：{path}\n'
                           +''.join(tail))
    print('日志已保存：',path,flush=True)
