"""Regression checks for interpreter isolation and visible notebook errors; no model downloads."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import install
from gpu_check import validate_device
from notebook_runner import run_logged


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent,prefix='install_test_')
        self.root=Path(self.temp.name)
        assert self.root.resolve().parent==Path(__file__).resolve().parent

    def tearDown(self):self.temp.cleanup()

    def test_python_313_host_bootstraps_the_pinned_311_interpreter(self):
        target=self.root/'env'
        with patch.object(install.sys,'version_info',(3,13,0)),patch.object(install.sys,'version','3.13.0 (test host)'),\
             patch.object(install.sys,'platform','linux'),\
             patch.object(install,'python_version',return_value='3.11.11'),\
             patch.object(install.subprocess,'check_call') as call:
            result=install.ensure_python(target)
        self.assertEqual(result,target/'bin/python')
        commands=[c.args[0] for c in call.call_args_list]
        self.assertTrue(any(c[-3:]==['python','install','3.11.11'] for c in commands))
        self.assertEqual(commands[-1][-5:],['venv','--python','3.11.11','--seed',str(target)])
        self.assertEqual(call.call_args_list[-1].kwargs['env']['UV_PYTHON_INSTALL_DIR'],str(self.root/'visualtorque_python_runtimes'))

    def test_incompatible_existing_environment_is_not_overwritten(self):
        target=self.root/'env';python=install.environment_python(target);python.parent.mkdir(parents=True);python.write_text('sentinel')
        with patch.object(install,'python_version',return_value='3.13.0'),\
             patch.object(install.subprocess,'check_call') as call:
            with self.assertRaisesRegex(RuntimeError,'expected 3.11.11'):install.ensure_python(target)
        call.assert_not_called();self.assertEqual(python.read_text(),'sentinel')

    def test_dependency_failure_does_not_mark_installation_complete(self):
        target=self.root/'env';target.mkdir()
        with patch.object(install.sys,'platform','linux'),\
             patch.object(install,'ensure_python',return_value=target/'bin/python'),\
             patch.object(install.subprocess,'check_call',side_effect=subprocess.CalledProcessError(1,['pip'])):
            with self.assertRaises(subprocess.CalledProcessError):install.install(target)
        self.assertFalse((target/'visualtorque_environment.json').exists())

    def test_child_stdout_and_stderr_are_visible_and_retained_on_failure(self):
        path=self.root/'logs/install.log';screen=io.StringIO()
        command=[sys.executable,'-u','-c',"import sys; print('fixture stdout'); print('specific cause',file=sys.stderr); sys.exit(7)"]
        with redirect_stdout(screen),self.assertRaisesRegex(RuntimeError,'specific cause'):
            run_logged(command,path)
        self.assertIn('fixture stdout',screen.getvalue());self.assertIn('specific cause',screen.getvalue())
        self.assertIn('specific cause',path.read_text(encoding='utf-8'))

    def test_gpu_kernel_failure_does_not_mark_installation_complete(self):
        target=self.root/'env';target.mkdir()
        with patch.object(install.sys,'platform','linux'),\
             patch.object(install,'ensure_python',return_value=target/'bin/python'),\
             patch.object(install.subprocess,'check_call'),\
             patch.object(install,'verify_gpu',side_effect=RuntimeError('no kernel image')):
            with self.assertRaisesRegex(RuntimeError,'no kernel image'):install.install(target)
        self.assertFalse((target/'visualtorque_environment.json').exists())

    def test_cached_environment_rechecks_gpu_without_reinstalling(self):
        target=self.root/'env';python=target/'bin/python';python.parent.mkdir(parents=True)
        python.write_text('fixture')
        (target/'visualtorque_environment.json').write_text(json.dumps(install.environment_identity()))
        with patch.object(install.sys,'platform','linux'),\
             patch.object(install,'python_version',return_value='3.11.11'),\
             patch.object(install,'ensure_python') as create,\
             patch.object(install,'verify_gpu',side_effect=RuntimeError('different GPU now')) as check:
            with self.assertRaisesRegex(RuntimeError,'different GPU now'):install.install(target)
        create.assert_not_called();check.assert_called_once_with(python,target)

    def test_blackwell_installer_uses_matched_cuda_128_wheels(self):
        target=self.root/'env';target.mkdir();marker=target/'visualtorque_environment.json'
        def check(python,env):self.assertFalse(marker.exists())
        with patch.object(install.sys,'platform','linux'),\
             patch.object(install,'ensure_python',return_value=target/'bin/python'),\
             patch.object(install.subprocess,'check_call') as calls,\
             patch.object(install,'verify_gpu',side_effect=check):install.install(target)
        commands=[c.args[0] for c in calls.call_args_list]
        torch_command=next(c for c in commands if 'torch==2.9.1' in c)
        xformers_command=next(c for c in commands if 'xformers==0.0.33.post2' in c)
        self.assertIn('torchvision==0.24.1',torch_command)
        for command in (torch_command,xformers_command):
            self.assertIn('https://download.pytorch.org/whl/cu128',command)
        self.assertTrue(marker.is_file())

    def test_device_discovery_is_not_mistaken_for_blackwell_compatibility(self):
        def fake(capability,cuda='12.8',available=True):
            return SimpleNamespace(__version__='test',version=SimpleNamespace(cuda=cuda),
                cuda=SimpleNamespace(is_available=lambda:available,
                    get_device_capability=lambda _:capability,is_bf16_supported=lambda:True,
                    get_device_name=lambda _:'synthetic device',get_arch_list=lambda:['sm_120'],
                    get_device_properties=lambda _:SimpleNamespace(total_memory=96*2**30)))
        with self.assertRaisesRegex(RuntimeError,'requires the new cu128'):
            validate_device(fake((12,0),'12.4'))
        self.assertEqual(validate_device(fake((12,0)))['capability'],[12,0])
        self.assertEqual(validate_device(fake((8,0)))['capability'],[8,0])
        with self.assertRaisesRegex(RuntimeError,'native BF16'):validate_device(fake((7,5)))
        with self.assertRaisesRegex(RuntimeError,'No CUDA GPU'):validate_device(fake((12,0),available=False))

    def test_new_notebook_keeps_old_family_selection_but_separates_results(self):
        from build_bundle import notebook
        previous=self.root/'results/deepseek_vl2_small/main/prediction/support.json'
        previous.parent.mkdir(parents=True);previous.write_text('frozen fixture')
        code=''.join(notebook('fixture','fixture',[],'')['cells'][1]['source'])
        code=code.replace("Path('/content/drive/MyDrive/VisualTorque')",'Path('+repr(str(self.root))+')')
        fake_colab=SimpleNamespace(drive=SimpleNamespace(mount=lambda _:None))
        namespace={}
        with patch.dict(sys.modules,{'google.colab':fake_colab}),redirect_stdout(io.StringIO()):
            exec(compile(code,'notebook_config','exec'),namespace)
        self.assertEqual(namespace['RUN_ID'],'blackwell_cu128')
        self.assertEqual(namespace['SOURCE_SUPPORT'],previous)
        self.assertEqual(previous.read_text(),'frozen fixture')


if __name__=='__main__':unittest.main(verbosity=2)
