"""Regression tests for checksum compatibility and notebook-only output formatting."""
import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--deps-dir', type=Path)
args, remaining = parser.parse_known_args()
if args.deps_dir:
    sys.path.insert(0, str(args.deps_dir.resolve()))
from notebook_runtime_tools import ROOT, bootstrap_cell, load_catalog, load_checks, rewrite_source

checks = load_checks()
sys.modules['visualtorque_checks'] = checks
H = hashlib.sha256(b'visualtorque regression fixture').hexdigest()
SHORT = H[:10] + '…' + H[-6:]


class HashCompatibility(unittest.TestCase):
    def test_original_functions_and_keyword_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'读出.bin'
            for row in load_catalog().values():
                original = {'Path':Path, 'hashlib':hashlib}
                updated = {}
                exec(row['source'], original)
                exec(row['import'], updated)
                old, new = original[row['name']], updated[row['name']]
                if row['name'] == 'normalize_sha':
                    values = ['sha256:'+H, H, None, 17]
                elif row['name'] == 'bytes_sha':
                    values = [b'', b'\x00\xff\r\n', bytearray(b'torque')]
                else:
                    values = [b'', b'\x00\xff\r\n', bytes(range(256))*8193]
                for value in values:
                    with self.subTest(function=row['name'], size=len(value) if hasattr(value,'__len__') else 0):
                        if row['name'] not in ('normalize_sha','bytes_sha'):
                            p.write_bytes(value)
                            value = p
                        self.assertEqual(old(value), new(value))
                        argument = old.__code__.co_varnames[0]
                        self.assertEqual(old(**{argument:value}), new(**{argument:value}))
                if row['name'] not in ('normalize_sha','bytes_sha'):
                    with self.assertRaises(FileNotFoundError):
                        new(Path(directory)/'missing.bin')

    def test_corrupted_file_still_fails_existing_assertion(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'frozen_directions.bin'
            p.write_bytes(b'frozen direction')
            expected = checks.file_hash(p)
            assert checks.file_hash(p) == expected
            p.write_bytes(b'different direction')
            with self.assertRaises(AssertionError):
                assert checks.file_hash(p) == expected

    def test_only_known_functions_are_replaced(self):
        for row in load_catalog().values():
            before = 'import hashlib\n' + row['source'] + '\nassert 2 + 2 == 4\n'
            after, keys = rewrite_source(before, load_catalog())
            self.assertEqual(after, 'import hashlib\n' + row['import'] + '\n\nassert 2 + 2 == 4\n')
            self.assertEqual(len(keys), 1)
        custom = 'def file_hash(path):\n    return path.name\n'
        self.assertEqual(rewrite_source(custom, load_catalog()), (custom, []))


class DisplayFormatting(unittest.TestCase):
    def test_text_lookup_and_values(self):
        lookup = {}
        self.assertEqual(checks.compact_text('SHA256: '+H,lookup),'SHA256: '+SHORT)
        self.assertEqual(lookup[SHORT],H)
        self.assertEqual(len(H),64)
        self.assertEqual(checks.compact_text('accuracy=0.7139; repairs=71'), 'accuracy=0.7139; repairs=71')
        self.assertEqual(checks.compact_text('x'+H+'y'), 'x'+H+'y')

    def test_html_links_scripts_styles_and_media_are_preserved(self):
        html = '<a href="/'+H+'">'+H+'</a><script>const h="'+H+'";</script><style>#'+H+'{}</style>'
        actual = checks.compact_html(html)
        expected = html.replace('>'+H+'</a>', '>'+SHORT+'</a>')
        self.assertEqual(actual, expected)
        data = {'text/plain':H, 'text/html':html, 'application/json':{'sha256':H,'accuracy':0.75},'image/png':H}
        result = checks.compact_mime(data)
        self.assertEqual(data['text/plain'],H)
        self.assertEqual(result['text/plain'],SHORT)
        self.assertIs(result['application/json'],data['application/json'])
        self.assertEqual(result['image/png'],H)

    def test_markdown_links_remain_valid(self):
        text='Checksum '+H+' [artifact](https://example.invalid/'+H+')'
        self.assertEqual(checks.compact_markdown(text),'Checksum '+SHORT+' [artifact](https://example.invalid/'+H+')')

    def test_multiline_html_and_split_hashes(self):
        html = ['<script>\n', 'const h="'+H+'";\n', '</script>\n', '<p>'+H+'</p>\n']
        self.assertEqual(''.join(checks.compact_value(html,'text/html')),
                         ''.join(html).replace('<p>'+H+'</p>','<p>'+SHORT+'</p>'))
        self.assertEqual(''.join(checks.compact_value([H[:30],H[30:]])), SHORT)

    def test_stream_protocol_and_redirected_files(self):
        target=io.StringIO()
        wrapped=checks._DisplayStream(target)
        self.assertEqual(wrapped.write('hash='+H+'\n'),len('hash='+H+'\n'))
        wrapped.flush()
        self.assertEqual(target.getvalue(),'hash='+SHORT+'\n')
        archive=io.StringIO()
        with contextlib.redirect_stdout(archive):
            print(H)
        self.assertEqual(archive.getvalue(),H+'\n')


class KernelIntegration(unittest.TestCase):
    def test_standalone_notebook_and_deepseek_log(self):
        try:
            import jupyter_client
        except ImportError:
            self.skipTest('Install IPython and ipykernel to run the kernel smoke test.')
        with tempfile.TemporaryDirectory() as directory:
            environment=dict(os.environ)
            if args.deps_dir:
                environment['PYTHONPATH']=str(args.deps_dir.resolve())
            environment['PYDEVD_DISABLE_FILE_VALIDATION']='1'
            kernel=jupyter_client.KernelManager(connection_file=str(Path(directory)/'kernel.json'))
            kernel.start_kernel(cwd=directory,env=environment)
            client=kernel.client()
            client.start_channels()
            try:
                client.wait_for_ready(timeout=45)
                def execute(source):
                    key=client.execute(source)
                    messages=[]
                    while True:
                        message=client.get_iopub_msg(timeout=45)
                        if message.get('parent_header',{}).get('msg_id')!=key:
                            continue
                        messages.append(message)
                        if message['msg_type']=='error':
                            self.fail(message['content']['ename']+': '+message['content']['evalue'])
                        if message['msg_type']=='status' and message['content']['execution_state']=='idle':
                            return messages
                execute(''.join(bootstrap_cell()['source']))
                messages=execute(
                    "from pathlib import Path\nimport sys, json, io\n"
                    "from contextlib import redirect_stdout\n"
                    "from IPython.display import display, HTML, JSON\n"
                    f"h={H!r}\n"
                    "print('Hash:',h)\nprint('Error hash:',h,file=sys.stderr)\n"
                    "display(HTML('<a href=\"/'+h+'\">'+h+'</a>'))\n"
                    "display(JSON({'sha256':h,'accuracy':0.75}))\n"
                    "Path('record.json').write_text(json.dumps({'sha256':h}))\n"
                    "archive=io.StringIO()\n"
                    "with redirect_stdout(archive):\n    print(h)\n"
                    "assert archive.getvalue().strip()==h\n"
                    "assert json.loads(Path('record.json').read_text())['sha256']==h\n"
                    "h\n")
                streams=''.join(m['content']['text'] for m in messages if m['msg_type']=='stream')
                self.assertIn(SHORT,streams)
                self.assertNotIn(H,streams)
                rendered=[m['content']['data'] for m in messages if m['msg_type'] in ('execute_result','display_data')]
                self.assertTrue(any(d.get('text/plain')==repr(SHORT) for d in rendered))
                self.assertTrue(any(d.get('application/json',{}).get('sha256')==H for d in rendered))
                self.assertTrue(any('href="/'+H+'"' in d.get('text/html','') and '>'+SHORT+'<' in d['text/html'] for d in rendered))
                # The original DeepSeek launcher writes full logs before any notebook formatting.
                runner=(ROOT/'support/deepseek_source/notebook_runner.py').read_text(encoding='utf-8')
                messages=execute(f"exec({runner!r})\n"
                                 "run_logged([sys.executable,'-c','print('+repr(h)+')'],Path('stage.log'))\n"
                                 "assert h in Path('stage.log').read_text()\n"
                                 "print('Finished:',h)\n")
                streams=''.join(m['content']['text'] for m in messages if m['msg_type']=='stream')
                self.assertIn(SHORT,streams)
                self.assertNotIn(H,streams)
                # Re-running setup must not stack stream wrappers or break subsequent cells.
                execute(''.join(bootstrap_cell()['source']))
                execute("assert sys.stdout._visualtorque_stream\nassert not getattr(sys.stdout._stream,'_visualtorque_stream',False)\n")
                # Installation cells ask for Run all after restarting the runtime.
                kernel.restart_kernel(now=True)
                client.wait_for_ready(timeout=45)
                execute(''.join(bootstrap_cell()['source']))
                messages=execute("from visualtorque_checks import file_hash\nfrom pathlib import Path\n"
                                 "assert len(file_hash(Path('record.json')))==64\n"
                                 f"print({H!r})\n")
                streams=''.join(m['content']['text'] for m in messages if m['msg_type']=='stream')
                self.assertIn(SHORT,streams)
                self.assertNotIn(H,streams)
            finally:
                client.stop_channels()
                kernel.shutdown_kernel(now=True)


if __name__=='__main__':
    unittest.main(argv=[sys.argv[0],*remaining],verbosity=2)
