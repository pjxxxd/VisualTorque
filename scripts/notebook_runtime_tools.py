"""Build and validate the shared notebook utility cell."""
import ast
import base64
import copy
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_checks():
    spec = importlib.util.spec_from_file_location('visualtorque_checks', ROOT/'support/visualtorque_checks.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_catalog():
    return json.loads((ROOT/'docs/runtime_helpers.json').read_text(encoding='utf-8'))


def rewrite_source(source, catalog):
    lines = source.splitlines(keepends=True)
    if lines and lines[0].lstrip().startswith('%%') and not lines[0].startswith('%%cell_progress'):
        return source, []
    masked = ''.join('\n' if line.lstrip().startswith(('%', '!')) else line for line in lines)
    try:
        tree = ast.parse(masked)
    except SyntaxError:
        return source, []
    changes = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.decorator_list:
            continue
        original = ''.join(lines[node.lineno-1:node.end_lineno])
        key = hashlib.sha256(original.encode()).hexdigest()
        if key in catalog:
            changes.append((node.lineno-1, node.end_lineno, catalog[key]['import'], key))
    for start, end, replacement, key in reversed(changes):
        lines[start:end] = [replacement + '\n']
    return ''.join(lines), [key for _, _, _, key in changes]


def transform_cell(cell, catalog, checks):
    result = copy.deepcopy(cell)
    source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    source, replacements = rewrite_source(source, catalog)
    if replacements:
        result['source'] = source.splitlines(keepends=True) if isinstance(cell['source'], list) else source
    hashes = {}
    for output in result.get('outputs', []):
        if 'text' in output:
            output['text'] = checks.compact_value(output['text'], hashes=hashes)
        if 'data' in output:
            output['data'] = checks.compact_mime(output['data'], hashes)
    if hashes:
        result.setdefault('metadata', {}).setdefault('visualtorque', {})['display_hashes'] = hashes
    return result, replacements, hashes


def bootstrap_cell(minor=5):
    payload = base64.b64encode(gzip.compress((ROOT/'support/visualtorque_checks.py').read_bytes(), mtime=0)).decode()
    source = (
        '# Notebook utilities: support/visualtorque_checks.py\n'
        'import base64 as _vt_base64, gzip as _vt_gzip, sys as _vt_sys, types as _vt_types\n'
        f'_vt_source = _vt_gzip.decompress(_vt_base64.b64decode("{payload}"))\n'
        '_vt_checks = _vt_types.ModuleType("visualtorque_checks")\n'
        'exec(compile(_vt_source, "<visualtorque_checks>", "exec"), _vt_checks.__dict__)\n'
        '_vt_sys.modules["visualtorque_checks"] = _vt_checks\n'
        '_vt_checks.install_display()\n'
        'del _vt_source\n')
    cell = {'cell_type':'code', 'execution_count':None, 'outputs':[],
            'metadata':{'jupyter':{'source_hidden':True}, 'tags':['hide-input'],
                        'visualtorque':{'role':'notebook_utilities'}},
            'source':source.splitlines(keepends=True)}
    if minor >= 5:
        cell['id'] = 'visualtorque-utilities'
    return cell


def check_bootstrap(nb):
    cells = [c for c in nb['cells'] if c.get('metadata',{}).get('visualtorque',{}).get('role')=='notebook_utilities']
    if len(cells) != 1:
        return False
    expected = bootstrap_cell(nb.get('nbformat_minor',5))
    # Counts and outputs may change when a user executes the setup cell.
    return cells[0]['source'] == expected['source']


def write_json(path, value, indent=2):
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=indent) + '\n').encode('utf-8'))


def refresh_manifests():
    manifest_path = ROOT/'docs/notebook_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for entry in manifest:
        payload = (ROOT/entry['destination']).read_bytes()
        nb = json.loads(payload)
        entry.update(sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload),
                     cells=len(nb['cells']), code_cells=sum(c['cell_type']=='code' for c in nb['cells']))
    write_json(manifest_path, manifest)
    lookup = {entry['destination']:entry for entry in manifest}
    path = ROOT/'docs/notebook_provenance.json'
    provenance = json.loads(path.read_text(encoding='utf-8'))
    for entry in provenance:
        current = lookup[entry['destination']]
        entry.update(repository_sha256=current['sha256'], repository_cells=current['cells'])
    write_json(path, provenance)
    path = ROOT/'docs/source_to_copy.csv'
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    for row in rows:
        for key in ('sha256','bytes','code_cells'):
            row[key] = lookup[row['destination']][key]
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sync_helpers():
    count = 0
    for path in sorted((ROOT/'notebooks').rglob('*.ipynb')):
        nb = json.loads(path.read_text(encoding='utf-8'))
        cells = [c for c in nb['cells'] if c.get('metadata',{}).get('visualtorque',{}).get('role')=='notebook_utilities']
        if len(cells) != 1:
            raise ValueError('Expected one utility cell: ' + str(path))
        expected = bootstrap_cell(nb.get('nbformat_minor',5))
        if cells[0]['source'] != expected['source']:
            cells[0]['source'] = expected['source']
            write_json(path, nb, indent=1)
            count += 1
    refresh_manifests()
    return count


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sync', action='store_true', help='Update embedded helper copies and file manifests.')
    args = parser.parse_args()
    if args.sync:
        print('Updated utility cells:', sync_helpers())
    else:
        parser.print_help()
