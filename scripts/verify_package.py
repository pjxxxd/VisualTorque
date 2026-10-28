"""Check copied files and navigation without executing experiments (Python 3.9+)."""
import argparse
import csv
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(rel):
    return json.loads((ROOT / rel).read_text(encoding='utf-8'))


def verify(original_root=None, downloads=None):
    errors = []
    manifest = read_json('docs/notebook_manifest.json')
    support = read_json('docs/support_manifest.json')
    expected = {r['destination'] for r in manifest}
    actual = {p.relative_to(ROOT).as_posix() for p in (ROOT / 'notebooks').rglob('*.ipynb')}
    if len(expected) != len(manifest) or expected != actual:
        errors.append('Notebook inventory differs from manifest.')
    for r in manifest:
        path = ROOT / r['destination']
        if not path.is_file():
            errors.append('Notebook missing: ' + r['destination'])
            continue
        if digest(path) != r['sha256']:
            errors.append('Hash changed: ' + r['destination'])
        try:
            nb = json.loads(path.read_text(encoding='utf-8'))
            valid = nb.get('nbformat') == 4 and isinstance(nb.get('cells'), list) and isinstance(nb.get('metadata'), dict)
            if not valid:
                raise ValueError('invalid notebook container')
            if len(nb['cells']) != r['cells']:
                raise ValueError('cell count differs')
            for i, cell in enumerate(nb['cells']):
                valid = cell.get('cell_type') in ('code', 'markdown', 'raw') and isinstance(cell.get('source'), (str, list)) and isinstance(cell.get('metadata'), dict)
                if not valid or (cell['cell_type'] == 'code' and not isinstance(cell.get('outputs'), list)):
                    raise ValueError('invalid cell ' + str(i))
        except (ValueError, TypeError, KeyError) as exc:
            errors.append('Notebook structure: ' + r['destination'] + ': ' + str(exc))
    for r in support:
        path = ROOT / r['destination']
        if not path.is_file() or digest(path) != r['sha256']:
            errors.append('Support missing or changed: ' + r['destination'])
    with (ROOT / 'docs/source_to_copy.csv').open(encoding='utf-8-sig', newline='') as stream:
        sources = list(csv.DictReader(stream))
    if len(sources) != len({r['source'] for r in sources}):
        errors.append('Duplicate source paths.')
    lookup = {r['destination']: r for r in manifest}
    for r in sources:
        item = lookup.get(r['destination'])
        if not item or r['sha256'] != item['sha256'] or r['source'] not in item['sources']:
            errors.append('Invalid source mapping: ' + r['source'])
    expected_pairs = {(r['destination'], s) for r in manifest for s in r['sources']}
    if expected_pairs != {(r['destination'], r['source']) for r in sources}:
        errors.append('Source mapping is incomplete.')

    docs = [ROOT / 'README.md', *sorted((ROOT / 'notebooks').rglob('README.md')),
            *[ROOT / 'docs' / n for n in ('CLAIM_TO_NOTEBOOK.md', 'RUNNING.md', 'COVERAGE.md', 'VALIDATION.md', 'CHANGELOG_20261028.md')],
            ROOT / 'docs/original_run_guides/README.md', ROOT / 'evidence/deepseek/README.md',
            ROOT / 'docs/SIMPLIFICATION.md', ROOT / 'docs/EXPERIMENT_NOTES.md',
            *sorted((ROOT / 'docs/experiment_notes').rglob('*.md'))]
    links_checked = 0
    for path in docs:
        if not path.is_file():
            errors.append('Missing document: ' + path.relative_to(ROOT).as_posix())
            continue
        for value in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            url = urlsplit(value.strip('<>'))
            if url.scheme or not url.path:
                continue
            target = (path.parent / unquote(url.path)).resolve()
            links_checked += 1
            if not target.is_relative_to(ROOT) or not target.exists():
                errors.append('Broken link: ' + path.relative_to(ROOT).as_posix() + ' -> ' + value)

    for entry in manifest:
        path = ROOT / entry['destination']
        nb = json.loads(path.read_text(encoding='utf-8'))
        for cell in nb['cells']:
            if cell['cell_type'] != 'markdown':
                continue
            content = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            for value in re.findall(r'\]\(([^)]+)\)', content):
                url = urlsplit(value.strip('<>'))
                if url.scheme or not url.path:
                    continue
                target = (path.parent / unquote(url.path)).resolve()
                links_checked += 1
                if not target.is_relative_to(ROOT) or not target.exists():
                    errors.append('Broken notebook link: ' + entry['destination'] + ' -> ' + value)

    originals_checked = support_originals_checked = archives_checked = previous_checked = 0
    code_unchanged = reading_copies_checked = 0
    if original_root:
        def source_path(source):
            if source.startswith('Downloads/'):
                if downloads is None:
                    raise ValueError('--downloads is required for full source verification')
                return downloads / Path(source).name
            return original_root / source

        for r in sources:
            path = source_path(r['source'])
            if not path.is_file() or digest(path) != r.get('source_sha256', r['sha256']):
                errors.append('Original notebook missing or changed: ' + r['source'])
            originals_checked += 1
        for r in read_json('docs/archive_sources.json'):
            path = source_path(r['source'])
            if not path.is_file() or digest(path) != r['sha256']:
                errors.append('Original archive missing or changed: ' + r['source'])
            archives_checked += 1
        archives = {}
        try:
            for r in support:
                if 'archive' in r:
                    key = r['archive']
                    if key not in archives:
                        archives[key] = zipfile.ZipFile(source_path(key))
                    original_hash = hashlib.sha256(archives[key].read(r['member'])).hexdigest()
                else:
                    path = source_path(r['source'])
                    original_hash = digest(path) if path.is_file() else None
                if original_hash != r.get('source_sha256', r['sha256']):
                    errors.append('Original support missing or changed: ' + r['source'])
                support_originals_checked += 1
        finally:
            for archive in archives.values():
                archive.close()
        old = original_root / 'VisualTorque_GitHub'
        old_manifest = old / 'docs/notebook_manifest.json'
        if old_manifest.exists():
            hashes = {r.get('source_sha256', r['sha256']) for r in manifest}
            for r in json.loads(old_manifest.read_text(encoding='utf-8')):
                path = old / r['destination']
                if r['sha256'] not in hashes or not path.is_file() or digest(path) != r['sha256']:
                    errors.append('Previous notebook missing or changed: ' + r['destination'])
                previous_checked += 1
        baseline = original_root / 'VisualTorque_GitHub_20261028'
        for change in read_json('docs/simplification_manifest.json'):
            path = baseline / change['destination']
            if not path.is_file() or digest(path) != change['original_sha256']:
                errors.append('Original organized notebook changed: ' + change['destination'])
                continue
            before = json.loads(path.read_text(encoding='utf-8'))
            after = json.loads((ROOT / change['destination']).read_text(encoding='utf-8'))
            before_other = {k:v for k,v in before.items() if k != 'cells'}
            after_other = {k:v for k,v in after.items() if k != 'cells'}
            before_code = [c for c in before['cells'] if c['cell_type'] != 'markdown']
            after_code = [c for c in after['cells'] if c['cell_type'] != 'markdown']
            if before_other != after_other or before_code != after_code:
                errors.append('Code/output/metadata changed: ' + change['destination'])
            for old_index, new_index in change['original_to_simplified_cell_index'].items():
                if new_index is None:
                    continue
                a = before['cells'][int(old_index)]
                b = after['cells'][new_index]
                if a['cell_type'] == 'markdown' and {k:v for k,v in a.items() if k != 'source'} != {k:v for k,v in b.items() if k != 'source'}:
                    errors.append('Markdown metadata changed: ' + change['destination'])
            code_unchanged += sum(c['cell_type']=='code' for c in before_code)
            reading_copies_checked += 1

    # Conservative pattern check; report paths only, never matching token text.
    credential_patterns = [
        re.compile(rb'\bhf_[A-Za-z0-9]{30,}\b'),
        re.compile(rb'\bgh[pousr]_[A-Za-z0-9]{30,}\b'),
        re.compile(rb'\bgithub_pat_[A-Za-z0-9_]{40,}\b'),
        re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    ]
    flagged = []
    files = [p for p in ROOT.rglob('*') if p.is_file()]
    for path in files:
        if path.suffix.lower() not in ('.ipynb', '.py', '.json', '.md', '.txt', '.csv', '.yaml', '.yml'):
            continue
        data = path.read_bytes()
        if any(pattern.search(data) for pattern in credential_patterns):
            flagged.append(path.relative_to(ROOT).as_posix())
    if flagged:
        errors.append('Potential credential markers: ' + ', '.join(flagged))
    largest = max(files, key=lambda p: p.stat().st_size)
    return {
        'ok': not errors,
        'notebooks': len(manifest),
        'source_paths': len(sources),
        'byte_identical_aliases': len(sources) - len(manifest),
        'groups': len({r['group'] for r in manifest}),
        'roles': dict(Counter(r['role'] for r in manifest)),
        'copied_support_files': len(support),
        'original_notebooks_checked': originals_checked,
        'original_support_files_checked': support_originals_checked,
        'original_archives_checked': archives_checked,
        'previous_package_notebooks_unchanged': previous_checked,
        'reading_copies_compared_with_original_package': reading_copies_checked,
        'code_cells_and_saved_outputs_unchanged': code_unchanged,
        'local_links_checked': links_checked,
        'notebooks_with_saved_errors': sum(bool(r['errors']) for r in manifest),
        'credential_pattern_flags': flagged,
        'max_file_bytes': largest.stat().st_size,
        'max_file_path': largest.relative_to(ROOT).as_posix(),
        'experiments_executed': False,
        'errors': errors,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-root', type=Path)
    parser.add_argument('--downloads', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = verify(args.original_root, args.downloads)
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.report:
        args.report.write_text(output + '\n', encoding='utf-8')
    print(output)
    raise SystemExit(0 if result['ok'] else 1)
