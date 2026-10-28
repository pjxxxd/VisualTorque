"""I/O-only acceleration. Same candidate files, hashes and geometry extraction.

Kept separate from the scientific implementation hash so an already frozen study
can resume. A separate I/O provenance file records this optimization.
"""
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

HISTORY_IO_VERSION = 'scandir-single-read-v1'
HISTORY_IO_WORKERS = 4
_history_progress_time = 0.


def _history_progress(message, force=False):
    global _history_progress_time
    now=time.monotonic()
    if force or now-_history_progress_time >= 4.:
        print(message,flush=True)
        if callable(globals().get('cell_stage')):
            cell_stage(message)
        _history_progress_time=now


def _history_candidate(name):
    lower=name.lower()
    return (lower.endswith(('.json','.jsonl')) and
            any(word in lower for word in ('manifest','support','famil','scene','geometr')))


def _history_scan_directory(folder, excluded_root):
    directories, candidates = [], []
    count=0
    # DirEntry caches file type; noncandidate files never call is_file/resolve.
    with os.scandir(folder) as entries:
        for entry in entries:
            count+=1
            if entry.is_dir(follow_symlinks=False):
                if os.path.normcase(os.path.abspath(entry.path)) != excluded_root:
                    directories.append(entry.path)
            elif _history_candidate(entry.name) and entry.is_file():
                candidates.append(entry.path)
    return directories,candidates,count


def _history_discover(roots, output_root):
    excluded=os.path.normcase(os.path.abspath(output_root))
    queue=deque(str(root) for root in roots
                if root.exists() and os.path.normcase(os.path.abspath(root)) != excluded)
    seen, candidates=set(),set()
    n_dirs=n_entries=0
    _history_progress('Scanning history directories; no GPU inference is running.',True)
    with ThreadPoolExecutor(max_workers=HISTORY_IO_WORKERS) as pool:
        pending={}
        while queue or pending:
            while queue and len(pending)<HISTORY_IO_WORKERS:
                folder=queue.popleft()
                key=os.path.normcase(os.path.abspath(folder))
                if key in seen or key==excluded or key.startswith(excluded+os.sep):
                    continue
                seen.add(key)
                pending[pool.submit(_history_scan_directory,folder,excluded)]=folder
            if not pending:
                continue
            ready,_=wait(pending,timeout=1.,return_when=FIRST_COMPLETED)
            for future in ready:
                folder=pending.pop(future)
                try:
                    directories,files,count=future.result()
                except OSError as exc:
                    raise RuntimeError('Cannot list historical directory: '+folder) from exc
                queue.extend(directories); candidates.update(files)
                n_dirs+=1; n_entries+=count
            active=next(iter(pending.values()),queue[0] if queue else '-')
            _history_progress(f'History scan: {n_dirs} directories; {n_entries} entries; '
                              f'{len(candidates)} candidate JSON files. Current: {active}')
    # Resolve only candidate paths; preserve the old canonical-path scope.
    result=set()
    for name in sorted(candidates):
        path=Path(name).resolve()
        if output_root not in path.parents:
            result.add(str(path))
    _history_progress(f'Directory scan complete: {n_dirs} directories; {len(result)} JSON files to read.',True)
    return sorted(result)


def _history_read_once(name):
    """Parse and hash the SAME bytes, with one Drive read and streaming JSONL."""
    path=Path(name); hasher=hashlib.sha256(); signatures=set()
    if path.suffix == '.jsonl':
        with path.open('rb') as stream:
            for line in stream:
                hasher.update(line)
                if line.strip():
                    signatures.update(geometries_in(json.loads(line.decode('utf-8'))))
    else:
        with path.open('rb') as stream:
            payload=stream.read()
        hasher.update(payload)
        signatures.update(geometries_in(json.loads(payload.decode('utf-8'))))
    return name,hasher.hexdigest(),signatures


def _history_parallel_items(paths, work, description):
    iterator=iter(paths); total=len(paths); done=0
    with ThreadPoolExecutor(max_workers=HISTORY_IO_WORKERS) as pool:
        pending={}
        exhausted=False
        while pending or not exhausted:
            while len(pending)<HISTORY_IO_WORKERS and not exhausted:
                try:
                    name=next(iterator)
                except StopIteration:
                    exhausted=True; break
                pending[pool.submit(work,name)]=name
            if not pending:
                continue
            ready,_=wait(pending,timeout=1.,return_when=FIRST_COMPLETED)
            for future in ready:
                name=pending.pop(future)
                try:
                    result=future.result()
                except Exception as exc:
                    raise RuntimeError('Cannot read historical input: '+name) from exc
                done+=1
                yield result
            active=next(iter(pending.values()),'-')
            _history_progress(f'{description}: {done}/{total} files; current: {active}')
    _history_progress(f'{description} complete: {done}/{total} files.',True)


def fast_history_inventory(project, output_root, extra_roots=()):
    output_root=Path(output_root).resolve()
    lock=output_root/'historical_geometry_snapshot.json'
    if lock.exists():
        obj=read_json(lock)
        _history_progress(f"Verifying the frozen history snapshot ({len(obj['files'])} files); no directory rescan.",True)
        for name,actual in _history_parallel_items(
                sorted(obj['files']),lambda name:(name,file_hash(name)),'Verify frozen history'):
            assert actual==obj['files'][name], 'A frozen historical input changed: '+name
        return {tuple(tuple(t) for t in sig) for sig in obj['signatures']},obj
    roots=[(Path(project)/'data').resolve(),(Path(project)/'results').resolve(),
           *(Path(p).resolve() for p in extra_roots)]
    # A complete file list is saved before reading it. An interrupted read can
    # reuse this original inventory, exactly as a completed snapshot would.
    listing=output_root/'history_file_list_v1.json'
    root_identity=[str(p) for p in roots]
    if listing.exists():
        obj=read_json(listing)
        assert obj['roots']==root_identity, 'Historical roots changed after the file list was frozen.'
        paths=obj['files']
        _history_progress(f'Reusing the frozen file list: {len(paths)} files; directory rescan skipped.',True)
    else:
        paths=_history_discover(roots,output_root)
        save_json(listing,dict(roots=root_identity,files=paths,created_at_utc=utc_now(),
                              scope='candidate set fixed after completed directory enumeration'))
    signatures,hashes=set(),{}
    for name,sha,geometries in _history_parallel_items(paths,_history_read_once,'Read and hash history'):
        signatures.update(geometries); hashes[name]=sha
    assert signatures, 'No historical scenes found. Restore the project data/results before generating new support.'
    obj=dict(files=hashes,signatures=sorted(signatures),created_at_utc=utc_now(),
             scope='available geometry-bearing JSON/JSONL files only; unimported archives are not indexed')
    save_json(lock,obj)
    _history_progress(f'History ready: {len(signatures)} distinct mirror-canonical geometries excluded.',True)
    return signatures,obj


def enable_history_io_acceleration():
    required=('inventory_geometries','geometries_in','save_json','file_hash')
    assert all(name in globals() for name in required), (
        'Run the original utility/configuration cells first, then run this patch in the same notebook namespace.')
    globals()['inventory_geometries']=fast_history_inventory
    # Distinguish the three tasks formerly sharing one unchanged stage label.
    original=globals().get('generate_support')
    if original and not getattr(original,'_history_stage_wrapped',False):
        def generate_with_progress(*args,**kwargs):
            _history_progress('Generating new physical families on CPU; history scanning is complete.',True)
            result=original(*args,**kwargs)
            _history_progress(f'Generated {len(result)} families. Preparing images next.',True)
            return result
        generate_with_progress._history_stage_wrapped=True
        globals()['generate_support']=generate_with_progress
    render=globals().get('render_support')
    if render and not getattr(render,'_history_stage_wrapped',False):
        def render_with_progress(*args,**kwargs):
            _history_progress('Rendering / reusing support images; history scanning is complete.',True)
            result=render(*args,**kwargs)
            _history_progress('Checking image hashes and freezing image identity.',True)
            return result
        render_with_progress._history_stage_wrapped=True
        globals()['render_support']=render_with_progress
    if 'RUN_ROOT' in globals():
        receipt=dict(version=HISTORY_IO_VERSION,
            io_implementation_sha256=globals().get('IO_IMPLEMENTATION_SHA256'),
            scientific_implementation_sha256=globals().get('IMPLEMENTATION_SHA256'),
            workers=HISTORY_IO_WORKERS,enabled_at_utc=utc_now(),
            policy='same history filename filter and full content hashes; no experimental parameters changed')
        save_json(Path(RUN_ROOT)/'io_acceleration_v1.json',receipt)
        if 'PHASE_ROOT' in globals():
            # The existing exporter already includes all phase files.
            save_json(Path(PHASE_ROOT)/'io_acceleration_v1.json',receipt)
    print('Fast history I/O enabled. Keep the same RUN_ID and rerun the interrupted support cell.',flush=True)


enable_history_io_acceleration()
