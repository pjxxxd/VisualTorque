"""Explicit 64 -> 52 family amendment, before any repair confirmation capture.

Run outside the unchanged original experiment package. Original model/runtime
identities remain truthful; this separate amendment and executable are hashed in
the new confirmation protocol. Never overwrite a frozen 64-family protocol.
"""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import sys
import traceback

TARGET_FAMILIES = 52
SEED = 2026102601
DESTINATION = 'confirmation_capacity52_v1'
STRATA = ('equal_weight_distance', 'weight_distance_conflict',
          'multi_stack_sum', 'aligned_cues')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_unstarted_original(root):
    """Allow only metadata left by the reported generator failure."""
    old = Path(root) / 'repair/confirmation'
    allowed = {'protocol.json', 'cohort/protocol.json', 'cohort/exclusions.json'}
    unexpected = [p.relative_to(old).as_posix() for p in old.rglob('*')
                  if p.is_file() and p.relative_to(old).as_posix() not in allowed]
    if unexpected:
        raise RuntimeError('Original confirmation has additional files; do not amend a started '
                           'or completed test. Preserve it for review: ' + repr(unexpected[:10]))
    return {p.relative_to(old).as_posix(): sha(p) for p in sorted(old.rglob('*')) if p.is_file()}


def capacity_counts(n, excluded):
    """Use the original generator's candidate construction, without redefining its domain."""
    source = inspect.getsource(n['repair_families'])
    marker = '    used=set(excluded); result=[]'
    assert source.count(marker) == 1, 'Unknown generator source; inspect before recovery.'
    prefix = source.split(marker)[0]
    tail = ("    return {kind: len({geometry_signature(s) for s in candidates[kind]} - set(excluded))\n"
            "            for kind in ('equal_weight_distance','weight_distance_conflict',\n"
            "                         'multi_stack_sum','aligned_cues')}\n")
    scope = dict(n)
    exec(compile(prefix + tail, '<original-generator-capacity-only>', 'exec'), scope)
    return scope['repair_families'](TARGET_FAMILIES, SEED, excluded)


def amended_confirmation(repair):
    """Retain the original intervention/analysis implementation; change cohort bookkeeping only."""
    source = inspect.getsource(repair.confirmation)
    replacements = [
        ("implementation=n['IMPLEMENTATION_SHA256'],families=64,seed=2026102601,conditions=35,",
         "implementation=n['IMPLEMENTATION_SHA256'],families=52,seed=2026102601,conditions=35,"),
        ("out=root/'confirmation';out.mkdir(exist_ok=True)",
         "protocol['capacity_amendment']=n['REPAIR_CAPACITY_AMENDMENT']\n"
         "    out=root/'confirmation_capacity52_v1';out.mkdir(exist_ok=True)"),
        ("out/'cohort',64,2026102601,protocol)", "out/'cohort',52,2026102601,protocol)"),
        ("Repair confirmation: 64 independent families, 35 conditions",
         "Repair confirmation: amended 52 independent families, 35 conditions"),
        ("assert len(records)==64*8*35", "assert len(records)==52*8*35"),
        ("status='heldout_repair_confirmation_complete',families=64,",
         "status='heldout_repair_confirmation_complete',families=52,"),
        ("presentations=512,condition_records=len(records),",
         "presentations=416,condition_records=len(records),\n"
         "        original_planned_families=64,capacity_amendment=n['REPAIR_CAPACITY_AMENDMENT'],"),
    ]
    for old, new in replacements:
        assert source.count(old) == 1, 'Unexpected repair implementation: ' + old
        source = source.replace(old, new)
    ast.parse(source)
    scope = dict(vars(repair))
    exec(compile(source, '<explicit-capacity52-confirmation>', 'exec'), scope)
    return scope['confirmation'], source


def prepare(n, original_module, patched_source):
    from core import fit_examples
    fit_examples(n)
    old_files = require_unstarted_original(n['ROOT'])
    root = n['ROOT'] / 'repair'
    selection_path = root / 'selected_route.json'
    selection = n['read_json'](selection_path)
    vectors_path = root / 'frozen_directions.npz'
    support_path = n['ROOT'] / 'prediction/support.json'
    assert selection['vectors_sha256'] == n['file_hash'](vectors_path)
    assert selection['direction_support_sha256'] == n['file_hash'](support_path)
    out = root / DESTINATION
    out.mkdir(parents=True, exist_ok=True)
    amendment_path = out / 'capacity_amendment.json'
    script_hash = sha(__file__)
    fixed = dict(
        version='capacity52_v1', reason='original equal_weight_distance pool had 13 unused families, fewer than 16',
        original_planned_families=64, amended_families=TARGET_FAMILIES,
        families_per_stratum=13, strata=list(STRATA), seed=SEED,
        rule='fixed 13 per stratum; refuse if any stratum has fewer; no adaptive further reduction',
        stage='before confirmation outcomes; development selection remains frozen',
        task_ranges='unchanged original repair_families generator',
        original_implementation=n['IMPLEMENTATION_SHA256'],
        original_repair_sha256=sha(original_module.__file__),
        recovery_script_sha256=script_hash,
        amended_confirmation_sha256=hashlib.sha256(patched_source.encode()).hexdigest(),
        selected_route_sha256=sha(selection_path), frozen_directions_sha256=sha(vectors_path),
        direction_support_sha256=sha(support_path), preserved_old_confirmation_files=old_files,
        limitations='52 rather than 64 independent families; lower precision; disclose protocol amendment')
    excluded, history = n['fresh_geometry_index'](n['PROJECT_ROOT'], n['DATASET_DIR'], out / 'cohort')
    counts = capacity_counts(n, excluded)
    print('Unused mirror-canonical families per stratum:', counts, flush=True)
    # Once a cohort exists, its own signatures are excluded from the history scan;
    # never reselect on resume, even if unrelated history later changes capacity.
    support = out / 'cohort/support.json'
    if not support.is_file() and min(counts.values()) < 13:
        raise RuntimeError('52-family amendment is not feasible. No model has been loaded and '
                           'no task ranges or exclusions were changed. Counts: ' + repr(counts))
    if amendment_path.is_file():
        amendment = n['read_json'](amendment_path)
        assert all(amendment[k] == v for k, v in fixed.items()), 'Frozen amendment changed; do not overwrite it.'
    else:
        # Feasibility check with the unmodified sampler before freezing this amendment.
        families = n['repair_families'](TARGET_FAMILIES, SEED, excluded)
        assert len(families) == TARGET_FAMILIES
        assert not n['geometries_in'](families) & excluded
        amendment = dict(fixed, available_at_amendment=counts,
                         geometry_history_files=history, results_used_for_capacity=False)
        n['freeze'](amendment_path, amendment)
    # Store the actual executable next to the declaration, and bind it by its hash.
    copy_path = out / 'recovery_source.py'
    if copy_path.is_file():
        assert sha(copy_path) == script_hash
    else:
        copy_path.write_bytes(Path(__file__).read_bytes())
    compiled_path = out / 'amended_confirmation_source.txt'
    if compiled_path.is_file():
        assert compiled_path.read_text(encoding='utf-8') == patched_source
    else:
        compiled_path.write_text(patched_source, encoding='utf-8')
    n['REPAIR_CAPACITY_AMENDMENT'] = dict(path=str(amendment_path), sha256=sha(amendment_path),
        original_planned_families=64, amended_families=52, families_per_stratum=13,
        recovery_script_sha256=script_hash)
    print('Explicit amendment frozen: 64 -> 52 families (13 per stratum), 416 presentations, '
          '14560 condition records. Original development selection and old protocol retained.', flush=True)
    return amendment


def export_recovery(n):
    from core import export
    from run import summary
    # Only redirect 4.5 report locations. Other completed sections remain untouched.
    source = inspect.getsource(summary)
    assert source.count('repair/confirmation/') == 2
    source = source.replace('repair/confirmation/', 'repair/' + DESTINATION + '/')
    scope = dict(summary.__globals__)
    exec(compile(source, '<capacity52-summary>', 'exec'), scope)
    scope['summary'](n)
    report = n['ROOT'] / 'REPORT_ZH.md'
    with report.open('a', encoding='utf-8') as f:
        f.write('\n\n4.5 协议修订：原计划 64 个家族因未使用几何不足而未启动确认；'
                '恢复版固定为四类各 13 个、共 52 个家族，416 种展示、14560 条条件记录。'
                '实际完成状态见上表。旧失败记录保留，不代表恢复版再次失败。'
                '层号、剂量、方向、物理范围、历史排除和统计检验均保留；'
                '不得把此结果写成原定 64 家族的确认。\n')
    return export(n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--package', required=True)
    p.add_argument('--project', required=True)
    p.add_argument('--run-id', default='blackwell_cu128')
    p.add_argument('--stage', choices=['confirmation', 'export'], required=True)
    args = p.parse_args()
    sys.path.insert(0, str(Path(args.package).resolve()))
    from core import base_namespace
    n = base_namespace(args.project, args.run_id)
    try:
        if args.stage == 'export':
            export_recovery(n)
        else:
            import repair
            confirmation, source = amended_confirmation(repair)
            prepare(n, repair, source)
            confirmation(n)
    except Exception:
        n['save_json'](n['ROOT'] / ('failure_capacity52_' + args.stage + '.json'),
                       dict(stage=args.stage, status='failed_not_complete', traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
