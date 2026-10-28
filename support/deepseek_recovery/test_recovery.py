"""CPU regression checks for the reported geometry exhaustion; no model outcomes."""
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import contextmanager

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[2] / 'VisualTorque/deepseek_vl2'
sys.path.insert(0, str(PACKAGE))
import core
import repair
import recover_repair as recovery


@contextmanager
def temporary_project():
    temp = tempfile.TemporaryDirectory(dir=HERE, prefix='software_test_')
    assert Path(temp.name).resolve().parent == HERE.resolve()
    with temp as directory:
        yield directory


def namespace():
    n = {'__name__': 'synthetic_capacity_test'}
    core.load_kernels(n, ['followup_core.py', 'independent_core.py'])
    return n


def exhausted_equal_pool(n):
    signatures = set()
    for units in range(1, 9):
        for left in range(1, 7):
            for right in range(1, 7):
                if left != right:
                    signatures.add(n['geometry_signature']([
                        dict(side='left', units=units, distance=left),
                        dict(side='right', units=units, distance=right)]))
    assert len(signatures) == 120
    return set(sorted(signatures)[:-13])


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = namespace()
        cls.excluded = exhausted_equal_pool(cls.n)

    def test_reproduce_exact_failure_then_52_preserves_domain_and_balance(self):
        n = self.n
        with self.assertRaisesRegex(AssertionError, "equal_weight_distance.*13"):
            n['repair_families'](64, recovery.SEED, self.excluded)
        counts = recovery.capacity_counts(n, self.excluded)
        self.assertEqual(counts['equal_weight_distance'], 13)
        families = n['repair_families'](52, recovery.SEED, self.excluded)
        self.assertEqual(n['Counter'](f['stratum'] for f in families),
                         {k: 13 for k in recovery.STRATA})
        self.assertEqual(len(n['geometries_in'](families)), 52)
        self.assertFalse(n['geometries_in'](families) & self.excluded)
        self.assertEqual(sum(len(f['scenes']) for f in families), 208)
        self.assertEqual(len(families) * 8 * 35, 14560)
        self.assertEqual(families, n['repair_families'](52, recovery.SEED, self.excluded))
        for f in families:
            self.assertEqual({(s['mirror'], s['style']) for s in f['scenes']},
                             {(m, s) for m in (False, True) for s in ('standard', 'alternate')})
            for scene in f['scenes']:
                for stack in scene['stacks']:
                    self.assertIn(stack['distance'], range(1, 7))
                    self.assertIn(stack['units'], range(1, 6 if len(scene['stacks']) == 4 else 9))

    def test_refuse_started_original_and_keep_metadata(self):
        with temporary_project() as d:
            root = Path(d)
            old = root / 'repair/confirmation'
            self.n['save_json'](old / 'protocol.json', {'families': 64})
            before = (old / 'protocol.json').read_bytes()
            self.assertEqual(set(recovery.require_unstarted_original(root)), {'protocol.json'})
            self.n['save_json'](old / 'trial_shards/first.json', {'result': 'synthetic'})
            with self.assertRaisesRegex(RuntimeError, 'started or completed'):
                recovery.require_unstarted_original(root)
            self.assertEqual((old / 'protocol.json').read_bytes(), before)

    def fixture(self, directory):
        n = dict(self.n)
        project = Path(directory)
        root = project / 'results/deepseek_vl2_small/blackwell_cu128'
        dataset = project / 'data/dataset_v1'
        dataset.mkdir(parents=True)
        rows = [dict(stacks=[dict(side=s, distance=d, units=u) for s, d, u in sig])
                for sig in sorted(self.excluded)]
        (dataset / 'scenes.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
        n.update(ROOT=root, PROJECT_ROOT=project, DATASET_DIR=dataset, IMPLEMENTATION_SHA256='original-fixture')
        n['save_json'](root / 'prediction/support.json', {'families': []})
        (root / 'repair').mkdir()
        (root / 'repair/frozen_directions.npz').write_bytes(b'synthetic-hash-only-fixture')
        n['save_json'](root / 'repair/selected_route.json', dict(
            vectors_sha256=recovery.sha(root / 'repair/frozen_directions.npz'),
            direction_support_sha256=recovery.sha(root / 'prediction/support.json'),
            selected={'eligible': False}, upstream_layer=13, dose=1.0))
        n['save_json'](root / 'repair/confirmation/protocol.json', {'families': 64})
        return n

    def test_freeze_resume_no_reselection_and_refuse_vector_tampering(self):
        with temporary_project() as d, patch.object(core, 'fit_examples', lambda n: None):
            n = self.fixture(d)
            old = n['ROOT'] / 'repair/confirmation/protocol.json'
            before = old.read_bytes()
            _, source = recovery.amended_confirmation(repair)
            first = recovery.prepare(n, repair, source)
            out = n['ROOT'] / 'repair' / recovery.DESTINATION
            amendment_bytes = (out / 'capacity_amendment.json').read_bytes()
            families, support = n['independent_support'](
                n['PROJECT_ROOT'], n['DATASET_DIR'], out / 'cohort', 52, recovery.SEED,
                {'capacity_amendment': n['REPAIR_CAPACITY_AMENDMENT']})
            cohort_bytes = support.read_bytes()
            second = recovery.prepare(n, repair, source)
            self.assertEqual(first, second)
            self.assertEqual(amendment_bytes, (out / 'capacity_amendment.json').read_bytes())
            resumed, _ = n['independent_support'](
                n['PROJECT_ROOT'], n['DATASET_DIR'], out / 'cohort', 52, recovery.SEED,
                {'capacity_amendment': n['REPAIR_CAPACITY_AMENDMENT']})
            self.assertEqual(families, resumed)
            self.assertEqual(cohort_bytes, support.read_bytes())
            self.assertEqual(before, old.read_bytes())
            self.assertFalse(first['results_used_for_capacity'])
            (n['ROOT'] / 'repair/frozen_directions.npz').write_bytes(b'changed')
            with self.assertRaises(AssertionError):
                recovery.prepare(n, repair, source)

    def test_insufficient_52_pool_does_not_shrink_further_or_freeze(self):
        with temporary_project() as d, patch.object(core, 'fit_examples', lambda n: None):
            n = self.fixture(d)
            available = n['repair_families'](52, recovery.SEED, self.excluded)
            equal = next(f for f in available if f['stratum'] == 'equal_weight_distance')
            n['save_json'](n['PROJECT_ROOT'] / 'results/other/support.json', {'families': [equal]})
            _, source = recovery.amended_confirmation(repair)
            with self.assertRaisesRegex(RuntimeError, 'not feasible'):
                recovery.prepare(n, repair, source)
            self.assertFalse((n['ROOT'] / 'repair' / recovery.DESTINATION / 'capacity_amendment.json').exists())

    def test_only_bookkeeping_changes_and_correct_completion_constants(self):
        function, source = recovery.amended_confirmation(repair)
        self.assertTrue(callable(function))
        original = inspect.getsource(repair.confirmation)
        # The entire intervention loop, gates, estimates and multiple-testing logic are identical.
        start = "        fid=family['family_id']"
        end = '    assert len(records)'
        self.assertEqual(source.split(start)[1].split(end)[0], original.split(start)[1].split(end)[0])
        start = "    frame=n['pd'].DataFrame"
        end = "    n['save_json'](out/'completion.json'"
        self.assertEqual(source.split(start)[1].split(end)[0], original.split(start)[1].split(end)[0])
        self.assertIn('assert len(records)==52*8*35', source)
        self.assertIn('presentations=416', source)
        self.assertIn("status='heldout_repair_confirmation_complete',families=52", source)
        self.assertIn("protocol['capacity_amendment']", source)

    def test_export_partial_and_completed_recovery_with_accurate_status(self):
        with temporary_project() as d:
            n = dict(self.n, ROOT=Path(d) / 'run')
            n['ROOT'].mkdir()
            recovery.export_recovery(n)
            rows = n['read_json'](n['ROOT'] / 'summary.json')
            self.assertFalse(rows[-1]['complete'])
            n['save_json'](n['ROOT'] / 'repair' / recovery.DESTINATION / 'completion.json',
                           dict(families=52, presentations=416, condition_records=14560))
            archive = recovery.export_recovery(n)
            rows = n['read_json'](n['ROOT'] / 'summary.json')
            self.assertTrue(rows[-1]['complete'])
            self.assertEqual(rows[-1]['details']['families'], 52)
            import zipfile
            with zipfile.ZipFile(archive) as z:
                self.assertIsNone(z.testzip())
                self.assertIn('repair/' + recovery.DESTINATION + '/completion.json', z.namelist())
            self.assertIn('52', (n['ROOT'] / 'REPORT_ZH.md').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
