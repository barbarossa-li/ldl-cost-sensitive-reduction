"""Frozen repeat-1 n192 development comparison; no test/audit labels used."""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'stability'))
from granulation_reference import TrainingGeometry
from fast_granulate import granulate
from search import search

PROTOCOL = {
    'datasets': ['Natural Scene', 'SBU_3DFE', 'Yeast_cold'],
    'repeat': 1, 'training_size': 192, 'inner_folds': 3,
    'methods': ['P', 'G', 'A'], 'epsilon': 0.05, 'tolerance': 0.0,
    'predictor': '5NN, stable distance sorting, fold-training min-max scaling',
    'geometry': {'beta': 1, 'zeta': 0.1, 'smin': 2, 'kappa': 0.1, 'alpha': 1},
    'scope': 'Development diagnostic on previously examined splits; not confirmatory evidence',
    'selection_labels': 'Saved 192 training objects only; test/audit rows not indexed',
    'refinement': 'Shared restarting deletion only; no exchange refinement',
    'parameters': 'Existing B2 epsilon default fixed in advance, no sweep or tuning',
    'timing': 'Per-method fresh caches; geometry fitted lazily and included in method seconds',
}


def run():
    OUT.joinpath('pilot_protocol.json').write_text(json.dumps(PROTOCOL, indent=2) + '\n')
    rows = []
    for name in PROTOCOL['datasets']:
        folder = (ROOT / 'stability/repeat_1_n192' if name == 'Natural Scene'
                  else ROOT / 'cross_dataset' / name / 'repeat_1_n192')
        split_path = folder / 'split.json'
        split = json.loads(split_path.read_text())
        ids = split['train_indices']
        assert len(ids) == len(set(ids)) == 192
        assert set(ids).isdisjoint(split['test_indices'])
        if name == 'Natural Scene':
            x = np.load(ROOT / 'pilot/data/feature.npy')[ids]
            labels = np.load(ROOT / 'pilot/data/label.npy')[ids]
            costs = json.loads((ROOT / 'stability/protocol.json').read_text())['costs']
        else:
            data = loadmat(ROOT / 'cross_dataset/data' / f'{name}.mat')
            x = np.asarray(data['features'], dtype=float)[ids]
            labels = np.asarray(data['labels'], dtype=float)[ids].copy()
            labels /= labels.sum(1, keepdims=True)
            costs = split['costs']
        blocks = []
        n, m = x.shape
        folds = split['fold_validation_positions']
        assert sorted(i for v in folds for i in v) == list(range(n))
        for val in folds:
            val = np.array(val)
            fit = np.setdiff1d(np.arange(n), val)
            lo = x[fit].min(0)
            span = x[fit].max(0) - lo
            span[span == 0] = 1
            a, b = (x[fit] - lo) / span, (x[val] - lo) / span
            blocks.append((((b[:, None, :] - a[None, :, :]) ** 2).transpose(2, 0, 1).copy(), fit, val))
        for method in PROTOCOL['methods']:
            cache, dcache = {}, {}
            model = None
            calls = 0

            def risk(subset):
                nonlocal calls
                calls += 1
                key = tuple(sorted(subset))
                if key not in cache:
                    total = 0.0
                    for mat, fit, val in blocks:
                        if key:
                            neighbors = np.argsort(mat[list(key)].sum(0), axis=1, kind='stable')[:, :5]
                            pred = labels[fit][neighbors].mean(1)
                        else:
                            pred = np.tile(labels[fit].mean(0), (len(val), 1))
                        total += np.sum((labels[val] - pred) ** 2)
                    cache[key] = float(total / n)
                return cache[key]

            def dep(subset):
                nonlocal model
                key = tuple(sorted(subset))
                if key not in dcache:
                    if model is None:
                        model = TrainingGeometry(x, labels, kinds=['numerical'] * m)
                    distance, keys = model.subset(key)
                    dcache[key] = granulate(distance, labels, keys)[2]
                return dcache[key]

            started = time.perf_counter()
            result = search(costs, risk, dep if method != 'P' else None,
                            method, epsilon=PROTOCOL['epsilon'], tolerance=0)
            seconds = time.perf_counter() - started
            actual_unique, actual_calls = len(cache), calls
            # Compare to the existing pre-test reference using training loss only.
            prior = json.loads((ROOT / 'predictive_revision' / name.replace(' ', '_') /
                                'selection_before_test.json').read_text())
            assert abs(result['limit'] - prior['inner_reference']) < 1e-12
            selected = set(result['selected'])
            assert result['feasible'] and risk(selected) <= result['limit']
            assert all(risk(selected - {a}) > result['limit'] for a in selected)
            # Independently check A's per-step prediction-efficiency bound.
            for step in result['trace']:
                if 'step' not in step or method != 'A':
                    continue
                before = set(step['before'])
                gains = {a: risk(before) - risk(before | {a}) for a in range(m) if a not in before}
                positive = [a for a in gains if gains[a] > 0]
                if positive:
                    best = max(gains[a] / costs[a] for a in positive)
                    assert gains[step['attribute']] / costs[step['attribute']] >= (1 - PROTOCOL['epsilon']) * best
            row = {'dataset': name, 'method': method, 'attributes': len(selected),
                   'cost': result['cost'], 'inner_loss': result['inner_loss'],
                   'full_reference': result['limit'], 'seconds': seconds,
                   'unique_risk_evaluations': actual_unique,
                   'risk_callback_calls': actual_calls,
                   'dependency_evaluations': len(dcache),
                   'forward_steps': sum('step' in t for t in result['trace']),
                   'deletions': sum(t.get('phase') == 'delete' for t in result['trace'])}
            rows.append(row)
            checkpoint = {'protocol': PROTOCOL, 'split_source': str(split_path.relative_to(ROOT)),
                          'split_sha256': hashlib.sha256(split_path.read_bytes()).hexdigest(),
                          'training_arrays_sha256': hashlib.sha256(x.tobytes() + labels.tobytes()).hexdigest(),
                          'result': result, 'summary': row,
                          'verified': {'training_reference_matches': True,
                                       'single_deletion_irreducible': True,
                                       'A_local_efficiency_bound': True if method == 'A' else None,
                                       'test_or_audit_evaluated': False}}
            job = OUT / 'pilot_results' / name.replace(' ', '_')
            job.mkdir(parents=True, exist_ok=True)
            (job / f'{method}.json').write_text(json.dumps(checkpoint, indent=2) + '\n')
            OUT.joinpath('pilot_summary.json').write_text(json.dumps(rows, indent=2) + '\n')
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    run()
