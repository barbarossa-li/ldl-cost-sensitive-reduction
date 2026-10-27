"""Check shared guarantees on arbitrary non-monotone subset tables."""
import json
from pathlib import Path
import random
from search import search

rng = random.Random(20261027)
for case in range(100):
    n = 6
    keys = [tuple(i for i in range(n) if mask >> i & 1) for mask in range(1 << n)]
    risks = {k: rng.random() for k in keys}
    deps = {k: rng.random() for k in keys}
    costs = [rng.randint(1, 10) for _ in range(n)]
    for method in ['P', 'G', 'A']:
        result = search(costs, risks.__getitem__, deps.__getitem__, method)
        selected = set(result['selected'])
        assert result['inner_loss'] <= result['limit']
        assert all(risks[tuple(sorted(selected - {a}))] > result['limit'] for a in selected)
        assert result == search(costs, risks.__getitem__, deps.__getitem__, method)
        for step in result['trace']:
            if 'step' not in step or method != 'A':
                continue
            before = set(step['before'])
            gains = {a: risks[tuple(sorted(before))] - risks[tuple(sorted(before | {a}))]
                     for a in range(n) if a not in before}
            positive = [a for a in gains if gains[a] > 0]
            if positive:
                assert gains[step['attribute']] / costs[step['attribute']] >= .95 * max(gains[a] / costs[a] for a in positive)

report = {'seed': 20261027, 'tables': 100, 'attributes': 6, 'method_runs': 300,
          'feasibility': True, 'single_deletion_irreducibility': True,
          'determinism': True, 'A_local_efficiency_bound': True,
          'scope': 'Logic checks only; no empirical performance guarantee'}
Path(__file__).with_name('logic_validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
