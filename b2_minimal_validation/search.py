"""B2 minimal pilot: deterministic P/G/A search; no data access or audit labels.

Call search(costs, risk, dependency, method) with training-inner-fold risk.
All three methods use identical feasibility, bridges and deletion.
"""
import math

def search(costs, risk, dependency=None, method="P", epsilon=0.05, tolerance=0.0):
    if method not in ("P", "G", "A"):
        raise ValueError("method must be P, G or A")
    if method != "P" and dependency is None:
        raise ValueError("G/A require training-only dependency")
    if not 0 <= epsilon < 1:
        raise ValueError("epsilon must be in [0,1)")
    costs = tuple(float(v) for v in costs)
    if not costs or any(not math.isfinite(v) or v <= 0 for v in costs):
        raise ValueError("costs must be finite and positive")
    m = len(costs)
    full = tuple(range(m))
    limit = float(risk(full)) + tolerance
    if not math.isfinite(limit):
        raise ValueError("nonfinite full reference")
    P = set()
    trace = []
    risk_cache = {}
    dep_cache = {}
    def R(subset):
        key = tuple(sorted(subset))
        if key not in risk_cache:
            risk_cache[key] = float(risk(key))
            if not math.isfinite(risk_cache[key]):
                raise ValueError("nonfinite inner risk")
        return risk_cache[key]
    def D(subset):
        key = tuple(sorted(subset))
        if key not in dep_cache:
            dep_cache[key] = float(dependency(key))
            if not math.isfinite(dep_cache[key]):
                raise ValueError("nonfinite dependency")
        return dep_cache[key]
    if R(P) <= limit:
        return {"selected": [], "cost": 0.0, "inner_loss": R(P),
                "limit": limit, "trace": [], "risk_evaluations": len(risk_cache),
                "dependency_evaluations": 0, "feasible": True}
    for step in range(m):
        if R(P) <= limit:
            break
        candidates = [a for a in range(m) if a not in P]
        base = R(P)
        gains = {a: base - R(P | {a}) for a in candidates}
        positive = [a for a in candidates if gains[a] > 0]
        dg = {}
        if not positive:
            a = min(candidates, key=lambda a: (-gains[a], costs[a], a))
            rule = "shared_signed_bridge"
        else:
            best = min(positive, key=lambda a: (-gains[a] / costs[a], costs[a], a))
            if method == "P":
                a, rule = best, "prediction"
            elif method == "G":
                d0 = D(P)
                dg = {a: D(P | {a}) - d0 for a in positive}
                both = [a for a in positive if dg[a] > 0]
                a = (min(both, key=lambda a: (-dg[a] / costs[a], costs[a], a))
                     if both else best)
                rule = "granular_positive" if both else "prediction_fallback"
            else:
                qmax = gains[best] / costs[best]
                near = [a for a in positive if gains[a] / costs[a] >=
                        (1.0 - epsilon) * qmax]
                if len(near) == 1:
                    a, rule = best, "single_near_best"
                else:
                    d0 = D(P)
                    dg = {a: D(P | {a}) - d0 for a in near}
                    a = min(near, key=lambda a: (-dg[a] / costs[a], costs[a], a))
                    rule = "granular_tiebreak"
                    assert gains[a] / costs[a] >= (1 - epsilon) * qmax
        trace.append({"step": step, "before": sorted(P), "attribute": a,
                      "prediction_gain": gains[a], "dependency_gain": dg.get(a),
                      "rule": rule})
        P.add(a)
    if R(P) > limit:
        raise AssertionError("full-set reference must be reachable")
    while True:
        removable = [a for a in sorted(P, key=lambda a: (-costs[a], a))
                     if R(P - {a}) <= limit]
        if not removable:
            break
        a = removable[0]
        trace.append({"phase": "delete", "attribute": a, "before": sorted(P)})
        P.remove(a)
    return {"selected": sorted(P), "cost": sum(costs[a] for a in P),
            "inner_loss": R(P), "limit": limit, "trace": trace,
            "risk_evaluations": len(risk_cache),
            "dependency_evaluations": len(dep_cache), "feasible": R(P) <= limit}

if __name__ == "__main__":
    # Deterministic synthetic invariants, not an LDL benchmark.
    costs = [1, 2, 3]
    def risk(P):
        return 10.0 - sum((3, 4, 5)[a] for a in P)
    def dep(P):
        return sum((1, 5, 9)[a] for a in P)
    for method in ("P", "G", "A"):
        result = search(costs, risk, dep, method)
        assert result["feasible"]
        assert result["inner_loss"] <= result["limit"]
    assert search(costs, risk, dep, "A", epsilon=0)["selected"] == search(costs, risk, dep, "P")["selected"]
    print("Synthetic invariants passed")
