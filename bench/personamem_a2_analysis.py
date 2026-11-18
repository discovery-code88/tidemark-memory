#!/usr/bin/env python3
"""Experiment A2 analysis (randomized order, identical-prompt controls); see ANALYSIS_PLAN.md in the run directory.
Holm family (confirmatory): the four comparisons in section 3; the first is primary.
usage: personamem_a2_analysis.py RUN_DIR"""
import csv, json, os, random, re, sys
from math import comb

D = sys.argv[1]
ARMS = ["plain", "plain_repeat", "kind", "novelty"]
runs = {a: ({r["question_id"]: r for r in map(json.loads, open(os.path.join(D, f"r.{a}.jsonl")))} if os.path.exists(os.path.join(D, f"r.{a}.jsonl")) else {}) for a in ARMS}
complete = all(len(runs[a]) >= 589 for a in ARMS)
print(f"answers per arm: { {a: len(runs[a]) for a in ARMS} }  -> {'complete' if complete else 'INCOMPLETE (a cap stopped the run or it is still running)'}")
for f in ("stopped",):
    if os.path.exists(os.path.join(D, f)):
        print("stopped by a cap:", open(os.path.join(D, f)).read().strip())
print()
rows = {r["question_id"]: r for r in csv.DictReader(open(os.path.expanduser("~/.cache/unlimited-memory-recall/practice/personamem_questions_32k.csv")))}
NOVEL = re.compile(
    r"(?i)\b(?:new|different|fresh|other|alternative)\s+(?:ideas?|ways?|things?|activit\w+|hobb\w+|options?|suggestions?|recommendations?|experiences?|approach\w*|places?)\b"
    r"|\bsomething (?:new|different|else)\b|\bhaven'?t tried\b|\bnever tried\b|\btry (?:out )?something\b|\bbranch out\b|\bswitch (?:it|things) up\b|\bchange things up\b")
flagged = {q for q, r in rows.items() if NOVEL.search(r["user_question_or_message"])}
NEWIDEAS = {q for q, r in rows.items() if r["question_type"] == "suggest_new_ideas"}
OTHER = set(rows) - NEWIDEAS
SEED = 20261102


def sign_p(w, l):
    n, k = w + l, min(w, l)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def clopper(k, n, a=0.05):
    """Exact 95% CI for a proportion (bisection on the binomial tails)."""
    def cdf(x, p): return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(x + 1))
    def solve(f):
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) else (lo, mid)
        return (lo + hi) / 2
    low = 0.0 if k == 0 else solve(lambda p: 1 - cdf(k - 1, p) < a / 2)
    high = 1.0 if k == n else solve(lambda p: cdf(k, p) > a / 2)
    return low, high


def compare(a, b, qs):
    both = sorted(q for q in qs if q in runs[a] and q in runs[b])
    xa, xb = [runs[a][q]["ok"] for q in both], [runs[b][q]["ok"] for q in both]
    w = sum(1 for p, q in zip(xa, xb) if q and not p)
    l = sum(1 for p, q in zip(xa, xb) if p and not q)
    n = len(both)
    rnd = random.Random(SEED)
    diffs = sorted(sum(xb[i] - xa[i] for i in idx) / n for idx in ([rnd.randrange(n) for _ in range(n)] for _ in range(10000))) if n else [0.0]
    return dict(n=n, missing=len(qs) - n, acc_a=sum(xa) / max(n, 1), acc_b=sum(xb) / max(n, 1), w=w, l=l,
                diff=(sum(xb) - sum(xa)) / max(n, 1), ci=(diffs[249], diffs[9749]), p=sign_p(w, l))


def fmt(name, a, b, c, holm=None):
    h = f"  Holm p={holm:.3f}" if holm is not None else ""
    return (f"{name:30s} n={c['n']:3d} (missing {c['missing']})  {a} {100*c['acc_a']:5.1f}%  {b} {100*c['acc_b']:5.1f}%  "
            f"{b} better {c['w']}, worse {c['l']}  diff {100*c['diff']:+5.1f} [95% CI {100*c['ci'][0]:+.1f}, {100*c['ci'][1]:+.1f}]  p={c['p']:.3f}{h}")


print("1. NOISE FLOOR (descriptive)")
for name, a, b, qs in (("plain_repeat vs plain (589)", "plain", "plain_repeat", set(rows)),
                       ("novelty vs kind, unflagged", "kind", "novelty", set(rows) - flagged)):
    c = compare(a, b, qs)
    d = c["w"] + c["l"]
    lo, hi = clopper(d, c["n"]) if c["n"] else (0, 0)
    print(f"  {fmt(name, a, b, c)}\n    flip rate {d}/{c['n']} = {100*d/max(c['n'],1):.1f}% [95% CI {100*lo:.1f}, {100*hi:.1f}]")

print("\n2. POSITION, ATTEMPTS, LATENCY (descriptive; associations, not causes)")
by_pos = {p: [] for p in (1, 2, 3, 4)}
for a in ARMS:
    for r in runs[a].values():
        by_pos[r["position"]].append(r)
for p, rs in by_pos.items():
    print(f"  position {p}: n={len(rs)} accuracy {100*sum(r['ok'] for r in rs)/max(len(rs),1):.1f}%  mean attempts {sum(r['attempts'] for r in rs)/max(len(rs),1):.2f}  "
          f"mean latency {sum(r['latency_s'] for r in rs)/max(len(rs),1):.1f} s")
first, last = {}, {}
for a in ARMS:
    for q, r in runs[a].items():
        if r["position"] == 1: first[q] = r["ok"]
        if r["position"] == 4: last[q] = r["ok"]
qs = [q for q in first if q in last]
w = sum(1 for q in qs if last[q] and not first[q]); l = sum(1 for q in qs if first[q] and not last[q])
print(f"  position 4 vs position 1 (paired, n={len(qs)}): last better {w}, worse {l}, p={sign_p(w, l):.3f}")

print("\n3. CONFIRMATORY (Holm family of four; order randomized)")
family = [("kind vs plain (other 496)", "plain", "kind", OTHER), ("novelty vs plain (other 496)", "plain", "novelty", OTHER),
          ("kind vs plain (93 new ideas)", "plain", "kind", NEWIDEAS), ("novelty vs plain (93 new ideas)", "plain", "novelty", NEWIDEAS)]
primary = "kind vs plain (other 496)"
res = [compare(a, b, q) for _, a, b, q in family]
order = sorted(range(len(res)), key=lambda i: res[i]["p"])
holm, run = [0.0] * len(res), 0.0
for rank, i in enumerate(order):
    run = max(run, min(1.0, (len(res) - rank) * res[i]["p"]))
    holm[i] = run
for (name, a, b, _), c, h in zip(family, res, holm):
    print(f"  {'PRIMARY  ' if name == primary else 'secondary'} {fmt(name, a, b, c, h)}")

print("\n4. EXPLORATORY (not confirmatory)")
for name, a, b in (("kind vs plain_repeat", "plain_repeat", "kind"), ("novelty vs plain_repeat", "plain_repeat", "novelty"), ("novelty vs kind", "kind", "novelty")):
    for label, qs in (("other 496", OTHER), ("93 new ideas", NEWIDEAS)):
        print("  " + fmt(f"{name} ({label})", a, b, compare(a, b, qs)))
for label, qs in (("93 new ideas", NEWIDEAS), ("other 496", OTHER)):
    for part, sub in (("flagged", qs & flagged), ("unflagged", qs - flagged)):
        for name, a, b in (("kind vs plain", "plain", "kind"), ("novelty vs plain", "plain", "novelty")):
            print("  " + fmt(f"{name} ({label}, {part} {len(sub)})", a, b, compare(a, b, sub)))
