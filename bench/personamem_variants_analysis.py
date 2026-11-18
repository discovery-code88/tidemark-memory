#!/usr/bin/env python3
"""Experiment A analysis (see ANALYSIS_PLAN.md in the run directory): label variants answered
from the same memories, compared pairwise with exact two-sided sign tests.

usage: personamem_variants_analysis.py RUN_DIR   (expects r.plain.jsonl, r.kind.jsonl, r.novelty.jsonl)
"""
import csv, json, os, random, re, sys
from math import comb

D = sys.argv[1]
V = ["plain", "kind", "novelty"]
runs = {v: {r["question_id"]: r for r in map(json.loads, open(os.path.join(D, f"r.{v}.jsonl")))} for v in V}
rows = {r["question_id"]: r for r in csv.DictReader(open(os.path.expanduser("~/.cache/unlimited-memory-recall/practice/personamem_questions_32k.csv")))}
NOVEL = re.compile(  # the server's asks_for_novelty pattern
    r"(?i)\b(?:new|different|fresh|other|alternative)\s+(?:ideas?|ways?|things?|activit\w+|hobb\w+|options?|suggestions?|recommendations?|experiences?|approach\w*|places?)\b"
    r"|\bsomething (?:new|different|else)\b|\bhaven'?t tried\b|\bnever tried\b|\btry (?:out )?something\b|\bbranch out\b|\bswitch (?:it|things) up\b|\bchange things up\b")
flagged = {q for q, r in rows.items() if NOVEL.search(r["user_question_or_message"])}
primary_set = {q for q, r in rows.items() if r["question_type"] == "suggest_new_ideas"}
other_set = set(rows) - primary_set


def sign_p(w, l):
    n, k = w + l, min(w, l)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def compare(a, b, qs):
    """b vs a on questions both answered: accuracy, wins/losses, diff with paired-bootstrap 95% CI, sign-test p."""
    both = sorted(q for q in qs if q in runs[a] and q in runs[b])
    xa = [runs[a][q]["ok"] for q in both]
    xb = [runs[b][q]["ok"] for q in both]
    wins = sum(1 for p, q in zip(xa, xb) if q and not p)
    losses = sum(1 for p, q in zip(xa, xb) if p and not q)
    n = len(both)
    rnd = random.Random(20261101)
    diffs = sorted(sum(xb[i] - xa[i] for i in idx) / n for idx in ([rnd.randrange(n) for _ in range(n)] for _ in range(10000))) if n else [0]
    return {"n": n, "missing": len(qs) - n, "acc_a": sum(xa) / max(n, 1), "acc_b": sum(xb) / max(n, 1), "wins": wins, "losses": losses,
            "diff": (sum(xb) - sum(xa)) / max(n, 1), "ci": (diffs[249], diffs[9749]), "p": sign_p(wins, losses)}


def line(name, a, b, c, p_holm=None):
    holm = f", Holm p={p_holm:.3f}" if p_holm is not None else ""
    return (f"{name:34s} n={c['n']:3d} (missing {c['missing']})  {a} {100 * c['acc_a']:5.1f}%  {b} {100 * c['acc_b']:5.1f}%  "
            f"{b} better on {c['wins']}, worse on {c['losses']}  diff {100 * c['diff']:+5.1f} pts [95% CI {100 * c['ci'][0]:+.1f}, {100 * c['ci'][1]:+.1f}]  p={c['p']:.3f}{holm}")


print("PRIMARY (93 suggest_new_ideas): kind vs plain")
fam = [("kind vs plain", "plain", "kind"), ("novelty vs plain", "plain", "novelty"), ("novelty vs kind", "kind", "novelty")]
res = [compare(a, b, primary_set) for _, a, b in fam]
# Holm across the three comparisons
order = sorted(range(3), key=lambda i: res[i]["p"])
holm, running = [0.0] * 3, 0.0
for rank, i in enumerate(order):
    running = max(running, min(1.0, (3 - rank) * res[i]["p"]))
    holm[i] = running
print("  " + line(fam[0][0], fam[0][1], fam[0][2], res[0], holm[0]))
print("SECONDARY (same 93)")
for i in (1, 2):
    print("  " + line(fam[i][0], fam[i][1], fam[i][2], res[i], holm[i]))
all_sig = all(r["p"] < 0.05 for r in res)
print(f"  all three raw p < 0.05: {all_sig} -> {'Holm-adjusted p govern' if all_sig else 'raw p reported; Holm shown for reference'}")

print("\nREGRESSION CHECKS")
for label, qs in (("all 589", set(rows)), ("other 496", other_set)):
    for name, a, b in fam:
        print("  " + line(f"{name} ({label})", a, b, compare(a, b, qs)))

print("\nEXPLORATORY: novelty-detector flagged vs unflagged")
for label, qs in (("93 primary", primary_set), ("496 other", other_set)):
    for part, sub in (("flagged", qs & flagged), ("unflagged", qs - flagged)):
        print(f"  {label}, {part}: {len(sub)} questions")
        for name, a, b in fam:
            print("    " + line(name, a, b, compare(a, b, sub)))
