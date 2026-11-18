#!/usr/bin/env python3
"""Accuracy per question type for one or more personamem_eval.py result files.
usage: personamem_report.py A.jsonl [B.jsonl ...]"""
import collections, json, sys

runs = {f: [json.loads(l) for l in open(f)] for f in sys.argv[1:]}
common = set.intersection(*({r["question_id"] for r in rs} for rs in runs.values()))
types = sorted({r["type"] for rs in runs.values() for r in rs})
print(f"{len(common)} questions answered in every run\n")
print(f"{'type':45s}" + "".join(f"{f.rsplit('/', 1)[-1][:18]:>20s}" for f in runs))
for t in types + ["ALL"]:
    cells = []
    for rs in runs.values():
        sel = [r for r in rs if r["question_id"] in common and (t == "ALL" or r["type"] == t)]
        cells.append(f"{sum(r['ok'] for r in sel)}/{len(sel)} {100 * sum(r['ok'] for r in sel) / max(1, len(sel)):5.1f}%")
    print(f"{t:45s}" + "".join(f"{c:>20s}" for c in cells))
