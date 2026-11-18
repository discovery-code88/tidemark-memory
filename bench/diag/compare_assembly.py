#!/usr/bin/env python3
"""Baseline vs evidence assembly on the development suite's memories (retrieval only).
usage: compare_assembly.py LABEL   (server on 127.0.0.1:18391; writes assembly/retrieval_comparison_LABEL.json)"""
import json, os, sys, urllib.request
R = os.path.expanduser("~/.cache/tidemark-dev/diag/run"); E = os.path.expanduser("~/.cache/tidemark-dev/diag/assembly")
sys.path.insert(0, R); from cases import CASES
sets = json.load(open(f"{R}/sets.json")); paused = json.load(open(f"{R}/paused.json"))
H = {"Authorization": "Token k", "Content-Type": "application/json"}
CHANGE = {"G1-1": ["I've re-engaged", "I decided to step back from organizing"], "G1-2": ["I decided to focus on safer"], "G1-3": ["I've been vegetarian"],
          "G2-1": ["I started attending trivia nights with friends"], "G2-2": ["I went to a comedy show with friends"], "G2-3": ["I love cooking with friends, except"],
          "G4-1": ["I actually dropped the pottery", "I also stopped my photography"], "G4-2": ["I've actually stopped reading graphic"], "G4-3": ["I had to stop attending yoga"],
          "G6-2": ["From now on"]}
# Relationship groups (evaluation only, from the gold evidence lists): statements that belong together.
GROUPS = {"G1-1": ["I've re-engaged", "I decided to step back from organizing"], "G1-3": ["I've been vegetarian", "I love a good steak"],
          "G2-1": ["I started attending trivia nights with friends", "I also participated in trivia nights but found them distracting"],
          "G2-2": ["leaving me questioning whether comedy shows", "I went to a comedy show with friends"],
          "G4-1": ["I actually dropped the pottery", "I also enrolled in a pottery class"],
          "G4-2": ["I've actually stopped reading graphic", "I've started enjoying graphic novels"],
          "G4-3": ["I had to stop attending yoga", "this connection truly helped with stress relief"]}
def search(c, assembly):
    body = {"query": c["request"], "user_id": sets[c["id"]]["user"], "top_k": 10, "assembly": assembly}
    return json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:18391/search", data=json.dumps(body).encode(), headers=H)).read())["data"]
out = {}; tot = {k: [0, 0] for k in ("be", "ce", "bc", "cc")}; disp_ev = added_n = 0
cases_complete = {"base": 0, "cand": 0, "n": 0}; grp = {"base": {"complete": 0, "split": 0, "absent": 0}, "cand": {"complete": 0, "split": 0, "absent": 0}}
added_log = []
print(f"{'case':5s} {'essential b→c':14s} {'change-type b→c':16s} added displaced(evidence) chars b→c")
for c in CASES:
    if c["id"] in paused: continue
    ev = sets[c["id"]]["evidence"]; base = search(c, False); cand = search(c, True)
    assert [h["id"] for h in base] == [r["id"] for r in sets[c["id"]]["retrieved"]], f"baseline changed for {c['id']}"
    bid = {h["id"] for h in base}; cid = {h["id"] for h in cand}
    def cov(ids, role=None, keys=None):
        items = [e for e in ev if (role is None or e["role"] == role) and (keys is None or any(e["key"].startswith(k) for k in keys))]
        return [sum(1 for e in items if any(i in ids for i in e["records"])), len(items)]
    added = [h for h in cand if h.get("kind") == "related"]; displaced = [h for h in base if h["id"] not in cid]
    ev_ids = {i for e in ev for i in e["records"]}; dev = [h["id"] for h in displaced if h["id"] in ev_ids]
    applicable = not (c.get("request_only") or c.get("no_essential")); keys = CHANGE.get(c["id"])
    be, ce = cov(bid, "essential"), cov(cid, "essential"); bc, cc = (cov(bid, None, keys), cov(cid, None, keys)) if keys else ([0, 0], [0, 0])
    if applicable:
        for k, v in (("be", be), ("ce", ce)): tot[k][0] += v[0]; tot[k][1] += v[1]
    for k, v in (("bc", bc), ("cc", cc)): tot[k][0] += v[0]; tot[k][1] += v[1]
    disp_ev += len(dev); added_n += len(added)
    if applicable:
        cases_complete["n"] += 1; cases_complete["base"] += be[0] == be[1]; cases_complete["cand"] += ce[0] == ce[1]
    if c["id"] in GROUPS:
        for side, ids in (("base", bid), ("cand", cid)):
            present = sum(1 for k in GROUPS[c["id"]] if any(i in ids for e in ev if e["key"].startswith(k) for i in e["records"]))
            grp[side]["complete" if present == len(GROUPS[c["id"]]) else "split" if present else "absent"] += 1
    rec_by_id = {h["id"]: h for h in cand}
    for h in added:
        terms = h.get("related_terms", [])
        def sentence_with(content):
            import re as _re
            for line in content.splitlines():
                if not line.lower().startswith("user:"): continue
                for s_ in _re.split(r"(?<=[.!?])\s+", line[5:]):
                    if any(t[:5] in s_.lower() for t in terms): return s_.strip()[:200]
            return ""
        added_log.append({"case": c["id"], "terms": terms, "added_sentence": sentence_with(h["content"]),
                          "anchor_sentence": sentence_with(rec_by_id.get(h.get("related_to"), {}).get("content", ""))})
    chars = (sum(len(h["content"]) for h in base), sum(len(h["content"]) for h in cand))
    print(f"{c['id']:5s} {(f'{be[0]}/{be[1]}→{ce[0]}/{ce[1]}' if applicable else 'n/a'):14s} {(f'{bc[0]}/{bc[1]}→{cc[0]}/{cc[1]}' if keys else '–'):16s} {len(added)}     {len(displaced)} ({len(dev)})          {chars[0]}→{chars[1]}")
    out[c["id"]] = {"baseline": base, "candidate": cand, "added": [{"id": h["id"], "related_to": h.get("related_to"), "content": h["content"][:300]} for h in added],
                    "displaced": [h["id"] for h in displaced], "displaced_evidence": dev, "chars": chars}
json.dump(out, open(f"{E}/retrieval_comparison_{sys.argv[1]}.json", "w"), indent=1)
print(f"\nessential items: baseline {tot['be'][0]}/{tot['be'][1]}, candidate {tot['ce'][0]}/{tot['ce'][1]}")
print(f"complete cases (all essential items): baseline {cases_complete['base']}/{cases_complete['n']}, candidate {cases_complete['cand']}/{cases_complete['n']}")
print(f"relationship groups ({len(GROUPS)}): baseline {grp['base']}, candidate {grp['cand']}")
json.dump(added_log, open(f"{E}/added_{sys.argv[1]}.json", "w"), indent=1)
print(f"change-type items: baseline {tot['bc'][0]}/{tot['bc'][1]}, candidate {tot['cc'][0]}/{tot['cc'][1]}")
print(f"records added {added_n}; displaced baseline records holding case evidence {disp_ev}")
