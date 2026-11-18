#!/usr/bin/env python3
"""Replays whole PersonaMem conversations into Tidemark (no questions, no answers), e.g. to audit preference extraction.
usage: personamem_replay.py --url URL --key KEY --users N --seed S [--label L]   (waits until extraction queues are empty)"""
import argparse, json, os, random, re, time, urllib.request
p = argparse.ArgumentParser()
p.add_argument("--url", required=True); p.add_argument("--key", required=True)
p.add_argument("--users", type=int, default=4); p.add_argument("--seed", type=int, default=0); p.add_argument("--label", default="audit")
p.add_argument("--data", default=os.path.expanduser("~/.cache/unlimited-memory-recall/practice"))
a = p.parse_args()
H = {"Authorization": f"Token {a.key}", "Content-Type": "application/json"}
def call(method, path, body=None):
    req = urllib.request.Request(a.url.rstrip("/") + path, data=json.dumps(body).encode() if body is not None else None, headers=H, method=method)
    return json.loads(urllib.request.urlopen(req, timeout=600).read() or b"null")
ctx = {}
for line in open(os.path.join(a.data, "personamem_shared_contexts_32k.jsonl")):
    ctx.update(json.loads(line))
chosen = random.Random(a.seed).sample(sorted(ctx), a.users)
print("users:", [c[:16] for c in chosen], flush=True)
for cid in chosen:
    user = f"pm:{a.label}:{cid[:16]}"
    msgs = [{"role": m["role"], "timestamp": 1683554160000 + i * 60000, "content": re.sub(r"^(User|Assistant):\s*", "", m["content"].strip())}
            for i, m in enumerate(ctx[cid]) if m["role"] in ("user", "assistant")]
    for j in range(0, len(msgs), 20):
        call("POST", "/add", {"request_id": f"{cid}:{j}", "user_id": user, "session_id": cid[:16], "messages": msgs[j:j + 20]})
    while call("GET", f"/preferences?user_id={urllib.request.quote(user)}")["pending_records"]:
        time.sleep(5)
    print(f"{user}: {len(msgs)} messages replayed, {len(call('GET', f'/preferences?user_id={urllib.request.quote(user)}')['data'])} current preferences", flush=True)
