#!/usr/bin/env python3
"""Development diagnostic suite: build memories, retrieval, curated sets (build); answer run (run); MC scoring (score).
usage: run_diag.py build|run|score   (directory: ~/.cache/tidemark-dev/diag/run)"""
import hashlib, json, os, random, re, sys, time, urllib.error, urllib.request, datetime as dt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cases import CASES

D = os.path.expanduser("~/.cache/tidemark-dev/diag/run")
DATA = os.path.expanduser("~/.cache/unlimited-memory-recall/practice/personamem_shared_contexts_32k.jsonl")
URL = "http://127.0.0.1:18390"
H = {"Authorization": "Token k", "Content-Type": "application/json"}
CONV = {"A3": "3cc19ddf369632cd", "A4": "40027b883505bfc5", "H": "e898d03fec683b1c"}
SEED = 20261105
T0 = dt.datetime(2023, 5, 8, 13, 56, tzinfo=dt.timezone.utc)
MC_SYSTEM = "You are a personal assistant. Use the memories about the user where they are relevant. Reply with the letter of the best reply only, e.g. (b)."
FR_SYSTEM = "You are a personal assistant. Use the memories about the user where they are relevant."


def norm(t):
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').lower()
    return " ".join(t.split())


def call(method, path, body=None):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode() if body is not None else None, headers=H, method=method)
    return json.loads(urllib.request.urlopen(req, timeout=600).read() or b"null")


def ms(t):
    return int(t.timestamp() * 1000)


def conversation(source):
    full = {}
    for line in open(DATA):
        full.update(json.loads(line))
    cid = next(c for c in full if c.startswith(CONV[source]))
    out = []
    for i, m in enumerate(m for m in full[cid] if m["role"] in ("user", "assistant")):
        out.append({"role": m["role"], "timestamp": ms(T0 + dt.timedelta(minutes=i)), "content": re.sub(r"^(User|Assistant):\s*", "", m["content"].strip())})
    return out


def with_inserts(case, host):
    """The host conversation plus the case's inserts: dated ones at noon UTC of their date, undated ones at seeded
    positions in the second half of the host conversation (between two messages)."""
    msgs = list(host)
    rnd = random.Random(f"{SEED}:insert:{case['id']}")
    positions = sorted(rnd.sample(range(len(host) // 2, len(host) - 1), len([x for x in case["inserts"] if x[0] is None])))
    k = 0
    for date, text in case["inserts"]:
        if date:
            t = ms(dt.datetime.fromisoformat(date + "T12:00:00+00:00"))
        else:
            p = positions[k]; k += 1
            t = (host[p]["timestamp"] + host[p + 1]["timestamp"]) // 2
        msgs.append({"role": "user", "timestamp": t, "content": text, "insert": True})
    msgs.sort(key=lambda m: m["timestamp"])
    return msgs


def user_of(case):
    return f"diag:{case['source']}" if case["source"] != "H" else f"diag:H:{case['id']}"


def build():
    os.makedirs(D, exist_ok=True)
    host = conversation("H")
    convs = {"diag:A3": conversation("A3"), "diag:A4": conversation("A4")}
    for c in CASES:
        if c["source"] == "H":
            convs[user_of(c)] = with_inserts(c, host)
    for user, msgs in convs.items():
        clean = [{k: v for k, v in m.items() if k != "insert"} for m in msgs]
        for j in range(0, len(clean), 20):
            call("POST", "/add", {"request_id": f"{user}:{j}", "user_id": user, "session_id": user, "messages": clean[j:j + 20]})
    json.dump(convs, open(f"{D}/conversations.json", "w"))
    import sqlite3
    db = sqlite3.connect(f"file:{D}/diag.sqlite?mode=ro", uri=True)
    out, paused, review = {}, {}, {}
    for c in CASES:
        user = user_of(c)
        q_end = dt.datetime.fromisoformat(c["q_date"] + "T00:00:00+00:00")
        recs = [{"id": i, "content": t, "created_at": a} for i, t, a in db.execute("SELECT id, content, created_at FROM records WHERE user_id = ? ORDER BY seq", (user,))]
        later = [r["id"] for r in recs if r["created_at"] and dt.datetime.fromisoformat(r["created_at"].replace("Z", "+00:00")) >= q_end]
        visible = [r for r in recs if r["id"] not in later]
        # Evidence records.
        ev = []
        for key, role in c["evidence"]:
            hits = [r["id"] for r in visible if norm(key) in norm(r["content"])]
            if not hits:
                paused.setdefault(c["id"], []).append(f"evidence not found in any record before the question: {key!r}")
            ev.append({"key": key, "role": role, "records": hits})
        # Retrieval (never modified).
        hits = call("POST", "/search", {"query": c["request"], "user_id": user, "top_k": 10})["data"]
        retrieved = [{"id": h["id"], "content": h["content"], "created_at": h.get("created_at")} for h in hits]
        late_ret = [h["id"] for h in retrieved if h["id"] in later]
        if late_ret:
            paused.setdefault(c["id"], []).append(f"retrieved records after the question date: {late_ret}")
        # Curated: all evidence records plus seeded padding; padding excludes records on the case's topic.
        ev_ids = sorted({i for e in ev for i in e["records"]}, key=lambda i: [r["id"] for r in visible].index(i))
        excluded = [r["id"] for r in visible if r["id"] not in ev_ids and any(t in norm(r["content"]) for t in c["topic"])]
        pool = [r for r in visible if r["id"] not in ev_ids and r["id"] not in excluded]
        if len(ev_ids) > 10:
            paused.setdefault(c["id"], []).append(f"{len(ev_ids)} evidence records exceed the 10-record budget")
        rnd = random.Random(SEED + CASES.index(c))
        padding = rnd.sample(pool, max(0, 10 - len(ev_ids)))
        curated = [r for r in visible if r["id"] in ev_ids] + padding
        rnd.shuffle(curated)
        recall = lambda role: (sum(1 for e in ev if e["role"] == role and any(i in {h["id"] for h in retrieved} for i in e["records"])),
                               sum(1 for e in ev if e["role"] == role))
        out[c["id"]] = {"user": user, "retrieved": retrieved, "curated": curated, "evidence": ev, "padding_excluded": excluded,
                        "essential_recall": None if c.get("request_only") or c.get("no_essential") else recall("essential"),
                        "context_recall": recall("context"),
                        "chars": {"retrieved": sum(len(r["content"]) for r in retrieved), "curated": sum(len(r["content"]) for r in curated)},
                        "count": {"retrieved": len(retrieved), "curated": len(curated)}}
        # For the contradiction review: every visible record on the topic that is not an evidence record.
        review[c["id"]] = [r["content"][:400] for r in visible if r["id"] in excluded]
    json.dump(out, open(f"{D}/sets.json", "w"), indent=1)
    json.dump(review, open(f"{D}/topic_review.json", "w"), indent=1)
    json.dump(paused, open(f"{D}/paused.json", "w"), indent=1)
    for cid, v in out.items():
        print(f"{cid}: essential {v['essential_recall'] or 'n/a'}  context {v['context_recall']}  chars retrieved/curated {v['chars']['retrieved']}/{v['chars']['curated']}  topic records excluded {len(v['padding_excluded'])}")
    print("paused:", paused or "none")


def prompt(case, records, fmt, order=None):
    mem = "\n\n".join(f"[{i + 1}] {r['content']}" for i, r in enumerate(records))
    head = f"Current date: {case['q_date']}\n\nMemories about the user:\n{mem}\n\nThe user now says: {case['request']}\n\n"
    if fmt == "free":
        return head + "Reply to the user in a few sentences."
    letters = "abcd"
    return head + "Which reply is best?\n" + "\n".join(f"({letters[i]}) {case['options'][k]['text']}" for i, k in enumerate(order))


def answer(system, user, ledger, cap, retries=12):
    url = os.environ["EVAL_ANSWER_URL"].rstrip("/") + "/chat/completions"
    body = {"model": os.environ["EVAL_ANSWER_MODEL"], "temperature": 0, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['EVAL_ANSWER_API_KEY']}"}
    for attempt in range(retries):
        n = sum(1 for _ in open(ledger)) if os.path.exists(ledger) else 0
        if n >= cap:
            print(f"CAP reached: {n} answer attempts", flush=True)
            sys.exit(3)
        open(ledger, "a").write(f"{time.time():.0f}\n")
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.loads(r.read())
            return d["choices"][0]["message"]["content"] or "", attempt + 1
        except (urllib.error.URLError, OSError, KeyError) as e:
            code = getattr(e, "code", 0)
            if code in (400, 401, 403, 404):
                raise
            time.sleep(min(300, (5 if code == 429 else 1) * 2 ** attempt))
    return None, retries


def run():
    sets = json.load(open(f"{D}/sets.json"))
    paused = json.load(open(f"{D}/paused.json"))
    cases = [c for c in CASES if c["id"] not in paused]
    out_path, ledger = f"{D}/answers.jsonl", f"{D}/attempts.log"
    done = {(r["case"], r["condition"], r["format"], r["rep"]) for r in map(json.loads, open(out_path))} if os.path.exists(out_path) else set()
    out = open(out_path, "a")
    for rep in range(1, 6):
        order_cases = list(cases); random.Random(f"{SEED}:cases:{rep}").shuffle(order_cases)
        for c in order_cases:
            tasks = [(cond, fmt) for cond in ("curated", "retrieved") for fmt in ("mc", "free")]
            random.Random(f"{SEED}:tasks:{c['id']}:{rep}").shuffle(tasks)
            for cond, fmt in tasks:
                if (c["id"], cond, fmt, rep) in done:
                    continue
                records = sets[c["id"]][cond]
                if fmt == "mc":
                    order = list(range(4)); random.Random(f"{SEED}:mc:{c['id']}:{cond}:{rep}").shuffle(order)
                    reply, attempts = answer(MC_SYSTEM, prompt(c, records, "mc", order), ledger, 750)
                    m = re.search(r"\(([a-d])\)", reply or "") or re.search(r"\b([a-d])\b", (reply or "").lower())
                    picked = order["abcd".index(m.group(1))] if m else None
                    rec = {"order": order, "picked_option": picked}
                else:
                    reply, attempts = answer(FR_SYSTEM, prompt(c, records, "free"), ledger, 750)
                    rec = {}
                if reply is None:
                    print(f"RETRY_LIMIT {c['id']} {cond} {fmt} rep {rep}", flush=True)
                    continue
                rec.update({"case": c["id"], "condition": cond, "format": fmt, "rep": rep, "reply": reply, "attempts": attempts})
                out.write(json.dumps(rec) + "\n"); out.flush()
                print(f"rep {rep} {c['id']} {cond} {fmt} attempts={attempts}", flush=True)
    open(f"{D}/finished", "w").write("ok\n")


def score():
    rows = [json.loads(l) for l in open(f"{D}/answers.jsonl")]
    by = {c["id"]: c for c in CASES}
    print(f"{'case':6s} {'cond':9s} n  support-fail compliance-fail  P(strong/weak/none)  unparsed")
    for c in CASES:
        for cond in ("curated", "retrieved"):
            rs = [r for r in rows if r["case"] == c["id"] and r["condition"] == cond and r["format"] == "mc"]
            labs = [by[c["id"]]["options"][r["picked_option"]] for r in rs if r["picked_option"] is not None]
            p = {k: sum(1 for l in labs if l["P"] == k) for k in ("strong", "weak", "none")}
            print(f"{c['id']:6s} {cond:9s} {len(rs)}  {sum(1 for l in labs if not l['S'])}            {sum(1 for l in labs if not l['C'])}               {p['strong']}/{p['weak']}/{p['none']}               {sum(1 for r in rs if r['picked_option'] is None)}")


if __name__ == "__main__":
    {"build": build, "run": run, "score": score}[sys.argv[1]]()
