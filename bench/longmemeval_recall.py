#!/usr/bin/env python3
"""Replay LongMemEval-S through the Add/Search API the way AML does and measure
evidence recall: whether the answer sessions (`answer_session_ids`) come back in the top-k.
A session counts as found in the top k when a returned record contains a snippet (the
first 80 characters) of one of its turns of at least 40 characters.

usage: longmemeval_recall.py longmemeval_s_cleaned.json [--url http://127.0.0.1:18200]
       [--key KEY] [--run NAME] [--per-type 8] [--out results.json] [--skip-add]
Each question has its own haystack, so each question is its own user_id. Sessions are
added 20 messages per request with the session's date as timestamp; messages are
"user: ..." / "assistant: ...". A deterministic sample of --per-type questions per
question type keeps the run affordable (the full set is 500 haystacks of ~50 sessions).
"""
import argparse, collections, json, re, time, urllib.request
from datetime import datetime, timezone


def call(url, key, path, body, retries=5):
    data = json.dumps(body).encode()
    for attempt in range(retries):
        req = urllib.request.Request(url + path, data=data, headers={"Content-Type": "application/json", "User-Agent": "tidemark-bench/1.0", **({"X-Api-Key": key} if key else {})})
        try:
            return json.load(urllib.request.urlopen(req, timeout=1800))
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def session_ms(text):
    # "2023/05/20 (Sat) 02:21"
    m = re.match(r"(\d{4})/(\d{2})/(\d{2}) \(\w+\) (\d{2}):(\d{2})", text.strip())
    dt = datetime(*map(int, m.groups()), tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def sample(data, per_type):
    by_type = collections.defaultdict(list)
    for q in data:
        by_type[q["question_type"]].append(q)
    out = []
    for t in sorted(by_type):
        out += sorted(by_type[t], key=lambda q: q["question_id"])[:per_type]
    return out


def norm(text):
    return re.sub(r"\s+", " ", text).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--url", default="http://127.0.0.1:18200")
    ap.add_argument("--key", default="")
    ap.add_argument("--run", default="lme")
    ap.add_argument("--per-type", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--out", default="")
    ap.add_argument("--skip-add", action="store_true")
    ap.add_argument("--search-options", default="{}", help='extra Search fields as JSON, e.g. {"user_weight": 1}')
    args = ap.parse_args()
    questions = sample(json.load(open(args.data)), args.per_type)
    results = []
    for qi, q in enumerate(questions):
        user = f"eval:{args.run}:longmemeval_s:{q['question_id']}"
        answer_sessions = set(q.get("answer_session_ids") or [])
        snippets = {}
        started = time.time()
        for si, (sid, date, turns) in enumerate(zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"])):
            ts = session_ms(date)
            msgs = []
            for t in turns:
                msgs.append({"role": t["role"], "timestamp": ts, "content": f"{t['role']}: {t['content']}"})
                if sid in answer_sessions and len(norm(t["content"])) >= 40:
                    snippets.setdefault(sid, []).append(norm(t["content"])[:80])
            if args.skip_add:
                continue
            for i in range(0, len(msgs), 20):
                body = {"request_id": f"{user}:s{si}:chunk-{i // 20}", "user_id": user,
                        "session_id": f"{q['question_id']}:{sid}", "messages": msgs[i:i + 20]}
                r = call(args.url, args.key, "/add", body)
                assert r.get("success"), r
        add_s = time.time() - started
        t0 = time.time()
        r = call(args.url, args.key, "/search", {"query": q["question"], "user_id": user, "top_k": args.top_k, **json.loads(args.search_options)})
        ms = (time.time() - t0) * 1000
        contents = [norm(d["content"]) for d in r["data"]]
        # rank of the first returned record containing any snippet of each answer session
        ranks = {sid: next((i for i, c in enumerate(contents) if any(sn in c for sn in sn_list)), None)
                 for sid, sn_list in snippets.items()}
        results.append({"question_id": q["question_id"], "question_type": q["question_type"], "question": q["question"],
                        "answer": q["answer"], "question_date": q["question_date"], "evidence_ranks": ranks,
                        "search_ms": ms, "add_s": add_s, "results": [d["content"] for d in r["data"]]})
        print(f"{qi + 1}/{len(questions)} {q['question_type']}: {len(q['haystack_sessions'])} sessions added in {add_s:.1f}s, "
              f"answer sessions found {sum(v is not None for v in ranks.values())}/{len(ranks)}", flush=True)
    report(results)
    if args.out:
        json.dump(results, open(args.out, "w"))


def report(results):
    ks = (10, 25, 50, 100)
    print(f"\n{'question type':28} {'n':>3} " + " ".join(f"{'all@' + str(k):>8}" for k in ks))
    for t in sorted({r["question_type"] for r in results}) + [None]:
        rows = [r for r in results if r["evidence_ranks"] and (t is None or r["question_type"] == t)]
        if not rows:
            continue
        cells = [sum(all(v is not None and v < k for v in r["evidence_ranks"].values()) for r in rows) / len(rows) for k in ks]
        print(f"{t or 'all':28} {len(rows):3d} " + " ".join(f"{c:8.1%}" for c in cells))
    no_ev = sum(1 for r in results if not r["evidence_ranks"])
    if no_ev:
        print(f"({no_ev} questions without marked evidence, e.g. abstention questions)")
    ms = sorted(r["search_ms"] for r in results)
    print(f"search latency p50 {ms[len(ms) // 2]:.0f} ms, max {ms[-1]:.0f} ms")


if __name__ == "__main__":
    main()
