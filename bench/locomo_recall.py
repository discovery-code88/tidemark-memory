#!/usr/bin/env python3
"""Replay LoCoMo through the Add/Search API the way AML does and measure
evidence recall: the share of each question's gold evidence turns found in the
top-k search results, by category.

usage: locomo_recall.py LOCOMO_JSON [--url http://127.0.0.1:18200] [--key KEY]
       [--run NAME] [--skip-add] [--out results.json] [--conversations conv-26,conv-30]
Adds are 20 messages per request (AML chunks at 20 messages); each message is
"Speaker: text", with a photo caption appended, and carries its session time.
"""
import argparse, json, re, time, urllib.request
from datetime import datetime, timezone

CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "inference", 4: "single-hop", 5: "adversarial"}


def call(url, key, path, body, retries=5):
    data = json.dumps(body).encode()
    for attempt in range(retries):
        req = urllib.request.Request(url + path, data=data, headers={"Content-Type": "application/json", "User-Agent": "tidemark-bench/1.0", **({"X-Api-Key": key} if key else {})})
        try:
            return json.load(urllib.request.urlopen(req, timeout=1800))
        except Exception as e:  # retried like the platform does
            if attempt == retries - 1:
                raise
            print("retry", path, e)
            time.sleep(2 * (attempt + 1))


def session_ms(text):
    # "1:56 pm on 8 May, 2023"
    dt = datetime.strptime(text.strip(), "%I:%M %p on %d %B, %Y").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def turns(conv):
    c = conv["conversation"]
    a = c["speaker_a"]
    n = 1
    while f"session_{n}" in c:
        ts = session_ms(c[f"session_{n}_date_time"])
        msgs = []
        for t in c[f"session_{n}"]:
            text = f"{t['speaker']}: {t['text']}"
            if t.get("blip_caption"):
                text += f" [shares a photo of {t['blip_caption']}]"
            msgs.append({"dia_id": t["dia_id"], "role": "user" if t["speaker"] == a else "assistant", "timestamp": ts, "content": text, "raw": t["text"]})
        yield n, msgs
        n += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("locomo")
    ap.add_argument("--url", default="http://127.0.0.1:18200")
    ap.add_argument("--key", default="")
    ap.add_argument("--run", default="bench")
    ap.add_argument("--skip-add", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--conversations", default="")
    ap.add_argument("--top-k", type=int, default=100)
    args = ap.parse_args()
    data = json.load(open(args.locomo))
    wanted = set(filter(None, args.conversations.split(",")))
    results = []
    for conv in data:
        cid = conv["sample_id"]
        if wanted and cid not in wanted:
            continue
        user = f"eval:{args.run}:locomo:{cid}"
        by_dia = {}
        started = time.time()
        for n, msgs in turns(conv):
            for m in msgs:
                by_dia[m["dia_id"]] = m["raw"]
            if args.skip_add:
                continue
            for i in range(0, len(msgs), 20):
                chunk = msgs[i:i + 20]
                body = {"request_id": f"eval:{args.run}:locomo:{cid}:s{n}:chunk-{i // 20}", "user_id": user, "session_id": f"{cid}:session_{n}",
                        "messages": [{"role": m["role"], "timestamp": m["timestamp"], "content": m["content"]} for m in chunk]}
                r = call(args.url, args.key, "/add", body)
                assert r.get("success") and r.get("request_id") == body["request_id"], r
        add_s = time.time() - started
        for qi, qa in enumerate(conv["qa"]):
            cat = int(qa["category"])
            evidence = [e.strip() for e in qa.get("evidence", []) if e.strip() in by_dia]
            t0 = time.time()
            r = call(args.url, args.key, "/search", {"query": qa["question"], "user_id": user, "top_k": args.top_k})
            ms = (time.time() - t0) * 1000
            contents = [d["content"] for d in r["data"]]
            assert len(contents) <= args.top_k

            def rank_of(dia):
                needle = re.sub(r"\s+", " ", by_dia[dia]).strip()[:80]
                for rank, c in enumerate(contents):
                    if needle and needle in re.sub(r"\s+", " ", c):
                        return rank
                return None

            ranks = {e: rank_of(e) for e in evidence}
            results.append({"sample_id": cid, "question_index": qi, "category": cat, "question": qa["question"],
                            "answer": qa.get("answer", qa.get("adversarial_answer")), "evidence_ranks": ranks, "search_ms": ms,
                            "results": contents})
        print(f"{cid}: added in {add_s:.1f}s, {len(conv['qa'])} questions searched", flush=True)
    report(results, args.top_k)
    if args.out:
        json.dump(results, open(args.out, "w"))


def report(results, top_k):
    ks = [k for k in (10, 25, 50, 100) if k <= top_k]
    print(f"\n{'category':12} {'n':>5} " + " ".join(f"{'all@'+str(k):>8}" for k in ks) + " " + " ".join(f"{'turn@'+str(k):>8}" for k in ks))
    for cat in [1, 2, 3, 4, 5, None]:
        rows = [r for r in results if r["evidence_ranks"] and (cat is None or r["category"] == cat)]
        if not rows:
            continue
        name = CATEGORIES.get(cat, "all")
        all_at = [sum(all(v is not None and v < k for v in r["evidence_ranks"].values()) for r in rows) / len(rows) for k in ks]
        ev = [v for r in rows for v in r["evidence_ranks"].values()]
        turn_at = [sum(v is not None and v < k for v in ev) / len(ev) for k in ks]
        print(f"{name:12} {len(rows):5d} " + " ".join(f"{x:8.1%}" for x in all_at) + " " + " ".join(f"{x:8.1%}" for x in turn_at))
    ms = sorted(r["search_ms"] for r in results)
    if ms:
        print(f"search latency p50 {ms[len(ms)//2]:.0f} ms, p95 {ms[int(len(ms)*0.95)]:.0f} ms")


if __name__ == "__main__":
    main()
