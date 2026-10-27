#!/usr/bin/env python3
"""Replay BEAM (Mohammadta/BEAM on Hugging Face) through the Add/Search API and measure
evidence recall: whether the evidence messages (`source_chat_ids`) come back in the top-k.
A message counts as found when a returned record contains its first 80 characters
(normalised whitespace); messages shorter than 20 characters are not used as evidence.

usage: beam_recall.py 100K.parquet [--url http://127.0.0.1:18200] [--key KEY] [--run NAME]
       [--conversations 20] [--out results.json] [--skip-add]
Needs pyarrow. Each conversation is one user_id; its sessions are added 20 messages per
request, with the session's time anchor as timestamp; messages are "user: ..." /
"assistant: ...". All probing questions of a conversation are then searched.

The output keeps every evidence rank and the length of every returned record, so
truncation (TIDEMARK_MAX_RESULTS / TIDEMARK_MAX_CHARS) can be evaluated offline
with --report results.json --max-results N --max-chars N.
"""
import argparse, ast, collections, json, re, time, urllib.request
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


def anchor_ms(text):
    # "March-15-2024"
    return int(datetime.strptime(text.strip(), "%B-%d-%Y").replace(hour=12, tzinfo=timezone.utc).timestamp() * 1000)


def norm(text):
    return re.sub(r"\s+", " ", text).strip()


def evidence_ids(value):
    if isinstance(value, dict):
        return [i for v in value.values() for i in evidence_ids(v)]
    if isinstance(value, (list, tuple)):
        return [i for v in value for i in evidence_ids(v)]
    return [value] if isinstance(value, int) else []


def run(args):
    import pyarrow.parquet as pq
    rows = pq.read_table(args.data).to_pylist()[: args.conversations]
    results = []
    for ci, row in enumerate(rows):
        user = f"eval:{args.run}:beam:{row['conversation_id']}"
        messages = {}
        started = time.time()
        anchor = None
        for si, session in enumerate(row["chat"]):
            msgs = []
            for m in session:
                anchor = m.get("time_anchor") or anchor
                messages[m["id"]] = m["content"]
                msg = {"role": m["role"], "content": f"{m['role']}: {m['content']}"}
                if anchor:
                    msg["timestamp"] = anchor_ms(anchor)
                msgs.append(msg)
            if args.skip_add:
                continue
            for i in range(0, len(msgs), 20):
                body = {"request_id": f"{user}:s{si}:chunk-{i // 20}", "user_id": user,
                        "session_id": f"{user}:s{si}", "messages": msgs[i:i + 20]}
                r = call(args.url, args.key, "/add", body)
                assert r.get("success"), r
        add_s = time.time() - started
        probing = ast.literal_eval(row["probing_questions"])
        found = total = 0
        for qtype, questions in probing.items():
            for qi, q in enumerate(questions):
                t0 = time.time()
                r = call(args.url, args.key, "/search", {"query": q["question"], "user_id": user, "top_k": args.top_k, **json.loads(args.search_options)})
                ms = (time.time() - t0) * 1000
                contents = [norm(d["content"]) for d in r["data"]]
                snippets = {i: norm(messages[i])[:80] for i in evidence_ids(q.get("source_chat_ids")) if i in messages and len(norm(messages[i])) >= 20}
                ranks = {str(i): next((k for k, c in enumerate(contents) if sn in c), None) for i, sn in snippets.items()}
                found += sum(v is not None for v in ranks.values())
                total += len(ranks)
                results.append({"conversation": row["conversation_id"], "question_type": qtype, "index": qi, "question": q["question"],
                                "rubric": q.get("rubric"), "evidence_ranks": ranks, "lengths": [len(c) for c in contents],
                                "search_ms": ms, "results": [d["content"] for d in r["data"]] if args.keep_results else None})
        chars = sum(len(c) for c in messages.values())
        print(f"{ci + 1}/{len(rows)} conversation {row['conversation_id']}: {len(messages)} messages ({chars / 1e6:.1f}M chars) "
              f"added in {add_s:.0f}s, evidence messages found in top {args.top_k}: {found}/{total}", flush=True)
    if args.out:
        json.dump(results, open(args.out, "w"))
    return results


def report(results, max_results=0, max_chars=0):
    def cut(r):
        # number of results kept under the truncation settings (the first is always kept)
        n = len(r["lengths"]) if not max_results else min(max_results, len(r["lengths"]))
        if max_chars:
            total = 0
            for k, length in enumerate(r["lengths"][:n]):
                total += length
                if k > 0 and total > max_chars:
                    return k
        return n

    ks = (10, 25, 50, 100)
    label = f"max_results={max_results or 'top_k'} max_chars={max_chars or 'off'}"
    print(f"\n{label}\n{'question type':26} {'n':>4} " + " ".join(f"{'any@' + str(k):>7} {'frac@' + str(k):>8}" for k in ks)
          + f" {'kept':>6} {'chars':>7}")
    for t in sorted({r["question_type"] for r in results}) + [None]:
        rows = [r for r in results if r["evidence_ranks"] and (t is None or r["question_type"] == t)]
        if not rows:
            continue
        cells = []
        for k in ks:
            hit = lambda r, v: v is not None and v < min(k, cut(r))
            anyk = sum(any(hit(r, v) for v in r["evidence_ranks"].values()) for r in rows) / len(rows)
            frac = sum(sum(hit(r, v) for v in r["evidence_ranks"].values()) / len(r["evidence_ranks"]) for r in rows) / len(rows)
            cells.append(f"{anyk:7.1%} {frac:8.1%}")
        kept = sum(cut(r) for r in rows) / len(rows)
        chars = sum(sum(r["lengths"][:cut(r)]) for r in rows) / len(rows)
        print(f"{t or 'all':26} {len(rows):4d} " + " ".join(cells) + f" {kept:6.1f} {chars:7.0f}")
    no_ev = sum(1 for r in results if not r["evidence_ranks"])
    if no_ev:
        print(f"({no_ev} questions without usable evidence ids, e.g. abstention questions)")
    ms = sorted(r["search_ms"] for r in results)
    print(f"search latency p50 {ms[len(ms) // 2]:.0f} ms, max {ms[-1]:.0f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?")
    ap.add_argument("--url", default="http://127.0.0.1:18200")
    ap.add_argument("--key", default="")
    ap.add_argument("--run", default="beam")
    ap.add_argument("--conversations", type=int, default=1000)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--out", default="")
    ap.add_argument("--skip-add", action="store_true")
    ap.add_argument("--search-options", default="{}", help='extra Search fields as JSON, e.g. {"user_weight": 1}')
    ap.add_argument("--keep-results", action="store_true", help="store the returned records in --out")
    ap.add_argument("--report", default="", help="report an earlier --out file instead of running")
    ap.add_argument("--max-results", type=int, default=0)
    ap.add_argument("--max-chars", type=int, default=0)
    args = ap.parse_args()
    results = json.load(open(args.report)) if args.report else run(args)
    report(results, args.max_results, args.max_chars)


if __name__ == "__main__":
    main()
