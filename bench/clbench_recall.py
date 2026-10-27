#!/usr/bin/env python3
"""Replay CL-bench (tencent/CL-bench on Hugging Face) multi-turn tasks through the
Add/Search API and measure how much of the reference document comes back.

usage: clbench_recall.py CL-bench.jsonl [--url http://127.0.0.1:18200] [--key KEY]
       [--run NAME] [--category "Rule System Application"] [--limit 200]
       [--search-options '{"expand_hits": 3}'] [--skip-add]
Only tasks with earlier turns are used (in single-turn tasks the document is part of
the question itself, so memory plays no part). The earlier user/assistant turns are
added as one session (the system prompt is not memory: AML passes it to the answer
model); the final user message is the query. The document is the longest earlier
user message; it counts as covered where a returned record contains its probe
snippets (80 characters taken every 1,000 characters).
"""
import argparse, json, re, time, urllib.request


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


def norm(text):
    return re.sub(r"\s+", " ", text).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--url", default="http://127.0.0.1:18200")
    ap.add_argument("--key", default="")
    ap.add_argument("--run", default="clbench")
    ap.add_argument("--category", default="Rule System Application")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--search-options", default="{}")
    ap.add_argument("--skip-add", action="store_true")
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(args.data)]
    tasks = [r for r in rows if r["metadata"]["context_category"] == args.category and len(r["messages"]) > 2][: args.limit]
    ts = 1735732800000
    covered, full, sizes = [], 0, []
    for n, task in enumerate(tasks):
        user = f"eval:{args.run}:clbench:{task['metadata']['task_id']}"
        earlier = [m for m in task["messages"][1:-1] if m["role"] in ("user", "assistant")]
        if not args.skip_add:
            msgs = [{"role": m["role"], "timestamp": ts, "content": m["content"]} for m in earlier]
            for c in range(0, len(msgs), 20):
                r = call(args.url, args.key, "/add", {"request_id": f"{user}:c{c}", "user_id": user, "session_id": f"{user}:s", "messages": msgs[c:c + 20]})
                assert r.get("success"), r
        doc = norm(max((m["content"] for m in earlier if m["role"] == "user"), key=len))
        probes = [doc[i:i + 80] for i in range(0, max(len(doc) - 80, 1), 1000)]
        query = task["messages"][-1]["content"]
        res = call(args.url, args.key, "/search", {"query": query, "user_id": user, "top_k": args.top_k, **json.loads(args.search_options)})
        text = " ".join(norm(d["content"]) for d in res["data"])
        share = sum(p in text for p in probes) / len(probes)
        covered.append(share)
        full += share >= 0.9
        sizes.append(sum(len(d["content"]) for d in res["data"]))
        if (n + 1) % 25 == 0:
            print(f"{n + 1}/{len(tasks)} mean document share {sum(covered) / len(covered):.1%}", flush=True)
    print(f"\n{args.category}: {len(tasks)} multi-turn tasks, options {args.search_options}")
    print(f"document share returned: mean {sum(covered) / len(covered):.1%}, tasks with >= 90%: {full / len(tasks):.1%}, "
          f"returned characters mean {sum(sizes) / len(sizes):,.0f}")


if __name__ == "__main__":
    main()
