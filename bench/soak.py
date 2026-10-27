#!/usr/bin/env python3
"""Soak test: AML-like load from parallel clients over many LongMemEval-S haystacks.

usage: soak.py longmemeval_s_cleaned.json URL SERVER_PID [--haystacks 300] [--workers 4]
Each worker takes the next haystack, adds its sessions (20 messages per request), searches
its question, and every few haystacks re-searches a random earlier user (forcing reloads of
evicted users). Reports errors by status, latency percentiles, and server memory over time.
"""
import argparse, collections, concurrent.futures, json, random, re, subprocess, threading, time, urllib.error, urllib.request
from datetime import datetime, timezone


def session_ms(text):
    m = re.match(r"(\d{4})/(\d{2})/(\d{2}) \(\w+\) (\d{2}):(\d{2})", text.strip())
    return int(datetime(*map(int, m.groups()), tzinfo=timezone.utc).timestamp() * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("url")
    ap.add_argument("pid", type=int)
    ap.add_argument("--haystacks", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--offset", type=int, default=0, help="skip the first N haystacks")
    args = ap.parse_args()
    data = json.load(open(args.data))[args.offset: args.offset + args.haystacks]
    lock = threading.Lock()
    statuses = collections.Counter()
    lat = collections.defaultdict(list)
    done_users = []
    rss = []
    stop = threading.Event()

    def call(path, body, kind):
        req = urllib.request.Request(args.url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "User-Agent": "tidemark-bench/1.0"})
        for attempt in range(10):
            t = time.time()
            try:
                with urllib.request.urlopen(req, timeout=1800) as r:
                    out = json.loads(r.read())
                    code = r.status
            except urllib.error.HTTPError as e:
                code, out = e.code, None
            except Exception:
                code, out = -1, None
            with lock:
                statuses[(kind, code)] += 1
                if code == 200:
                    lat[kind].append(time.time() - t)
            if code == 200:
                return out
            time.sleep(2 if code == 429 else 5)
        return None

    def work(i):
        q = data[i]
        user = f"soak:{q['question_id']}"
        for si, (sid, date, turns) in enumerate(zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"])):
            msgs = [{"role": t["role"], "timestamp": session_ms(date), "content": f"{t['role']}: {t['content']}"} for t in turns]
            for c in range(0, len(msgs), 20):
                call("/add", {"request_id": f"{user}:s{si}:c{c}", "user_id": user, "session_id": f"{user}:{sid}", "messages": msgs[c:c + 20]}, "add")
        call("/search", {"query": q["question"], "user_id": user, "top_k": 100}, "search")
        with lock:
            done_users.append((user, q["question"]))
            old = random.choice(done_users) if i % 3 == 0 and len(done_users) > 10 else None
        if old:
            call("/search", {"query": old[1], "user_id": old[0], "top_k": 100}, "search_old_user")

    def sampler():
        while not stop.is_set():
            try:
                rss.append((time.time(), int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(args.pid)]).strip()) // 1024))
            except Exception:
                pass
            stop.wait(15)

    started = time.time()
    threading.Thread(target=sampler, daemon=True).start()
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        for n, _ in enumerate(pool.map(work, range(len(data))), 1):
            if n % 25 == 0:
                print(f"{n}/{len(data)} haystacks, {time.time() - started:.0f}s, server {rss[-1][1] if rss else '?'} MB", flush=True)
    stop.set()
    print(f"\nfinished {len(data)} haystacks in {time.time() - started:.0f}s with {args.workers} workers")
    print("responses by (kind, status):", dict(sorted(statuses.items())))
    for kind, xs in sorted(lat.items()):
        xs = sorted(xs)
        print(f"{kind:17s} n={len(xs):5d} p50 {xs[len(xs) // 2] * 1000:7.0f} ms  p95 {xs[int(len(xs) * .95)] * 1000:7.0f} ms  max {xs[-1] * 1000:7.0f} ms")
    if rss:
        mems = [m for _, m in rss]
        thirds = [mems[: len(mems) // 3], mems[len(mems) // 3: 2 * len(mems) // 3], mems[2 * len(mems) // 3:]]
        print("server memory MB: max", max(mems), "| mean by thirds of the run:", [round(sum(t) / max(len(t), 1)) for t in thirds])


if __name__ == "__main__":
    main()
