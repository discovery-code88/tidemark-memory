#!/usr/bin/env python3
"""Checks the Add/Search API against the Agent Memory Leaderboard contract.

usage: conformance.py URL KEY
Run against a server started with TIDEMARK_API_KEYS=KEY and an empty database.
"""
import concurrent.futures, json, sys, urllib.error, urllib.request, uuid

URL, KEY = sys.argv[1].rstrip("/"), sys.argv[2]
failures = []


def call(path, body=None, headers=None, method=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(URL + path, data=data, method=method or ("POST" if body is not None else "GET"),
                                 headers={"Content-Type": "application/json", "User-Agent": "tidemark-bench/1.0", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  {detail}"))
    if not ok:
        failures.append(name)


auth = {"X-Api-Key": KEY}
run = uuid.uuid4().hex[:8]
user = f"eval:{run}:conformance:u1"

status, _ = call("/health")
check("health is reachable without credentials (2xx)", 200 <= status < 300, status)

add = {"request_id": f"eval:{run}:c0", "user_id": user, "session_id": f"eval:{run}:s0",
       "messages": [{"role": "user", "timestamp": 1683554160000, "content": "Caroline: I adopted a puppy named Ned yesterday."},
                    {"role": "assistant", "timestamp": 1683554160000, "content": "Melanie: Congrats! I painted a sunrise last week."}]}
status, _ = call("/add", add)
check("Add without credentials is rejected", status in (401, 403), status)
for scheme, headers in [("X-Api-Key", {"X-Api-Key": KEY}), ("Bearer", {"Authorization": f"Bearer {KEY}"}), ("Token", {"Authorization": f"Token {KEY}"})]:
    status, _ = call("/search", {"query": "x", "user_id": "nobody", "top_k": 5}, headers)
    check(f"{scheme} credentials accepted", status == 200, status)

status, body = call("/add", add, auth)
check("Add returns 200 with success true", status == 200 and body.get("success") is True, body)
check("Add echoes request_id, user_id and session_id",
      all(body.get(k) == add[k] for k in ("request_id", "user_id", "session_id")), body)

status, body = call("/search", {"query": "What is the name of Caroline's puppy?", "user_id": user, "top_k": 100}, auth)
contents = [d.get("content", "") for d in body.get("data", [])]
check("an added message is searchable as soon as Add returns", any("Ned" in c for c in contents), contents)
check("records have id and non-empty content", all(d.get("id") and d.get("content") for d in body["data"]), body)
check("records carry the resolved date of relative times", any("yesterday = 7 May 2023" in c for c in contents), contents)

status, body = call("/search", {"query": "puppy", "user_id": user, "top_k": 100}, auth)
before = len(body["data"])
status, _ = call("/add", add, auth)
status, body = call("/search", {"query": "puppy", "user_id": user, "top_k": 100}, auth)
check("a retried request_id does not duplicate records", before > 0 and len(body["data"]) == before, (before, len(body["data"])))

status, body = call("/search", {"query": "puppy", "user_id": user, "top_k": 1}, auth)
check("Search returns at most top_k records", len(body["data"]) <= 1, len(body["data"]))
status, body = call("/search", {"query": "puppy", "user_id": "eval:other:user", "top_k": 100}, auth)
check("another user_id sees nothing (sample isolation)", status == 200 and body["data"] == [], body)
status, body = call("/search", {"query": "Which pet?", "options": ["A. A puppy", "B. A cat"], "user_id": user, "top_k": 10}, auth)
check("multiple-choice options are accepted", status == 200 and len(body["data"]) > 0, body)
scores = [d.get("score") for d in body["data"]]
check("results are ordered by descending score", scores == sorted(scores, reverse=True), scores)

parts = {"request_id": f"eval:{run}:mm", "user_id": user, "session_id": f"eval:{run}:s1",
         "messages": [{"role": "user", "timestamp": 1683554160000, "content": [{"type": "text", "text": "Caroline: look at my kayak photo"},
                                                                              {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]}
status, body = call("/add", parts, auth)
check("content given as a list of parts is accepted", status == 200 and body.get("success") is True, body)

long_text = "Caroline: " + " ".join(f"word{i}" for i in range(2500)) + " zebracornfinale"
status, body = call("/add", {"request_id": f"eval:{run}:long", "user_id": user, "session_id": f"eval:{run}:s2",
                             "messages": [{"role": "user", "content": long_text}]}, auth)
status, body = call("/search", {"query": "zebracornfinale", "user_id": user, "top_k": 3}, auth)
check("the end of a long message is searchable", any("zebracornfinale" in d["content"] for d in body["data"]), body)

concurrent_user = f"eval:{run}:conformance:u2"


def concurrent_add(i):
    return call("/add", {"request_id": f"eval:{run}:cc{i}", "user_id": concurrent_user, "session_id": f"eval:{run}:cs",
                         "messages": [{"role": "user", "timestamp": 1683554160000, "content": f"Dave: fact number {i} about cars"}]}, auth)[0]


with concurrent.futures.ThreadPoolExecutor(8) as pool:
    statuses = list(pool.map(concurrent_add, range(24)))
status, body = call("/search", {"query": "fact about cars", "user_id": concurrent_user, "top_k": 100}, auth)
check("concurrent Adds all succeed and are all stored", statuses == [200] * 24 and len(body["data"]) == 24, (statuses, len(body["data"])))

status, body = call("/add", {"request_id": f"eval:{run}:bad", "user_id": user}, auth)
check("a malformed Add is a 4xx, not a 5xx", 400 <= status < 500, status)

print(f"\n{len(failures)} failed" if failures else "\nall checks passed")
sys.exit(1 if failures else 0)
