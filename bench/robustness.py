#!/usr/bin/env python3
"""Robustness checks beyond the AML contract (bench/conformance.py).

usage: robustness.py URL [KEY]
Run against a server started with a small cache (e.g. TIDEMARK_CACHE_MB=16) and an empty
database, so that eviction and reloading are exercised.
"""
import concurrent.futures, json, sys, time, urllib.error, urllib.request, uuid

URL = sys.argv[1].rstrip("/")
KEY = sys.argv[2] if len(sys.argv) > 2 else ""
H = {"X-Api-Key": KEY} if KEY else {}
failures = []
run = uuid.uuid4().hex[:8]
TS = 1683554160000


def call(path, body, timeout=600):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "User-Agent": "tidemark-bench/1.0", **H})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, (json.loads(e.read() or b"null") if e.headers.get("Content-Type", "").startswith("application/json") else None)


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  {detail}"))
    if not ok:
        failures.append(name)


def add(user, rid, messages):
    return call("/add", {"request_id": f"{run}:{rid}", "user_id": user, "session_id": f"{run}:s", "messages": messages})


def search(user, query, top_k=100, options=None):
    body = {"query": query, "user_id": user, "top_k": top_k}
    if options:
        body["options"] = options
    return call("/search", body)


# 1. Text without spaces (Chinese), far beyond an embedding input limit
u = f"eval:{run}:cjk"
long_cjk = "我们周末去了海边。" * 2500 + "结尾标记独角兽"
st, body = add(u, "cjk", [{"role": "user", "timestamp": TS, "content": "李华: " + long_cjk}])
check("a 22,000-character message without spaces is accepted", st == 200 and body.get("success"), (st, body))
st, body = search(u, "结尾标记独角兽", 5)
check("its end is searchable", st == 200 and any("独角兽" in d["content"] for d in body["data"]), st)

# 1b. Japanese, and partial phrases in Chinese and Japanese
u = f"eval:{run}:ja"
st, body = add(u, "ja", [
    {"role": "user", "timestamp": TS, "content": "佐藤: 先週の土曜日に東京タワーへ行って、夜景がとてもきれいでした。"},
    {"role": "user", "timestamp": TS, "content": "佐藤: 来月は京都で紅葉を見る予定です。"},
    {"role": "user", "timestamp": TS, "content": "王芳: 我们周末去了海边看日落，非常漂亮。"},
] + [{"role": "user", "timestamp": TS, "content": f"佐藤: 今日は仕事が忙しかった{i}。"} for i in range(12)])
check("Japanese text is accepted", st == 200 and body.get("success"), (st, body))
st, body = search(u, "東京タワー 夜景", 3)
check("a partial Japanese phrase finds its sentence first", st == 200 and body["data"] and "東京タワー" in body["data"][0]["content"], st)
st, body = search(u, "海边日落", 3)
check("a partial Chinese phrase finds its sentence first", st == 200 and body["data"] and "海边" in body["data"][0]["content"], st)

# 2. A large request (about 6 MB, mostly a base64 image as in AML's multimodal format;
#    AML text requests are at most 20 messages or 2,000 words)
u = f"eval:{run}:big"
image = "data:image/png;base64," + "A" * (6 * 1024 * 1024)
big = [{"role": "user", "timestamp": TS, "content": [{"type": "text", "text": "Sam: here is the photo of my new bike bigmarker19"},
                                                      {"type": "image_url", "image_url": {"url": image}}]}]
big += [{"role": "user", "timestamp": TS, "content": f"Sam: message {i} about the bike ride"} for i in range(19)]
st, body = add(u, "big", big)
check("a 6 MB Add request is accepted", st == 200, st)
st, body = search(u, "bigmarker19 bike photo", 5)
check("its text is searchable", st == 200 and any("bigmarker19" in d["content"] for d in body["data"]), st)

# 3. Empty, whitespace-only and image-only messages; missing timestamp; unusual roles
u = f"eval:{run}:odd"
odd = [
    {"role": "user", "timestamp": TS, "content": ""},
    {"role": "user", "timestamp": TS, "content": "   "},
    {"role": "user", "timestamp": TS, "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]},
    {"role": "tool", "content": "tool: weather lookup returned sunny skies for Lisbon"},
    {"content": "Rafael: no role and no timestamp here, mentions the Lisbon tram"},
    {"role": "system", "timestamp": TS, "content": "Persona: Rafael enjoys emoji 🚋 and Arabic مرحبا"},
]
st, body = add(u, "odd", odd)
check("empty, image-only, role-less and timestamp-less messages are accepted", st == 200 and body.get("success"), (st, body))
st, body = search(u, "Lisbon tram", 10)
check("the readable ones are searchable", st == 200 and any("tram" in d["content"] for d in body["data"]), (st, body))
check("emoji and right-to-left text survive", any("🚋" in d["content"] and "مرحبا" in d["content"] for d in body["data"]) or
      any("🚋" in d["content"] for d in search(u, "emoji persona", 10)[1]["data"]), body)

# 4. Odd searches
st, body = search(f"eval:{run}:nobody", "anything", 10)
check("search for an unknown user returns []", st == 200 and body["data"] == [], (st, body))
st, body = search(u, "Lisbon", 0)
check("top_k = 0 returns []", st == 200 and body["data"] == [], (st, body))
st, body = search(u, "Lisbon", 10000)
check("a huge top_k returns at most the stored records", st == 200 and 0 < len(body["data"]) <= 10, len(body.get("data", [])))
st, body = call("/search", {"query": [{"type": "text", "text": "Lisbon tram"}], "user_id": u, "top_k": 5})
check("a query given as content parts is accepted", st == 200 and len(body["data"]) > 0, (st, body))
st, body = search(u, "Which city?", 5, options=[{"label": "A", "text": "Lisbon"}, "B. Porto"])
check("options given as objects or strings are accepted", st == 200, st)

# 5. Concurrent Adds and Searches on the same user
u = f"eval:{run}:race"
def adder(i):
    return add(u, f"race{i}", [{"role": "user", "timestamp": TS + i, "content": f"Kai: race fact number {i} about kites"}])[0]
def searcher(i):
    return search(u, "race fact kites", 100)[0]
with concurrent.futures.ThreadPoolExecutor(6) as pool:
    futures = [pool.submit(adder, i) for i in range(30)] + [pool.submit(searcher, i) for i in range(30)]
    statuses = [f.result() for f in futures]
st, body = search(u, "race fact kites", 100)
check("concurrent Adds and Searches on one user: no errors", all(s in (200, 429) for s in statuses), sorted(set(statuses)))
added = sum(1 for s in statuses[:30] if s == 200)
import re as _re
facts = _re.findall(r"race fact number (\d+) ", " ".join(d["content"] for d in body["data"]) + " ")
check("every accepted Add is stored exactly once", added > 0 and len(facts) == len(set(facts)) == added, (added, len(facts), len(set(facts))))

# 6. Eviction and reload keep each user's data and isolation
users = [f"eval:{run}:many{i}" for i in range(40)]
for i, user in enumerate(users):
    st, _ = add(user, f"many{i}", [{"role": "user", "timestamp": TS, "content": f"Owner{i}: my secret word is zebra{i} " + "filler text " * 3000}])
    if st != 200:
        check(f"add for user {i}", False, st)
ok = True
for i, user in enumerate(users):
    st, body = search(user, f"secret word zebra{i}", 5)
    texts = " ".join(d["content"] for d in body.get("data", []))
    if st != 200 or f"zebra{i} " not in texts or any(f"zebra{j} " in texts for j in range(40) if j != i):
        ok = False
        print("   user", i, st, texts[:80])
        break
check("after eviction and reload every user finds only their own data", ok)

# 7. Eviction under concurrency: users added and searched in parallel while others are
#    evicted and reloaded; every result must contain only the user's own data.
cusers = [f"eval:{run}:conc{i}" for i in range(40)]
def conc_add(i):
    return add(cusers[i], f"conc{i}", [{"role": "user", "timestamp": TS, "content": f"Owner{i}: my code word is falcon{i} " + "padding text " * 3000}])[0]
def conc_search(i):
    st, body = search(cusers[i], f"code word falcon{i}", 5)
    texts = " ".join(d["content"] for d in (body or {}).get("data", []))
    own = f"falcon{i} " in texts
    foreign = any(f"falcon{j} " in texts for j in range(40) if j != i)
    return st, own, foreign
with concurrent.futures.ThreadPoolExecutor(6) as pool:
    add_status = list(pool.map(conc_add, range(40)))
    results = list(pool.map(conc_search, list(range(40)) * 2))
check("parallel Adds of 40 users succeed", all(s in (200,) for s in add_status), sorted(set(add_status)))
check("under parallel eviction every search finds the user's own data", all(st == 200 and own for st, own, _ in results),
      [(i % 40, st, own) for i, (st, own, _) in enumerate(results) if not (st == 200 and own)][:5])
check("under parallel eviction no search sees another user's data", not any(f for _, _, f in results))

print(f"\n{len(failures)} failed" if failures else "\nall checks passed")
sys.exit(1 if failures else 0)
