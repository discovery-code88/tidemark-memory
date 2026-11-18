#!/usr/bin/env python3
"""Checks of the memory API beyond Add and Search (list, get, correct, delete,
delete user, history), including user isolation and that Search reflects every
change at once.

usage: api_check.py URL KEY
Run against a server with an empty database (placeholder embeddings are fine).
"""
import json, sys, urllib.error, urllib.parse, urllib.request, uuid

URL = sys.argv[1].rstrip("/")
KEY = sys.argv[2]
H = {"Authorization": f"Token {KEY}", "Content-Type": "application/json", "User-Agent": "tidemark-bench/1.0"}
failures = []
run = uuid.uuid4().hex[:8]
TS = 1683554160000


def call(method, path, body=None, headers=H):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode() if body is not None else None, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        raw = e.read()
        return e.code, (json.loads(raw) if raw[:1] in (b"{", b"[") else None)


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  {detail}"))
    if not ok:
        failures.append(name)


q = urllib.parse.quote
alice, bob = f"api:{run}:alice", f"api:{run}:bob"
for user, text in [(alice, "Alice: I keep a red kite in the garage."), (bob, "Bob: My boat is called Seagull.")]:
    st, _ = call("POST", "/add", {"request_id": f"{run}:{user}", "user_id": user, "session_id": f"{user}:s1",
                                  "messages": [{"role": "user", "timestamp": TS, "content": text}]})
    check(f"add for {user.split(':')[-1]}", st == 200, st)

st, body = call("GET", f"/memories?user_id={q(alice)}")
check("list returns the user's memories", st == 200 and body["total"] == 1 and "kite" in body["data"][0]["content"], (st, body))
mid = body["data"][0]["id"]
check("list carries id, session_id, created_at", all(k in body["data"][0] for k in ("id", "session_id", "created_at")), body["data"][0])

st, body = call("GET", f"/memories?user_id={q(alice)}&session_id={q(alice + ':nope')}")
check("list filters by session", st == 200 and body["total"] == 0, body)

st, body = call("GET", f"/memories/{q(mid, safe='')}?user_id={q(alice)}")
check("get returns one memory", st == 200 and body["data"]["id"] == mid, (st, body))

st, _ = call("GET", f"/memories/{q(mid, safe='')}?user_id={q(bob)}")
check("another user cannot get it (404)", st == 404, st)
st, _ = call("PATCH", f"/memories/{q(mid, safe='')}", {"user_id": bob, "content": "hijacked"})
check("another user cannot change it (404)", st == 404, st)
st, _ = call("DELETE", f"/memories/{q(mid, safe='')}?user_id={q(bob)}")
check("another user cannot delete it (404)", st == 404, st)

st, _ = call("GET", f"/memories?user_id={q(alice)}", headers={"User-Agent": "tidemark-bench/1.0"})
check("list without credentials is rejected (401)", st == 401, st)
st, _ = call("DELETE", f"/users/{q(alice, safe='')}", headers={"User-Agent": "tidemark-bench/1.0"})
check("delete user without credentials is rejected (401)", st == 401, st)

st, body = call("PATCH", f"/memories/{q(mid, safe='')}", {"user_id": alice, "content": "Alice: I keep a blue kite in the garage."})
check("a correction is accepted", st == 200 and "blue" in body["data"]["content"] and body["data"]["id"] == mid, (st, body))
st, body = call("POST", "/search", {"query": "kite colour", "user_id": alice, "top_k": 5})
texts = " ".join(d["content"] for d in body["data"])
check("search shows the corrected text at once", "blue kite" in texts and "red kite" not in texts, texts)
st, body = call("PATCH", f"/memories/{q(mid, safe='')}", {"user_id": alice, "content": "  "})
check("an empty correction is rejected (400)", st == 400, st)

st, body = call("GET", f"/memories/{q(mid, safe='')}/history?user_id={q(alice)}")
check("history keeps the previous text", st == 200 and len(body["data"]) == 1 and body["data"][0]["action"] == "updated" and "red kite" in body["data"][0]["content"], body)
st, body = call("GET", f"/memories/{q(mid, safe='')}/history?user_id={q(bob)}")
check("another user sees no history", st == 200 and body["data"] == [], body)

st, body = call("DELETE", f"/memories/{q(mid, safe='')}?user_id={q(alice)}")
check("delete a memory", st == 200 and body.get("deleted") == mid, (st, body))
st, body = call("POST", "/search", {"query": "kite", "user_id": alice, "top_k": 5})
check("search no longer returns it", st == 200 and not any("kite" in d["content"] for d in body["data"]), body)
st, _ = call("GET", f"/memories/{q(mid, safe='')}?user_id={q(alice)}")
check("get after delete is 404", st == 404, st)
st, body = call("GET", f"/memories/{q(mid, safe='')}/history?user_id={q(alice)}")
check("history records the deletion", [h["action"] for h in body["data"]] == ["updated", "deleted"], body)
st, _ = call("DELETE", f"/memories/{q(mid, safe='')}?user_id={q(alice)}")
check("deleting again is 404", st == 404, st)

st, body = call("POST", "/search", {"query": "boat", "user_id": bob, "top_k": 5})
check("the other user's data is untouched", st == 200 and any("Seagull" in d["content"] for d in body["data"]), body)

st, body = call("DELETE", f"/users/{q(bob, safe='')}")
check("delete a user", st == 200 and body.get("deleted_memories") == 1, (st, body))
st, body = call("GET", f"/memories?user_id={q(bob)}")
check("the deleted user has no memories", st == 200 and body["total"] == 0, body)
st, body = call("POST", "/search", {"query": "boat", "user_id": bob, "top_k": 5})
check("search for the deleted user returns nothing", st == 200 and body["data"] == [], body)
st, _ = call("POST", "/add", {"request_id": f"{run}:{bob}", "user_id": bob, "session_id": f"{bob}:s1",
                               "messages": [{"role": "user", "timestamp": TS, "content": "Bob: My boat is called Seagull."}]})
st2, body = call("GET", f"/memories?user_id={q(bob)}")
check("after deletion the same request_id stores again (request log cleared)", st == 200 and body["total"] == 1, (st, body))

print(f"\n{len(failures)} failed" if failures else "\nall checks passed")
sys.exit(1 if failures else 0)
