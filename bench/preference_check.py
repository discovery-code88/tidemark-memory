#!/usr/bin/env python3
"""Checks of preference memory with the rules (run the server with
TIDEMARK_PREFERENCES=rules): extraction from the user's own messages, search slots, corrections
as versions, history, retraction when a memory is deleted, manual preferences,
user isolation and deletion.

usage: preference_check.py URL KEY
Run against a server with an empty database (placeholder embeddings are fine).
"""
import json, sys, urllib.error, urllib.parse, urllib.request, uuid

URL = sys.argv[1].rstrip("/")
KEY = sys.argv[2]
H = {"Authorization": f"Token {KEY}", "Content-Type": "application/json", "User-Agent": "tidemark-bench/1.0"}
failures = []
run = uuid.uuid4().hex[:8]
TS = 1683554160000
q = urllib.parse.quote


def call(method, path, body=None):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode() if body is not None else None, headers=H, method=method)
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


alice, bob = f"pref:{run}:alice", f"pref:{run}:bob"
msgs = [
    {"role": "user", "timestamp": TS, "content": "Hi! Can you help me plan my weekend?\nI really love hiking in the mountains. I'm vegetarian, so keep that in mind for food."},
    {"role": "assistant", "timestamp": TS + 60000, "content": "Sure! I love that idea. Do you prefer mornings?"},
    {"role": "user", "timestamp": TS + 120000, "content": "Yes, I usually start early, around 7. Also, the weather looked nice yesterday."},
]
st, _ = call("POST", "/add", {"request_id": f"{run}:a1", "user_id": alice, "session_id": "s1", "messages": msgs})
check("add a conversation", st == 200, st)
call("POST", "/add", {"request_id": f"{run}:b1", "user_id": bob, "session_id": "s1",
                      "messages": [{"role": "user", "timestamp": TS, "content": "My favourite drink is green tea."}]})

st, body = call("GET", f"/preferences?user_id={q(alice)}")
statements = [p["statement"] for p in body["data"]]
check("the user's preference sentences are found, verbatim",
      statements == ["I really love hiking in the mountains.", "I'm vegetarian, so keep that in mind for food.", "Yes, I usually start early, around 7."], statements)
check("the assistant's words are not the user's preferences", not any("I love that idea" in s for s in statements), statements)
check("each carries its evidence, record and date",
      all(p["evidence"] == p["statement"] and p["record_id"] and p["stated_at"] and p["source"] == "rules" and p["version"] == 1 for p in body["data"]), body["data"])
check("nothing is pending with the rules", body.get("pending_records") == 0, body)
hike = next(p for p in body["data"] if "hiking" in p["statement"])

st, body = call("GET", f"/preferences?user_id={q(bob)}")
check("users see only their own preferences", [p["statement"] for p in body["data"]] == ["My favourite drink is green tea."], body)
st, _ = call("GET", f"/preferences/{hike['id']}?user_id={q(bob)}")
check("another user cannot read it (404)", st == 404, st)
st, _ = call("PATCH", f"/preferences/{hike['id']}", {"user_id": bob, "statement": "x"})
check("another user cannot change it (404)", st == 404, st)

st, body = call("POST", "/search", {"query": "ideas for an outdoor activity on Saturday", "user_id": alice, "top_k": 5})
check("search without preference slots is unchanged", st == 200 and not any(d.get("kind") == "preference" for d in body["data"]), body)
st, body = call("POST", "/search", {"query": "hiking ideas for Saturday", "user_id": alice, "top_k": 5, "preference_slots": 2})
prefs = [d for d in body["data"] if d.get("kind") == "preference"]
check("preference slots add the most relevant preferences at the end",
      st == 200 and len(body["data"]) <= 5 and len(prefs) == 2 and body["data"][-2:] == prefs and "hiking" in prefs[0]["content"], body)
check("a preference result is labelled and dated", prefs and prefs[0]["content"].startswith("[preference, stated 8 May 2023]"), prefs)

st, body = call("PATCH", f"/preferences/{hike['id']}", {"user_id": alice, "when_to_apply": "weekend and holiday plans"})
check("a correction makes a new version", st == 200 and body["data"]["version"] == 2 and body["data"]["replaces"] == hike["id"]
      and body["data"]["statement"] == hike["statement"] and body["data"]["source"] == "manual", body)
v2 = body["data"]["id"]
st, body = call("GET", f"/preferences/{q(hike['id'])}/history?user_id={q(alice)}")
check("history lists both versions, oldest first", [p["version"] for p in body["data"]] == [1, 2] and body["data"][0]["status"] == "superseded", body)
st, _ = call("PATCH", f"/preferences/{hike['id']}", {"user_id": alice, "statement": "y"})
check("a superseded version cannot be changed (409)", st == 409, st)
st, body = call("POST", "/search", {"query": "plans for the holiday", "user_id": alice, "top_k": 5, "preference_slots": 1})
check("search shows the current version with its situations", "applies when: weekend and holiday plans" in body["data"][-1]["content"], body["data"][-1])

st, body = call("POST", "/preferences", {"user_id": alice, "statement": "The user writes in British English.", "category": "communication"})
check("a preference can be stated directly", st == 200 and body["data"]["source"] == "manual" and body["data"]["evidence"] is None, body)
st, _ = call("POST", "/preferences", {"user_id": alice, "statement": "the user writes in British English"})
check("an equal current preference is refused (409)", st == 409, st)
manual = body["data"]["id"]
st, _ = call("DELETE", f"/preferences/{manual}?user_id={q(alice)}")
st2, body = call("GET", f"/preferences?user_id={q(alice)}")
check("a deleted preference is no longer current", st == 200 and manual not in [p["id"] for p in body["data"]], body)

veg = next(p for p in body["data"] if "vegetarian" in p["statement"])
st, _ = call("DELETE", f"/memories/{q(veg['record_id'], safe='')}?user_id={q(alice)}")
st2, body = call("GET", f"/preferences?user_id={q(alice)}&status=all")
retracted = {p["statement"]: p["status"] for p in body["data"] if p["record_id"] == veg["record_id"]}
check("deleting a memory retracts the preferences drawn from it (an earlier version stays superseded)",
      st == 200 and retracted.get(veg["statement"]) == "retracted" and "current" not in retracted.values(), retracted)
check("a corrected version does not stand on the memory and stays current", any(p["id"] == v2 and p["status"] == "current" for p in body["data"]), body)

st, _ = call("POST", "/add", {"request_id": f"{run}:a2", "user_id": alice, "session_id": "s2",
                               "messages": [{"role": "user", "timestamp": TS + 86400000, "content": "Line one of a note\nand I hate crowded places."}]})
st, body = call("GET", f"/preferences?user_id={q(alice)}")
check("later lines of a multiline message count", any(p["statement"] == "and I hate crowded places." for p in body["data"]), body)

st, body = call("DELETE", f"/users/{q(alice, safe='')}")
st2, body = call("GET", f"/preferences?user_id={q(alice)}&status=all")
check("deleting the user removes their preferences", st == 200 and body["data"] == [], body)

print(f"\n{len(failures)} failed" if failures else "\nall checks passed")
sys.exit(1 if failures else 0)
