#!/usr/bin/env python3
"""Checks of explicit standing rules (POST/GET/PATCH/DELETE /rules, history, Search's separate `rules` field):
a rule is added on purpose, listed, changed, revoked or purged; Search returns active rules apart from its results
without taking a slot; contradicting rules both stay active and point at each other; users are isolated; deleting
the user removes their rules.

usage: rules_check.py URL KEY
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


alice, bob = f"rules:{run}:alice", f"rules:{run}:bob"
msgs = [{"role": "user", "timestamp": TS + i * 60000, "content": t} for i, t in enumerate([
    "I keep my project notes in plain text files.", "The backup runs every night at two.", "We talked about version control yesterday.",
    "My laptop is a ThinkPad.", "I usually work from the library on Fridays.", "The deadline for the report is the 31st.",
    "I'd like summaries to be short.", "My editor is Neovim.", "Snapshots go to an external disk.", "The server runs Debian.",
    "Lunch is at noon.", "I prefer tea over coffee."])]
st, _ = call("POST", "/add", {"request_id": f"{run}:a", "user_id": alice, "session_id": "s1", "messages": msgs})
check("add conversation", st == 200, st)

def search(user, **extra):
    st, body = call("POST", "/search", dict({"query": "how should I keep snapshots of my project", "user_id": user, "top_k": 5}, **extra))
    return st, body or {}

st, before = search(alice)
check("search without rules has no rules field", st == 200 and "rules" not in before, before.keys())

st, body = call("POST", "/rules", {"user_id": alice, "text": "Don't use Git", "scope": "projects"})
rule = (body or {}).get("rule", {})
check("add a rule", st == 201 and body.get("created") is True and rule.get("status") == "active" and rule.get("version") == 1, (st, body))
rid = rule.get("id", "")
st, body = call("POST", "/rules", {"user_id": alice, "text": "don't use git", "scope": "projects"})
check("adding the same rule again returns it", st == 200 and body.get("created") is False and body["rule"]["id"] == rid, (st, body))
st, body = call("POST", "/rules", {"user_id": alice, "text": "   "})
check("an empty rule is refused", st == 400, st)

st, after = search(alice)
check("search returns the rule separately", st == 200 and [r["text"] for r in after.get("rules", [])] == ["Don't use Git"], after.get("rules"))
check("the rule takes no result slot", after.get("data") == before.get("data"), "results changed")
st, off = search(alice, rules=False)
check("rules: false leaves them out", "rules" not in off, off.keys())
st, other_scope = search(alice, rule_scope="email")
check("rule_scope excludes other scopes", "rules" not in other_scope, other_scope.get("rules"))
st, same_scope = search(alice, rule_scope="Projects")
check("rule_scope keeps its own scope", len(same_scope.get("rules", [])) == 1, same_scope.get("rules"))

st, body = call("POST", "/rules", {"user_id": alice, "text": "Use Git for every snapshot"})
other = (body or {}).get("rule", {}).get("id", "")
st, listing = call("GET", f"/rules?user_id={q(alice)}")
by_id = {r["id"]: r for r in (listing or {}).get("data", [])}
check("contradicting rules both stay active", len(by_id) == 2 and all(r["status"] == "active" for r in by_id.values()), listing)
check("and point at each other", by_id.get(rid, {}).get("possible_conflicts") == [other] and by_id.get(other, {}).get("possible_conflicts") == [rid], by_id)

st, body = call("PATCH", f"/rules/{rid}", {"user_id": alice, "text": "Don't use Git; use tarball snapshots"})
check("edit makes a new version", st == 200 and body["rule"]["version"] == 2 and body["rule"]["scope"] == "projects", (st, body))
st, body = call("PATCH", f"/rules/{rid}", {"user_id": alice, "scope": ""})
check("an empty scope clears it", st == 200 and body["rule"]["scope"] is None and body["rule"]["version"] == 3, (st, body))
st, body = call("GET", f"/rules/{rid}?user_id={q(bob)}")
check("another user cannot see the rule", st == 404, st)
st, body = call("DELETE", f"/rules/{rid}?user_id={q(bob)}")
check("another user cannot revoke the rule", st == 404, st)

st, body = call("DELETE", f"/rules/{rid}?user_id={q(alice)}")
check("revoke", st == 200 and body["rule"]["status"] == "revoked" and body["rule"]["version"] == 4, (st, body))
st, body = call("DELETE", f"/rules/{rid}?user_id={q(alice)}")
check("revoking twice is a conflict", st == 409, st)
st, body = call("PATCH", f"/rules/{rid}", {"user_id": alice, "text": "x"})
check("a revoked rule cannot be edited", st == 409, st)
st, body = call("GET", f"/rules/{rid}/history?user_id={q(alice)}")
check("history keeps every version", st == 200 and [v["version"] for v in body["versions"]] == [1, 2, 3, 4], body)
st, revoked = call("GET", f"/rules?user_id={q(alice)}&status=revoked")
check("revoked rules are listed on request", [r["id"] for r in revoked["data"]] == [rid], revoked)
st, after = search(alice)
check("search returns only active rules", [r["id"] for r in after.get("rules", [])] == [other], after.get("rules"))
check("with no possible conflict left", after["rules"][0]["possible_conflicts"] == [], after.get("rules"))

st, body = call("DELETE", f"/rules/{other}?user_id={q(alice)}&purge=true")
check("purge erases a rule", st == 200 and body.get("purged") is True, (st, body))
st, body = call("GET", f"/rules/{other}/history?user_id={q(alice)}")
check("a purged rule has no history", st == 404, st)

# ---- user boundaries
st, _ = call("POST", "/add", {"request_id": f"{run}:b", "user_id": bob, "session_id": "s1", "messages": msgs})
st, body = call("POST", "/rules", {"user_id": alice, "text": "Never print API keys"})
akeys = body["rule"]["id"]
st, body = call("POST", "/rules", {"user_id": bob, "text": "Never print API keys"})
check("the same text for another user is a separate rule", st == 201 and body["rule"]["id"] != akeys, (st, body))
bkeys = body["rule"]["id"]
st, listing = call("GET", f"/rules?user_id={q(bob)}&status=all")
check("a user lists only their own rules", [r["id"] for r in listing["data"]] == [bkeys], listing)
st, _ = call("PATCH", f"/rules/{akeys}", {"user_id": bob, "text": "Print everything"})
check("another user cannot edit the rule", st == 404, st)
st, _ = call("GET", f"/rules/{akeys}/history?user_id={q(bob)}")
check("another user cannot read its history", st == 404, st)
st, _ = call("DELETE", f"/rules/{akeys}?user_id={q(bob)}&purge=true")
check("another user cannot purge the rule", st == 404, st)
st, body = call("GET", f"/rules/{akeys}?user_id={q(alice)}")
check("after those attempts the rule is unchanged", st == 200 and body["version"] == 1 and body["status"] == "active", body)
st, bsearch = search(bob)
check("search returns only the searching user's rules", [r["id"] for r in bsearch.get("rules", [])] == [bkeys], bsearch.get("rules"))
st, _ = call("DELETE", f"/users/{q(bob)}")
st, body = call("GET", f"/rules/{akeys}?user_id={q(alice)}")
check("deleting one user leaves another user's rules", st == 200 and body["status"] == "active", (st, body))

# ---- scopes and revocation in search
st, body = call("POST", "/rules", {"user_id": alice, "text": "Answer in French", "scope": "email"})
fr = body["rule"]["id"]
st, body = call("POST", "/rules", {"user_id": alice, "text": "Keep answers under 100 words", "scope": "chat"})
chat = body["rule"]["id"]
ids = lambda b: sorted(r["id"] for r in b.get("rules", []))
st, b1 = search(alice, rule_scope="email")
check("rule_scope returns that scope plus unscoped rules", ids(b1) == sorted([akeys, fr]), b1.get("rules"))
st, b2 = search(alice)
check("without rule_scope every active rule is returned", ids(b2) == sorted([akeys, fr, chat]), b2.get("rules"))
call("DELETE", f"/rules/{fr}?user_id={q(alice)}")
st, b3 = search(alice, rule_scope="email")
check("a revoked rule is not returned, even for its scope", ids(b3) == [akeys], b3.get("rules"))
st, body = call("GET", f"/rules?user_id={q(alice)}&status=bogus")
check("an unknown status is refused", st == 400, st)

# ---- conflict links are a hint, not a guarantee
st, body = call("POST", "/rules", {"user_id": alice, "text": "Always reply in English", "scope": "chat"})
en = body["rule"]["id"]
st, body = call("POST", "/rules", {"user_id": alice, "text": "Answer in French", "scope": "chat"})
fr2 = body["rule"]["id"]
st, listing = call("GET", f"/rules?user_id={q(alice)}")
conf = {r["id"]: r["possible_conflicts"] for r in listing["data"]}
check("both contradicting rules stay active", en in conf and fr2 in conf, conf)
check("a contradiction without a shared word is not linked (the hint is incomplete)", conf.get(en) == [] and conf.get(fr2) == [], conf)

st, _ = call("DELETE", f"/users/{q(alice)}")
st, listing = call("GET", f"/rules?user_id={q(alice)}&status=all")
check("deleting the user removes their rules", st == 200 and listing["data"] == [], listing)

print("\nall checks passed" if not failures else f"\n{len(failures)} failed: {failures}")
sys.exit(1 if failures else 0)
