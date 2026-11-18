#!/usr/bin/env python3
"""Rules and passages through the Python client (clients/python), against a running server with an empty database.

usage: client_check.py URL KEY   (import the client from the package being checked, e.g. after pip install)
"""
import sys, uuid
from tidemark import Tidemark, TidemarkError, __version__

URL, KEY = sys.argv[1], sys.argv[2]
t = Tidemark(URL, KEY, retries=0)
failures = []
run = uuid.uuid4().hex[:8]


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  {detail}"))
    if not ok:
        failures.append(name)


print("client", __version__, "from", sys.modules["tidemark"].__file__)
me, other = f"client:{run}:me", f"client:{run}:other"
long_text = ("We reviewed the quarter and the numbers were fine. " * 8 + "From now on snapshots are tarballs kept outside /tmp, "
             "because I don't want Git here. " + "Then we talked about lunch plans and the weather. " * 8)
t.add(me, "s1", [{"role": "user", "content": long_text, "timestamp": 1700000000000}])
t.add(other, "s1", [{"role": "user", "content": "Snapshots are fine either way.", "timestamp": 1700000000000}])

r = t.add_rule(me, "Don't use Git", scope="projects")
check("add_rule", r["status"] == "active" and r["version"] == 1, r)
check("rules lists it", [x["id"] for x in t.rules(me)] == [r["id"]])
check("get_rule", t.get_rule(me, r["id"])["text"] == "Don't use Git")
try:
    t.get_rule(other, r["id"]); check("another user cannot get it", False)
except TidemarkError as e:
    check("another user cannot get it", e.status == 404, e.status)
res = t.search_with_rules(me, "how are snapshots kept", top_k=3)
check("search_with_rules returns rules apart from results", [x["text"] for x in res["rules"]] == ["Don't use Git"] and len(res["data"]) >= 1, res["rules"])
check("rule_scope filters", t.search_with_rules(me, "snapshots", rule_scope="email")["rules"] == [])
check("search_with_rules for another user has no rules", t.search_with_rules(other, "snapshots")["rules"] == [])
check("plain search still returns a list", isinstance(t.search(me, "snapshots", top_k=3), list))
c = t.add_rule(me, "Use Git for every snapshot")
check("a contradicting rule is kept and linked", t.get_rule(me, r["id"])["possible_conflicts"] == [c["id"]])
u = t.update_rule(me, r["id"], text="Don't use Git; use tarballs")
check("update_rule makes a new version", u["version"] == 2 and u["scope"] == "projects", u)
v = t.revoke_rule(me, r["id"])
check("revoke_rule", v["status"] == "revoked", v)
check("rule_history", [x["version"] for x in t.rule_history(me, r["id"])] == [1, 2, 3])
check("revoked rules on request", [x["id"] for x in t.rules(me, "revoked")] == [r["id"]])
check("purge_rule", t.purge_rule(me, c["id"]).get("purged") is True)

items = t.search(me, "how are snapshots kept without Git", top_k=3, passages=True)
item = next((i for i in items if "tarballs" in i["content"]), None)
p = (item or {}).get("passage") or {}
check("passages through search(..., passages=True)", "tarballs" in p.get("text", ""), p)
check("content[start:end] == passage text", item is not None and item["content"][p["start"]:p["end"]] == p["text"], p)
check("no passage unless asked", all("passage" not in i for i in t.search(me, "snapshots", top_k=3)))

t.delete_user(me); t.delete_user(other)
check("delete_user removes rules", t.rules(me, "all") == [])
print("\nall checks passed" if not failures else f"\n{len(failures)} failed: {failures}")
sys.exit(1 if failures else 0)
