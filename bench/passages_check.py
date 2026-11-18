#!/usr/bin/env python3
"""Checks of best-matching passages (`passages: true`): a long mixed-topic record returns the paragraph that matches
the query, with its speaker and character offsets into the complete content; short records get none; the default
response has no passage field and the record text is unchanged.

usage: passages_check.py URL KEY
Run against a server with an empty database (placeholder embeddings are fine).
"""
import json, sys, urllib.error, urllib.request, uuid

URL = sys.argv[1].rstrip("/")
KEY = sys.argv[2]
H = {"Authorization": f"Token {KEY}", "Content-Type": "application/json", "User-Agent": "tidemark-bench/1.0"}
failures = []
run = uuid.uuid4().hex[:8]
user = f"passages:{run}"


def call(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers=H, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  {detail}"))
    if not ok:
        failures.append(name)


filler = "We went over the plan for the quarter and the budget looked fine. " * 6
long_msg = (f"{filler}Separately, keep incomplete records explicitly incomplete and never reconstruct the raw reply from "
            f"the delivered response. {filler}The diagnostic flag stays off by default.")
call("/add", {"request_id": f"{run}:1", "user_id": user, "session_id": "s1",
              "messages": [{"role": "user", "timestamp": 1700000000000, "content": long_msg}]})
call("/add", {"request_id": f"{run}:2", "user_id": user, "session_id": "s2",
              "messages": [{"role": "user", "timestamp": 1700000600000, "content": "My incomplete jigsaw puzzle is on the table."}]})

query = "how should incomplete records be handled"
plain = call("/search", {"query": query, "user_id": user, "top_k": 5})["data"]
with_p = call("/search", {"query": query, "user_id": user, "top_k": 5, "passages": True})["data"]
check("default results carry no passage", all("passage" not in it for it in plain), plain)
check("passages leave ranking and content unchanged", [(i["id"], i["content"]) for i in plain] == [(i["id"], i["content"]) for i in with_p])
long_item = next((it for it in with_p if "Separately, keep incomplete" in it["content"]), None)
p = (long_item or {}).get("passage") or {}
check("the long record gets the matching passage", "keep incomplete records explicitly incomplete" in p.get("text", ""), p)
check("the passage names its speaker", p.get("speaker") == "user", p)
check("offsets point into the complete content", long_item is not None and long_item["content"][p.get("start", 0):p.get("end", 0)] == p.get("text"), p)
check("the passage is short", 0 < len(p.get("text", "")) <= 400, len(p.get("text", "")))
short_item = next((it for it in with_p if "jigsaw" in it["content"]), None)
check("a short record gets no passage", short_item is not None and "passage" not in short_item, short_item)
tight = call("/search", {"query": query, "user_id": user, "top_k": 5, "passages": True, "passage_max_chars": 60})["data"]
tp = next((it.get("passage") for it in tight if "Separately" in it["content"]), None) or {}
check("passage_max_chars limits the passage", 0 < len(tp.get("text", "")) <= 60 and "incomplete" in tp.get("text", ""), tp)

# ---- Unicode: offsets are code points into the stored content and reproduce the exact text
uni = f"unicode:{run}"
pad = "Wir haben über das Budget gesprochen 🎉 und 👩‍💻 Teams in 東京 und القاهرة verglichen. " * 6
target = "Die Entscheidung: “Café” (mit e\u0301) bleibt der Treffpunkt für das Quartal 🙂."
cjk = "我们决定每周五在图书馆开会。"
msg = f"{pad}{target} {pad}{cjk} {pad}"
call("/add", {"request_id": f"{run}:u", "user_id": uni, "session_id": "s1", "messages": [{"role": "user", "timestamp": 1700001200000, "content": msg}]})
for label, query, expect in [("accents, emoji and curly quotes", "Entscheidung Treffpunkt Quartal", target), ("CJK", "图书馆开会", cjk)]:
    res = call("/search", {"query": query, "user_id": uni, "top_k": 3, "passages": True})["data"]
    item = next((it for it in res if expect in it["content"]), None)
    p = (item or {}).get("passage") or {}
    text = p.get("text", "")
    check(f"Unicode ({label}): passage found", expect in text and (label != "CJK" or text == cjk), p)
    check(f"Unicode ({label}): content[start:end] reproduces the text exactly", item is not None and item["content"][p.get("start", 0):p.get("end", 0)] == text and len(text) > 0, p)
    check(f"Unicode ({label}): the bytes match the stored content", item is not None and text.encode() in item["content"].encode(), p)

print("\nall checks passed" if not failures else f"\n{len(failures)} failed: {failures}")
sys.exit(1 if failures else 0)
