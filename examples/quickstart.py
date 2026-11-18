#!/usr/bin/env python3
"""Tidemark in two minutes: store a conversation, search it, correct and delete.

usage: python examples/quickstart.py [URL] [API_KEY]
(start the service first, e.g. `docker compose up -d`; see README.md)
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "clients", "python"))
from tidemark import Tidemark  # noqa: E402

url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18200"
key = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("TIDEMARK_API_KEY", "change-me")
tm = Tidemark(url, api_key=key)
user = "quickstart-alice"
tm.delete_user(user)  # start clean

may_8 = 1683554160000  # 8 May 2023, 13:56 UTC, in milliseconds
tm.add(user, "session-1", [
    {"role": "user", "timestamp": may_8, "content": "Alice: I adopted a cat named Pixel yesterday!"},
    {"role": "assistant", "timestamp": may_8, "content": "Congratulations! How is Pixel settling in?"},
    {"role": "user", "timestamp": may_8, "content": "Alice: Always call her Pixel, never 'the cat', please."},
])

print("Search: 'When did I adopt my cat?'")
hits = tm.search(user, "When did I adopt my cat?", top_k=3)
for h in hits:
    print(f"  {h['score']:.3f}  {h['content'][:120]}")
# The record shows the message's date and resolves "yesterday" to 7 May 2023.

memory = tm.list(user)["data"][0]
tm.update(user, memory["id"], memory["content"].replace("Pixel", "Pixel Moon"))
print("\nAfter a correction:", tm.get(user, memory["id"])["content"][:120])
print("History:", [(v["version"], v["action"]) for v in tm.history(user, memory["id"])])

print("\nDelete the user:", tm.delete_user(user))
print("Memories left:", tm.list(user)["total"])
