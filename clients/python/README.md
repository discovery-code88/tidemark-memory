# tidemark-client

Python client for the [Tidemark](../../README.md) memory service. Standard library only.

```bash
pip install ./clients/python
```

```python
from tidemark import Tidemark

tm = Tidemark("http://127.0.0.1:18200", api_key="change-me")
tm.add("alice", "2024-05-08", [{"role": "user", "content": "I adopted a cat named Pixel yesterday."}])
hits = tm.search("alice", "What is my cat called?", top_k=5)
memory = hits[0]
tm.update("alice", memory["id"], "I adopted a cat named Pixel on 7 May 2024.")
tm.history("alice", memory["id"])        # earlier versions
tm.delete("alice", memory["id"])         # one memory
tm.delete_user("alice")                  # everything stored for alice

tm.preferences("alice")                  # what alice likes, avoids, usually does
tm.search("alice", "plan my weekend", preference_slots=3)   # with her most relevant preferences
tm.add_preference("alice", "The user writes in British English.", when_to_apply="any text for her")
```

Errors raise `TidemarkError` (with `.status` and `.message`); busy (429) and provider-unavailable (503) answers are retried.
