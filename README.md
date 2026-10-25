# Tidemark

Tidemark is an open-source memory layer for conversational agents with two operations, **Add** and **Search**. It is the open-source memory component of Unlimited Memory and implements the [Agent Memory Leaderboard](https://agentmemoryleaderboard.ai) Add/Search contract.

> **Status:** the source code is published with the `v1.0` release tag, before evaluation begins. This repository currently holds the description and license only.

## Method

**Add**
- Messages are stored verbatim in a durable write log, grouped into records of a few consecutive messages.
- Each record carries the date of its messages.
- Relative time expressions are resolved against that date by deterministic rules, without a language model, and written next to the message. Examples: `yesterday = 7 May 2023`, `last week = the week before 8 May 2023`, `two years ago = 2021`.
- Add returns only once the records are durable and searchable. A repeated `request_id` is acknowledged without storing anything twice.

**Search**
- Only the records of the requested `user_id` are considered.
- The Tidemark memory engine (the memory core of Unlimited Memory: write gating, per-memory salience and conflict scores, lexical ranking with query intents and temporal clauses) ranks the records first.
- The remaining places up to `top_k` are filled by BM25 combined with `text-embedding-v4` similarity through reciprocal-rank fusion.
- Search returns stored records only. It never generates answers.

No language model is called at Add or Search time. The only model is the embedding model, `text-embedding-v4`.

## API

| Endpoint | Body | Reply |
|---|---|---|
| `GET /health` | – | `{"status":"ok"}` |
| `POST /add` | `{"request_id", "user_id", "session_id", "messages": [{"role", "timestamp", "content"}]}` | `{"success": true, "request_id", "user_id", "session_id"}` |
| `POST /search` | `{"query", "user_id", "top_k", "options"?}` | `{"data": [{"id", "content", "score", "created_at"}]}` |

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
