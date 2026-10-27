# Tidemark

A small memory layer for conversational agents with two operations, **Add** and **Search**. It keeps every message verbatim, dates it, and resolves relative times when the message is written. Search ranks one user's records with BM25 and embedding similarity.

It is the open-source memory component of Unlimited Memory and implements the [Agent Memory Leaderboard](https://agentmemoryleaderboard.ai) Add/Search contract.

## Method

**Write (Add)**
- **Records of a few messages.** Messages are stored as given, grouped into records of up to `TIDEMARK_WINDOW` (default 5) consecutive messages of one Add request. A message over `TIDEMARK_MAX_WORDS` words (default 300) gets its own record(s), split into overlapping parts.
- **Date header.** Each record starts with the timestamp of its first message, e.g. `[Monday 8 May 2023, 13:56]`.
- **Time notes.** Relative time expressions are resolved against that date and appended, e.g. `[time: yesterday = 7 May 2023]`, `last Friday = the Friday before 8 May 2023`, `two years ago = 2021`, `for 3 years now = since about 2020`, `next month = June 2023`, `last summer = summer 2022`. The notes only restate what the message and its date already say. Week-, weekend- and weekday-based expressions stay relative to the message date, as answer judges such as AML's require. The notes are computed by deterministic rules (`src/dates.rs`), with no model involved.
- **Reply context.** With one message per record (`TIDEMARK_WINDOW=1`), the start of the previous message in the same session is attached, e.g. `(in reply to: Melanie: How have you…)`, so a short reply such as "Yes, last week!" keeps its meaning.
- **Durability.** Records are embedded and committed to SQLite before Add returns. A repeated `request_id` is acknowledged without storing anything twice.
- **Bounded memory.** SQLite holds all records. Only recently used users are kept in memory (their records, tokens, embeddings and their own engine instance), up to `TIDEMARK_CACHE_MB`. A user not in memory is reloaded from SQLite on their next Add or Search.

**Read (Search)**
- **Isolation.** Only the records of the requested `user_id` are considered.
- **Memory engine first.** The memory engine (below) ranks the records first; the hybrid ranking fills the remaining places up to `top_k`.
- **Hybrid ranking.** BM25 runs over the record text, including dates and time notes. Cosine similarity compares the query embedding with each message embedding. Multiple-choice `options` are appended to the query.
- **Fusion.** The two rankings are combined by reciprocal-rank fusion (`TIDEMARK_LEXICAL_WEIGHT`, `TIDEMARK_RRF_K`). At most `top_k` records are returned, best first.
- **No answers.** Search returns stored records only. It never generates answers.
- **Load limit.** Beyond `TIDEMARK_MAX_CONCURRENT` concurrent requests the service answers HTTP 429 with `Retry-After`.

No language model is called at write or read time. The only model is the embedding model.

## Memory engine (`engine/`)

`engine/` is the memory core of Unlimited Memory, released as the `tidemark-engine` library. It covers:
- ingestion, with write gating and per-memory salience, stability, utility and conflict scores;
- retrieval: lexical ranking with query intents, temporal clauses and fact handling;
- semantic core, procedural memory, date resolution, answering and verification, and person profiles.

It runs in memory without a database. The host (this service) keeps the durable write log and replays it into the engine on start. Local ONNX models (embedding index, reranker/NLI scorer) are opt-in (`SEMANTIC_RETRIEVAL=onnx`, `SEMANTIC_SCORER=onnx`) and off by default.

With `TIDEMARK_BACKEND=engine` (the default), Search puts the engine's ranking first and fills the remaining slots up to `top_k` with the hybrid BM25 + embedding ranking. Measured on LoCoMo with 5-message records (`bench/locomo_recall.py`, placeholder embeddings):

| | all@10 | all@25 | all@100 |
|---|---|---|---|
| hybrid ranking | 72.5% | 83.5% | 96.5% |
| engine first + hybrid fill | 74.1% | 85.1% | 96.6% |

Temporal questions gain the most (all@10 76.9% → 85.9%).

## API

| Endpoint | Body | Reply |
|---|---|---|
| `GET /health` | – | `{"status":"ok","service":"tidemark","version":"1.0.0"}` (no credentials needed) |
| `POST /add` | `{"request_id", "user_id", "session_id", "messages": [{"role", "timestamp" (Unix ms, optional), "content"}]}` | `{"success": true, "request_id", "user_id", "session_id"}` once the records are durable and searchable |
| `POST /search` | `{"query", "user_id", "top_k", "options" (optional)}` | `{"data": [{"id", "content", "score", "created_at"}]}`, at most `top_k` items |

`content` may be a string or a list of parts. Text parts are stored; image parts are ignored.

Responses are gzip-compressed when the client sends `Accept-Encoding: gzip`, as `httpx` and `requests` do by default.

Credentials (when `TIDEMARK_API_KEYS` is set) can be sent as `X-Api-Key: KEY`, `Authorization: Bearer KEY` or `Authorization: Token KEY`.

## Running

```sh
cargo build --release
TIDEMARK_EMBED_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1 \
TIDEMARK_EMBED_API_KEY=... \
TIDEMARK_API_KEYS=choose-a-key \
TIDEMARK_DB=tidemark.sqlite TIDEMARK_HOST=127.0.0.1 TIDEMARK_PORT=18200 MALLOC_ARENA_MAX=2 \
./target/release/tidemark
```

## Memory

On Linux with glibc, run the service with `MALLOC_ARENA_MAX=2`. In a soak test of 500 LongMemEval-S haystacks with a 512 MB cache, memory levelled off at about 580 MB with it and kept growing (past 900 MB) without it, because of allocator fragmentation.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `TIDEMARK_EMBED_URL` | – | OpenAI-compatible embeddings endpoint (base URL or full `/embeddings` URL) |
| `TIDEMARK_EMBED_API_KEY` | – | Key for it. **Without a key a hashed bag-of-words placeholder is used: for offline development only** |
| `TIDEMARK_EMBED_MODEL` | `text-embedding-v4` | Embedding model |
| `TIDEMARK_EMBED_DIMS` | model default | Requested embedding size |
| `TIDEMARK_EMBED_BATCH` | 10 | Texts per embedding request |
| `TIDEMARK_EMBED_PARALLEL` | 4 | Embedding requests of one Add sent at the same time |
| `TIDEMARK_SKIP_EMBED_CHECK` | – | Set to skip the startup check (one embedding call; the service refuses to start if it fails) |
| `TIDEMARK_API_KEYS` | – | Comma-separated accepted keys (unset = no authentication) |
| `TIDEMARK_DB` | `tidemark.sqlite` | SQLite database file |
| `TIDEMARK_HOST`, `TIDEMARK_PORT` | `127.0.0.1`, `18200` | Listen address |
| `TIDEMARK_MAX_WORDS` | 300 | Words per record before a message is split |
| `TIDEMARK_PREVIOUS_CONTEXT_CHARS` | 160 | Characters of the previous message attached as reply context (0 = off) |
| `TIDEMARK_LEXICAL_WEIGHT` | 0.5 | Share of the lexical ranking in the fusion |
| `TIDEMARK_RRF_K` | 60 | Reciprocal-rank-fusion constant |
| `TIDEMARK_BACKEND` | `engine` | `engine` = memory-engine ranking first, hybrid ranking fills the rest; `builtin` = hybrid ranking only |
| `TIDEMARK_WINDOW` | 5 | Consecutive messages of one Add request per record (non-overlapping) |
| `TIDEMARK_CACHE_MB` | 2048 | Estimated memory budget for users held in memory; least recently used users are evicted and reloaded on demand |
| `TIDEMARK_MAX_CONCURRENT` | 8 | Concurrent requests; beyond it HTTP 429 with `Retry-After: 2` |
| `TIDEMARK_RERANK_DIR`, `TIDEMARK_RERANK_CANDIDATES` | –, 0 | Optional local cross-encoder reranker (ONNX) and how many candidates it re-orders |
| `TIDEMARK_RERANK_SCOPE`, `TIDEMARK_RERANK_MAX_TOKENS` | hybrid, 512 | `all` reranks the engine's results together with the hybrid candidates (otherwise the engine's stay in front); tokens per question–record pair |
| `TIDEMARK_MAX_RESULTS` | 0 | Most results per search even if `top_k` is larger (0 = `top_k`); a search may override it with `max_results` |
| `TIDEMARK_MAX_CHARS` | 0 | Most characters of content per search; the first result is always returned (0 = no limit); a search may override it with `max_chars` |
| `TIDEMARK_USER_WEIGHT` | 1 | Weight of a second ranking over the user's own lines (`user: …`), added to the main ranking (0 = off); a search may override it with `user_weight` |
| `TIDEMARK_INSTRUCTION_SLOTS` | 10 | Slots at the end of the results for the user's standing instructions and preferences ("Always …", "I prefer …"), most relevant first (0 = off); a search may override it with `instruction_slots` |
| `TIDEMARK_ADOPT_EMBEDDINGS` | – | Set once to record the configured embedding model for a database made by an older version (otherwise it is only checked for matching dimensions and a warning is logged); a database whose recorded model or dimensions differ from the configuration is refused |
| `TIDEMARK_PART_SPEAKERS` | off | Experimental, not yet measured on AML's input: `on` repeats a named speaker in every part of a split message |
| `TIDEMARK_HEADING_WORDS` | off | Experimental: `on` stops "Reminder: …", "Note: …" and similar headings from being taken for a speaker's name |
| `TIDEMARK_USER_LINES` | first | Experimental: `continued` also counts the following lines of a multiline user message for the user ranking and instructions |
| `TIDEMARK_INSTRUCTION_DEDUP` | off | Experimental: `on` skips an instruction whose record is already among the results and gives its slot to the next record |
| `TIDEMARK_SPEAKER_BOOST` | 0 | Extra weight for messages by a person the query names (measured: no gain on LoCoMo, so off) |

## Technical report

`docs/REPORT.md` describes the method and every measurement behind the defaults, including ideas that did not help.

## Benchmarks and checks

- `bench/locomo_recall.py LOCOMO_JSON --url URL --key KEY`: replays LoCoMo through Add (20 messages per request, with session times) and reports, by category, the share of gold evidence found in the top 10, 25, 50 and 100 results.
- `bench/longmemeval_recall.py longmemeval_s_cleaned.json --url URL`: replays a sample of LongMemEval-S (one user per haystack) and reports whether the answer sessions come back in the top 10, 25, 50 and 100, by question type.
- `bench/robustness.py URL [KEY]`: checks beyond the contract. It covers text without spaces (e.g. Chinese), 6 MB requests, empty, image-only, role-less and timestamp-less messages, odd searches, concurrent Adds and Searches on one user, and isolation after eviction and reload (run with a small `TIDEMARK_CACHE_MB`).
- `bench/soak.py longmemeval_s_cleaned.json URL SERVER_PID`: AML-like load from parallel clients over many LongMemEval-S haystacks. It reports errors by status, latency percentiles and server memory over time.
- `bench/conformance.py URL KEY`: checks the Add/Search contract. It covers credentials, echoed IDs, immediate searchability, idempotent retries, `top_k`, user isolation, options, content parts, long messages and concurrent writes.

## Origin

Tidemark was written for Unlimited Memory (2026). The name: a tidemark is the line the tide leaves behind, a dated trace of what happened. It contains no third-party code beyond the Rust crates listed in `Cargo.toml`.
