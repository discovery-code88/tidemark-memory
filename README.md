# Tidemark

[![CI](https://github.com/discovery-code88/tidemark-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/discovery-code88/tidemark-memory/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**A self-hosted memory service for AI assistants and agents.** Tidemark stores each user's conversations verbatim,
dates every record and resolves relative times ("yesterday", "last summer") when they are written, and returns the
records most relevant to a question. It never writes answers itself, so your own model stays in charge, and it uses
no LLM at write or search time: storage and ranking are deterministic and cheap.

- **Memory engine underneath:** write gating, meta-memory scores, query intents, temporal relations and latest-state
  handling, from the memory core of Unlimited Memory (`engine/`, ~64,000 lines of Rust).
- **Hybrid retrieval:** the engine's ranking plus BM25 and embedding similarity; optionally, lines that state
  standing instructions or preferences kept in view (`TIDEMARK_INSTRUCTION_SLOTS`, off by default since 1.2).
- **Complete memory API:** add, search, list, get, correct (with history), delete a memory, delete a user.
- **Operational:** durable writes (SQLite), safe retries, per-user isolation, bounded memory, embedding-model checks,
  gzip, Docker image, Python client, 300+ tests in CI.

It implements the [Agent Memory Leaderboard](https://agentmemoryleaderboard.ai) Add/Search contract; the version
evaluated there is tagged [v1.0](https://github.com/discovery-code88/tidemark-memory/tree/v1.0).
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how it works and [docs/REPORT.md](docs/REPORT.md) for every
measurement.

## Quickstart

```bash
git clone https://github.com/discovery-code88/tidemark-memory && cd tidemark-memory
cp .env.example .env              # set TIDEMARK_API_KEYS and your embedding endpoint and key
docker compose up -d              # serves http://127.0.0.1:18200, data in a Docker volume
pip install ./clients/python
python examples/quickstart.py http://127.0.0.1:18200 <your-api-key>
```

```python
from tidemark import Tidemark

tm = Tidemark("http://127.0.0.1:18200", api_key="change-me")
tm.add("alice", "s1", [{"role": "user", "timestamp": 1683554160000,
                        "content": "Alice: I adopted a cat named Pixel yesterday!"}])
for hit in tm.search("alice", "When did I adopt my cat?", top_k=5):
    print(hit["score"], hit["content"])   # "... [time: yesterday = 7 May 2023]"
```

Without an embedding key, Tidemark runs with offline placeholder embeddings (for development only). Any embedding
provider works: see [Embedding providers](#embedding-providers).
More: [examples/curl.md](examples/curl.md), [clients/python](clients/python/README.md).

## Method

**Write (Add)**
- **Records of a few messages.** Messages are stored as given, grouped into records of up to `TIDEMARK_WINDOW` (default 5) consecutive messages of one Add request. A message over `TIDEMARK_MAX_WORDS` words (default 300) gets its own record(s), split into overlapping parts.
- **Date header.** Each record starts with the timestamp of its first message, e.g. `[Monday 8 May 2023, 13:56]`, in UTC. A message may carry `utc_offset_minutes` (the sender's offset from UTC at that moment, e.g. `120` for UTC+02:00, between -720 and 840). The header then shows the sender's local time and names the offset, `[Thursday 1 January 2026, 00:30 UTC+02:00]`, and time notes use the sender's calendar day, so "yesterday" said at 00:30 local time is the local day before. `created_at` stays the UTC instant. A fixed offset is not a time zone: it holds for that message only and cannot tell when daylight-saving time starts or ends. Without the field everything is in UTC, as in 1.0.0; records already stored are not rewritten.
- **Time notes.** Relative time expressions are resolved against that date and appended, e.g. `[time: yesterday = 7 May 2023]`, `last Friday = the Friday before 8 May 2023`, `two years ago = 2021`, `for 3 years now = since about 2020`, `next month = June 2023`, `last summer = summer 2022`. The notes only restate what the message and its date already say. A duration gets a start date only when its sentence clearly looks back ("I've been attending for three months", "for 3 years now"); a bare "for three months", which can also look forward ("I've frozen my membership for three months"), and future or hypothetical wording ("I'll have been … for") get no note. A relative expression inside quotation marks gets a note only when the quotation is what the user says at the time of the message ("I told her today, 'I arrived yesterday'"); fiction, lyrics, other people's words, undated or recalled quotations ("Today I remembered saying, '…yesterday…'") and quotations inside quotations get none, while expressions outside the quotation are resolved as usual. Apostrophes in contractions and possessives are not quotation marks. This guards against unsupported notes; it does not interpret reported speech, and speech reported without quotation marks is resolved as before. Both rules differ from 1.0.0 and apply to records added from 1.1 on; notes already stored are not rewritten. Week-, weekend- and weekday-based expressions stay relative to the message date, as answer judges such as AML's require. The notes are computed by deterministic rules (`src/dates.rs`), with no model involved.
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

With `TIDEMARK_BACKEND=engine` (the default), Search puts the engine's ranking first and fills the remaining slots up to `top_k` with the hybrid BM25 + embedding ranking. The engine's hits can fill all `top_k` slots, leaving the hybrid ranking no influence (seen on every case of an author-built development suite at `top_k` 10; see docs/ARCHITECTURE.md).

**Retrieval measurements (November 2026).** These are retrieval-only measurements. Both backends use the same binary, the same text-embedding-v4 embeddings and the same settings except `TIDEMARK_BACKEND`, with `top_k` 10.

| Evidence found in the top 10 | engine first + hybrid fill (default) | hybrid ranking only (`builtin`) |
|---|---|---|
| LoCoMo (10 conversations, categories 1–4): evidence turns | 66.4% | 77.0% |
| LoCoMo (all 1,977 questions with evidence): every evidence turn | 73.2% | 81.9% |
| LongMemEval-S (54 questions): turns that contain the answer | 69.0% | 88.5% |

Hybrid-only was no slower and returned no more text.

Hybrid-only also loses some evidence that the engine finds. On LoCoMo it lost 99 evidence turns and gained 374. The losses are mostly multi-hop questions that ask for a list ("What items has … bought?").

At `top_k` 25 the two backends are close: +1.3 points for hybrid-only.

Limitations:
- Both datasets were used during development.
- The LongMemEval sample is small (6 of 60 planned questions were not run).

An earlier LoCoMo measurement with placeholder embeddings put engine first slightly ahead (all evidence in the top 10: 74.1% vs 72.5%). That result does not hold with real embeddings.

**Better retrieval has not yet been shown to give better answers.** On an author-created holdout the answer comparison was inconclusive, and in some cases added evidence led to worse recommendations. The default therefore stays engine first, and hybrid-only remains a candidate under evaluation. Details are in docs/ARCHITECTURE.md.

## API

| Endpoint | Body | Reply |
|---|---|---|
| `GET /health` | – | `{"status":"ok","service":"tidemark","version":"1.2.0"}` (no credentials needed) |
| `POST /add` | `{"request_id", "user_id", "session_id", "messages": [{"role", "timestamp" (Unix ms, optional), "utc_offset_minutes" (optional), "content"}]}` | `{"success": true, "request_id", "user_id", "session_id"}` once the records are durable and searchable |
| `POST /search` | `{"query", "user_id", "top_k", "options" (optional), "rules"? (bool), "rule_scope"?, "passages"? (bool)}` | `{"data": [{"id", "content", "score", "created_at", "passage"?}]}`, at most `top_k` items (`passage`: see [Best-matching passages](#best-matching-passages)); plus `"rules": [...]` (the user's active [standing rules](#standing-rules), never counted in `top_k`) when there are any |
| `GET /memories?user_id=…` | optional `session_id`, `limit` (≤ 1000, default 100), `offset` | `{"data": [{"id", "session_id", "request_id", "content", "created_at"}], "total", "limit", "offset"}` in storage order |
| `GET /memories/{id}?user_id=…` | – | `{"data": {…}}`; 404 if the memory is not this user's |
| `PATCH /memories/{id}` | `{"user_id", "content"}` | the corrected memory; it is re-embedded and its previous text kept in its history |
| `DELETE /memories/{id}?user_id=…` | – | `{"success": true, "deleted": id}` |
| `GET /memories/{id}/history?user_id=…` | – | `{"data": [{"version", "action", "content", "changed_at"}]}` (earlier texts; the last text of a deleted memory) |
| `DELETE /users/{user_id}` | – | `{"success": true, "deleted_memories": n}`; removes the user's memories, request log, history, preferences and rules |
| `GET /preferences?user_id=…` | optional `status=all` | `{"data": [{"id", "statement", "when_to_apply", "category", "kind", "evidence", "record_id", "source", "status", "version", "replaces", "stated_at", "updated_at"}], "pending_records"}`; see [Preference memory](#preference-memory) |
| `POST /preferences` | `{"user_id", "statement", "when_to_apply"?, "category"?, "kind"?}` | the stored preference (`source: "manual"`); 409 if an equal one is current |
| `GET /preferences/{id}?user_id=…` | – | `{"data": {…}}` |
| `PATCH /preferences/{id}` | `{"user_id", "statement"?, "when_to_apply"?, "category"?}` | the new version; the old one becomes `superseded` |
| `DELETE /preferences/{id}?user_id=…` | – | `{"success": true, "deleted": id}`; kept in its history as `deleted` |
| `GET /preferences/{id}/history?user_id=…` | – | every version, oldest first |
| `POST /rules` | `{"user_id", "text", "scope"?}` | 201 `{"created": true, "rule": {"id", "text", "scope", "status", "version", "created_at", "updated_at", "possible_conflicts"}}` (other active rules that may contradict it; a hint, may be incomplete); an identical active rule: 200 with `"created": false` |
| `GET /rules?user_id=…` | optional `status=active` (default), `revoked`, `all` | `{"data": [rule, …], "total"}` in the order added |
| `GET /rules/{id}?user_id=…` | – | the rule; 404 if it is not this user's |
| `PATCH /rules/{id}` | `{"user_id", "text"?, "scope"?}` (`"scope": ""` clears it) | `{"rule": …}`, a new version; 409 if the rule is revoked |
| `DELETE /rules/{id}?user_id=…` | optional `purge=true` | revokes the rule (kept with its history); with `purge=true` erases it and its history |
| `GET /rules/{id}/history?user_id=…` | – | `{"versions": [...]}`, oldest first |

Memory IDs contain `#`; URL-encode them in paths (`%23`). Every request names its `user_id`, and a memory is only
visible to and changeable by its own user.

`content` may be a string or a list of parts. Text parts are stored; image parts are ignored.

Responses are gzip-compressed when the client sends `Accept-Encoding: gzip`, as `httpx` and `requests` do by default.

Credentials (when `TIDEMARK_API_KEYS` is set) can be sent as `X-Api-Key: KEY`, `Authorization: Bearer KEY` or `Authorization: Token KEY`.

## Standing rules

Rules are what a user tells Tidemark to follow, said on purpose rather than guessed from conversation text:
"Don't use Git", "Never print API keys", "Answer in French" (scope `email`). They are managed only through
`/rules`; Add, Search and preference extraction never create, change or revoke one.

- **Add** `POST /rules` with a `text` and an optional free-text `scope`. Adding an identical active rule returns it.
- **View, change, revoke:** `GET /rules`, `PATCH /rules/{id}` (each change is a new version), `DELETE /rules/{id}`
  (revoked; history kept) or `DELETE /rules/{id}?purge=true` (erased).
- **Search** returns the user's active rules in a separate `rules` field next to `data`. They never take one of the
  `top_k` places. `rules: false` leaves them out; `rule_scope: "email"` returns rules with that scope plus unscoped
  ones. Users without rules get exactly the response they got before (no `rules` field).
  `TIDEMARK_SEARCH_RULES=off` turns the field off for every search.
- **Conflicts are shown, not resolved.** Adding a rule never replaces another. Rules that may contradict each other
  stay active and name each other under `possible_conflicts`. **These links are a hint and may be incomplete;** an
  empty list does not mean a rule conflicts with nothing. A rule is flagged against another when their scopes
  overlap (equal, or either unscoped), they share a content word, and exactly one is a prohibition ("don't",
  "never", "avoid", …). That check only points; deciding which rule holds is left to the user (edit or revoke one) or
  to the application. It can miss contradictions worded without a shared word, and can flag rules that don't
  really clash.
- Scopes are labels, not a hierarchy: `rule_scope` matches a scope exactly (ignoring case and spacing).

Rules are separate from the instruction slots (`TIDEMARK_INSTRUCTION_SLOTS`, off by default), which guess
instructions from conversation wording, and from preference memory, which stores tastes and habits.

## Best-matching passages

A record can be long: a whole exchange, or a message covering several topics. With `passages: true` on a search (or
`TIDEMARK_PASSAGES=on`), each long record in the results also carries the part that matches the question:

```json
{"id": "…", "content": "<the complete record>", "score": 0.9,
 "passage": {"text": "Keep incomplete records explicitly incomplete; don't infer the raw reply …",
             "speaker": "user", "start": 1184, "end": 1346, "matched_terms": ["incomplete", "record"]}}
```

- **The record is authoritative; the passage is a reading aid.** It is chosen by words, so it can miss a paraphrase
  or leave out a qualification stated elsewhere in the record; read `content` when it matters.
- `content` stays complete; `start`/`end` are character offsets (Unicode code points; in JavaScript use
  `Array.from(content).slice(start, end).join("")`, not `content.slice`, which counts UTF-16 units) into it, and
  `GET /memories/{id}` returns the same record. Ranking and `content` are unchanged by this option.
- The passage is the run of consecutive sentences within one message (never across speakers) that covers the
  question's words best, rarer words counting more, up to `TIDEMARK_PASSAGE_MAX_CHARS` (default 400; per search
  `passage_max_chars`). A single longer sentence is cut around its first matching word.
- Records under `TIDEMARK_PASSAGE_MIN_WORDS` words (default 120; per search `passage_min_words`; each CJK character
  counts as a word) get no passage: they are short enough to read whole. Sentences end at . ! ? before a space, at
  the CJK marks 。！？ and at line breaks.
- **Limits:** the choice is by words, not meaning: no embedding call is made (no cost with paid providers), and a
  record that matches only by meaning gets no passage, or a passage that is not the answer. Inflected forms are
  joined where the light stemmer splits them ("settle"/"settled", "cap"/"capped"); this also joins a few unrelated
  pairs (plane/plan). Passages help read a returned record; they do not change which records are returned.

## Preference memory

Tidemark keeps what each user likes, avoids, habitually does or wants as short statements, apart from the
conversation records, so they are found when they matter even if the question uses other words ("plan my weekend"
→ "I really love hiking in the mountains."). Each preference has:

- a **statement**, and **when it applies** (with a language model);
- a **kind**: `likes` (a taste), `does` (something the user already does: a hobby, routine or tradition) or
  `instruction`. Search shows every preference as "[preference]" by default; with `TIDEMARK_PREFERENCE_LABELS=kind`
  what the user already does is labelled as such, and with `novelty` also as "not new to them" for a request for
  something new. Neither option has shown a benefit so far (see [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md));
- **evidence**: the user's own words, verbatim, and the record they come from;
- a **version** and **status**: `current`, `superseded` (a newer version replaces it), `retracted` (its memory was
  deleted or corrected), `deleted`. Nothing is overwritten; `GET /preferences/{id}/history` shows every version.

Extraction (`TIDEMARK_PREFERENCES`):

| Mode | How | Cost |
|---|---|---|
| `off` (default) | None are extracted; preferences given through the API are still stored and searchable. | – |
| `rules` | Sentences of the user's messages (every line, also of multiline messages) with a preference cue: "I love / enjoy / hate / usually …", "my favourite …", "I'm vegetarian / allergic …", standing instructions ("Always …", "From now on …"). The sentence is the statement and its own evidence. Inside Add. | A few short extra embedding inputs |
| `llm` | The language model (`TIDEMARK_LLM_*`) reads the user's messages after Add returns, from a durable queue (resumed after a restart), batched per user. It proposes statements, when they apply, a category, an evidence quote and which earlier preference a new one replaces. Code keeps a proposal only if its quote is the user's own words in that record, and a replacement only if it names a current preference. A batch the model keeps failing on falls back to the rules. | One model call per batch of new user messages |

Only messages with role `user` are read; assistant messages never become preferences. With `llm`, the model sees the
user's most related current preferences, so it can mark changes ("I switched to cycling" replaces "loves running");
its prompt can be replaced (`TIDEMARK_PREFERENCE_PROMPT_FILE`, reply format `{"preferences": [{"record", "statement",
"when_to_apply", "category", "evidence", "replaces"}]}`). Preferences can also be stated, corrected and deleted
through the API; a correction becomes a new version with `source: "manual"`.

Search adds the most relevant current preferences in `preference_slots` final slots (`TIDEMARK_PREFERENCE_SLOTS`,
default 0 = none, so search results are unchanged unless asked), each marked `"kind": "preference"`:

```text
[preference, stated 8 May 2023] The user enjoys hiking in the mountains.
applies when: weekend and holiday plans
user's words: "I really love hiking in the mountains."
```

**Measured status:** on PersonaMem 32k neither extraction mode has shown an overall gain so far, which is why preference
extraction is off by default; see [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

The design follows the preference points of [M-flow](https://github.com/FlowElement-xinliuyuansu/m_flow)
(Apache-2.0); Tidemark's implementation is independent.

## Running

With Docker: `docker compose up -d` (see Quickstart; settings in `.env`). From source:

```sh
cargo build --release
TIDEMARK_EMBED_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1 \
TIDEMARK_EMBED_API_KEY=... \
TIDEMARK_API_KEYS=choose-a-key \
TIDEMARK_DB=tidemark.sqlite TIDEMARK_HOST=127.0.0.1 TIDEMARK_PORT=18200 MALLOC_ARENA_MAX=2 \
./target/release/tidemark
```

## Models

Embeddings are required; reranking and a language model are optional. Add and Search never call a language model
(preference extraction with a model runs after Add returns).

### Embedding providers

Any embedding model can be used. Set `TIDEMARK_EMBED_PROTOCOL` and the provider's settings:

| Provider | Settings |
|---|---|
| OpenAI, DashScope, SiliconFlow, vLLM, LM Studio, LocalAI, … (OpenAI-compatible) | `URL`, `API_KEY`, `MODEL` (`PROTOCOL=openai` for a server without a key) |
| Azure OpenAI | `PROTOCOL=azure`, `URL=https://NAME.openai.azure.com`, `API_KEY`, `MODEL=deployment` |
| Google Gemini | `PROTOCOL=gemini`, `API_KEY`, `MODEL=gemini-embedding-001` |
| Cohere | `PROTOCOL=cohere`, `API_KEY`, `MODEL=embed-v4.0` |
| Voyage AI | `PROTOCOL=voyage`, `API_KEY`, `MODEL=voyage-3.5` |
| Jina AI | `PROTOCOL=jina`, `API_KEY`, `MODEL=jina-embeddings-v3` |
| Ollama | `PROTOCOL=ollama`, `MODEL=nomic-embed-text` (URL defaults to `http://127.0.0.1:11434`) |
| Hugging Face Text Embeddings Inference | `PROTOCOL=tei`, `URL=http://host:8080` |
| Anything else with a JSON API | `PROTOCOL=template`, `URL`, `TEMPLATE`, `TEMPLATE_VECTORS` |

(All names prefixed with `TIDEMARK_EMBED_`.) Providers that distinguish questions from stored texts are told which is
which (see `QUERY_TYPE` / `DOCUMENT_TYPE`). A template is a JSON body in which `"{{inputs}}"` becomes the list of
texts and `{{model}}`, `{{input_type}}`, `{{dims}}` are filled in, for example:

```sh
TIDEMARK_EMBED_PROTOCOL=template
TIDEMARK_EMBED_URL=https://embed.example.com/v2/vectors
TIDEMARK_EMBED_TEMPLATE='{"model": "{{model}}", "texts": "{{inputs}}", "mode": "{{input_type}}"}'
TIDEMARK_EMBED_TEMPLATE_VECTORS=result.vectors      # or data.*.embedding, embeddings, …
TIDEMARK_EMBED_TEMPLATE_TOKENS=usage.tokens         # optional
```

#### Local embeddings (no provider, no key, nothing billed)

`TIDEMARK_EMBED_PROTOCOL=local` runs a sentence-embedding model on this machine with ONNX Runtime. No text leaves the
machine. It is an alternative to a provider, not a default: without any embedding setting Tidemark still uses the
offline placeholder.

```sh
TIDEMARK_EMBED_PROTOCOL=local
TIDEMARK_EMBED_DIR=/path/to/multilingual-e5-small   # contains model.onnx and tokenizer.json
# optional:
TIDEMARK_EMBED_THREADS=6          # default: all available cores
TIDEMARK_EMBED_MAX_TOKENS=512     # model input limit; longer text is cut for the vector (BM25 still sees all of it)
TIDEMARK_EMBED_QUERY_PREFIX=…     # default "query: " / "passage: " when the directory name contains "e5", else none
TIDEMARK_EMBED_DOCUMENT_PREFIX=…
```

- **Model tested:** `intfloat/multilingual-e5-small` exported to ONNX (384 dimensions, `model.onnx` 118 MB,
  `tokenizer.json` 17 MB). Other BERT-style sentence-embedding models with the same two files should load (mean
  pooling, normalized vectors), but only this one was tested.
- **CPU cost, measured on one machine** (Intel i5-12490F, 6 cores, no GPU): adding 3,379 conversation messages
  (1,609 records) took 2 min 56 s; adding LoCoMo's 5,882 turns took about 1 minute. Searches including the query's
  embedding: p50 8–11 ms on LoCoMo. Embeddings are computed one batch at a time per process, so heavy concurrent Add
  traffic queues. Memory use was not measured separately.
- **Quality:** on LoCoMo the local model with hybrid-only search (`TIDEMARK_BACKEND=builtin`) put all evidence in the
  first 10 for 81.4 % of questions (engine-first 74.3 %). Retrieval only; not compared with paid providers on the
  same data, and not measured on answers.
- The model name is recorded in the database like any provider's; moving an existing database to or from it uses
  `tidemark reembed`.

### Reranking

Optional: a cross-encoder re-orders the leading `TIDEMARK_RERANK_CANDIDATES` search candidates (off by default; measure
it on your own data before turning it on). It can run locally
(`TIDEMARK_RERANK_DIR`, ONNX) or as an API:

| Provider | Settings |
|---|---|
| Cohere; vLLM, SiliconFlow, Xinference (Cohere-compatible `/rerank`) | `PROTOCOL=cohere`, `API_KEY`, `MODEL=rerank-v3.5` (and `URL` for the others) |
| Jina AI | `PROTOCOL=jina`, `API_KEY`, `MODEL=jina-reranker-v2-base-multilingual` |
| Voyage AI | `PROTOCOL=voyage`, `API_KEY`, `MODEL=rerank-2.5` |
| Hugging Face Text Embeddings Inference | `PROTOCOL=tei`, `URL=http://host:8080` |
| Alibaba DashScope | `PROTOCOL=dashscope`, `API_KEY`, `MODEL=gte-rerank-v2` (international: `URL=https://dashscope-intl.aliyuncs.com/api/v1`) |
| Anything else | `PROTOCOL=template`, `URL`, `TEMPLATE` with `"{{query}}"`, `"{{documents}}"`, `{{model}}`, `{{top_n}}`; `TEMPLATE_RESULTS` (a list of objects with `TEMPLATE_INDEX` and `TEMPLATE_SCORE`, or of plain scores in document order) |

(All names prefixed with `TIDEMARK_RERANK_`.) A reranking API is called once at start; the service refuses to start
if the call fails. If a reranking call fails during a search, the search keeps its order.

### Language model

Optional, for features that need one (`TIDEMARK_LLM_*`); Add and Search work without it. Any chat API works:

| Provider | Settings |
|---|---|
| OpenAI and compatible (vLLM, Ollama, LM Studio, OpenRouter, DashScope, DeepSeek, …) | `URL` (e.g. `https://api.openai.com/v1`), `API_KEY`, `MODEL` |
| Anthropic | `PROTOCOL=anthropic`, `URL=https://api.anthropic.com/v1`, `API_KEY`, `MODEL` |
| Google Gemini | `PROTOCOL=gemini`, `URL=https://generativelanguage.googleapis.com/v1beta`, `API_KEY`, `MODEL` |
| Any other JSON API | `PROTOCOL=custom`, `URL`, `REQUEST_TEMPLATE` (with `{{system}}`, `{{user}}`, `{{model}}`, `"{{max_tokens}}"`, `"{{temperature}}"`), `RESPONSE_PATH` (JSON pointer to the reply text) |
| A local command-line tool (`llama-cli`, `ollama run`, Codex CLI, …) | `PROTOCOL=command`, `COMMAND` (prompt on stdin, reply on stdout or in `{{output_file}}`; no shell involved) |

(All names prefixed with `TIDEMARK_LLM_`.) Replies that must be JSON use the API's JSON mode where it has one and
fall back to asking for JSON in the instructions where it is rejected. `HEADERS` adds request headers (a header
containing `{{api_key}}` replaces the default key header, e.g. Azure's `api-key`). Token counts are read from every
built-in API's reply; for custom APIs and commands, `USAGE_INPUT_PATH` / `USAGE_OUTPUT_PATH` (JSON pointers) or
`USAGE_REGEX` say where they are. The model is called once at start (`TIDEMARK_SKIP_LLM_CHECK=1` skips it).

### Changing the embedding model

A database records the model its vectors come from (with document type and prefix), and the
service refuses to start with another one. To switch, stop the service and run with the new settings:

```sh
tidemark reembed --estimate   # records and characters to embed; no provider call
tidemark reembed              # re-embeds every record; resumes if interrupted
tidemark reembed --abort      # drops an unfinished run; the stored vectors are unchanged
```

New vectors are staged and swapped in, with the recorded model, in one transaction once every record has one; until
then the old vectors stay in place and the service refuses to start. With Docker:
`docker compose run --rm tidemark tidemark reembed`.

## Memory

On Linux with glibc, run the service with `MALLOC_ARENA_MAX=2`. In a soak test of 500 LongMemEval-S haystacks with a 512 MB cache, memory levelled off at about 580 MB with it and kept growing (past 900 MB) without it, because of allocator fragmentation.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `TIDEMARK_EMBED_PROTOCOL` | `openai` if URL and key are set | Embedding API: `openai` (any OpenAI-compatible endpoint), `azure`, `gemini`, `cohere`, `voyage`, `jina`, `ollama`, `tei`, `template`, `hash`, or `local`. See [Embedding providers](#embedding-providers) |
| `TIDEMARK_EMBED_DIR` | – | For `local`: a directory with `model.onnx` and `tokenizer.json` (e.g. multilingual-e5-small exported to ONNX). Runs on this machine; nothing is sent or billed. A directory name containing "e5" gets the "query: " / "passage: " prefixes. Text beyond `TIDEMARK_EMBED_MAX_TOKENS` (default 512) does not reach the model (BM25 still sees it). CPU cost: about 3 minutes to add 3,400 messages on this machine. Tested on retrieval only, not on answers |
| `TIDEMARK_EMBED_URL` | per protocol | Endpoint (base URL; for `openai` also the full `/embeddings` URL). Required for `azure`, `tei`, `template` |
| `TIDEMARK_EMBED_API_KEY` | – | Key. **With neither a protocol nor URL and key, a hashed bag-of-words placeholder is used: for offline development only** |
| `TIDEMARK_EMBED_MODEL` | `text-embedding-v4` for `openai` | Embedding model (Azure: deployment name) |
| `TIDEMARK_EMBED_DIMS` | model default | Requested embedding size |
| `TIDEMARK_EMBED_QUERY_TYPE`, `TIDEMARK_EMBED_DOCUMENT_TYPE` | per protocol | Input type sent for search questions and for stored texts (Cohere `search_query`/`search_document`, Voyage `query`/`document`, Jina `retrieval.query`/`retrieval.passage`, Gemini `RETRIEVAL_QUERY`/`RETRIEVAL_DOCUMENT`); `none` sends none |
| `TIDEMARK_EMBED_QUERY_PREFIX`, `TIDEMARK_EMBED_DOCUMENT_PREFIX` | – | Text put before questions / stored texts, for models trained with prefixes (E5: `query: ` / `passage: `; nomic: `search_query: ` / `search_document: `) |
| `TIDEMARK_EMBED_HEADERS` | – | Extra request headers, `Name: value` separated by `;` |
| `TIDEMARK_EMBED_AUTH` | per protocol | How the key is sent: `bearer`, `none` or `header:Name` (Azure `api-key`, Gemini `x-goog-api-key`, others bearer) |
| `TIDEMARK_EMBED_API_VERSION` | `2024-10-21` | Azure API version |
| `TIDEMARK_EMBED_TEMPLATE`, `TIDEMARK_EMBED_TEMPLATE_VECTORS`, `TIDEMARK_EMBED_TEMPLATE_TOKENS` | –, `data.*.embedding`, – | `template` protocol: JSON request body, and paths to the vectors and the token count in the reply |
| `TIDEMARK_EMBED_MAX_CHARS` | 6000 | Characters of a record embedded (BM25 always sees all of it) |
| `TIDEMARK_EMBED_BATCH` | per protocol (10 for `openai`) | Texts per embedding request |
| `TIDEMARK_EMBED_PARALLEL` | 4 | Embedding requests of one Add sent at the same time |
| `TIDEMARK_REEMBED_CHUNK` | 200 | `tidemark reembed`: records embedded and saved per step |
| `TIDEMARK_SKIP_EMBED_CHECK` | – | Set to skip the startup check (one embedding call; the service refuses to start if it fails) |
| `TIDEMARK_API_KEYS` | – | Comma-separated accepted keys (unset = no authentication) |
| `TIDEMARK_DB` | `tidemark.sqlite` | SQLite database file |
| `TIDEMARK_HOST`, `TIDEMARK_PORT` | `127.0.0.1`, `18200` | Listen address |
| `TIDEMARK_MAX_WORDS` | 300 | Words per record before a message is split |
| `TIDEMARK_PREVIOUS_CONTEXT_CHARS` | 160 | Characters of the previous message attached as reply context (0 = off) |
| `TIDEMARK_LEXICAL_WEIGHT` | 0.5 | Share of the lexical ranking in the fusion |
| `TIDEMARK_RRF_K` | 60 | Reciprocal-rank-fusion constant |
| `TIDEMARK_BACKEND` | `engine` | `engine` = memory-engine ranking first, hybrid ranking fills the rest; `builtin` = hybrid ranking only |
| `TIDEMARK_ENGINE_MERGE` | `first` | With `engine`: `first` = the engine's hits first (unchanged); `fused` = engine ranking (depth ≥ 100) and hybrid ranking combined by reciprocal rank (`TIDEMARK_RRF_K`). Per request `engine_merge`. Experimental: LoCoMo (local e5, all evidence in the first 10) 74.3 % → 77.2 %, no category worse; hybrid-only scored 81.4 % on the same data; at 25 results identical to `first`. Retrieval only; effect on answers not measured |
| `TIDEMARK_WINDOW` | 5 | Consecutive messages of one Add request per record (non-overlapping) |
| `TIDEMARK_CACHE_MB` | 2048 | Estimated memory budget for users held in memory; least recently used users are evicted and reloaded on demand |
| `TIDEMARK_MAX_CONCURRENT` | 8 | Concurrent requests; beyond it HTTP 429 with `Retry-After: 2` |
| `TIDEMARK_RERANK_CANDIDATES` | 0 | How many leading candidates the reranker re-orders (0 = off); a search may override it with `rerank_candidates`. On development cases about facts that change, 30 candidates with `TIDEMARK_RERANK_SCOPE=all` and a local bge-reranker-base raised complete evidence at top 10 from 28 to 42 of 51 questions, at about 2 s per search on CPU (docs/EXPERIMENTS.md) |
| `TIDEMARK_RERANK_DIR` | – | Local cross-encoder reranker (directory with `model.onnx` and `tokenizer.json`) |
| `TIDEMARK_RERANK_PROTOCOL` | – | Reranking API instead: `cohere` (also vLLM, SiliconFlow, Xinference), `jina`, `voyage`, `tei`, `dashscope`, `template`. See [Reranking](#reranking) |
| `TIDEMARK_RERANK_URL`, `_API_KEY`, `_MODEL`, `_HEADERS`, `_AUTH` | per protocol | As for embeddings |
| `TIDEMARK_RERANK_BATCH`, `TIDEMARK_RERANK_MAX_CHARS` | per protocol, 8000 | Documents per API request; characters of a record sent |
| `TIDEMARK_RERANK_TEMPLATE`, `_TEMPLATE_RESULTS`, `_TEMPLATE_INDEX`, `_TEMPLATE_SCORE` | –, `results`, `index`, `relevance_score` | `template` protocol: JSON body and where the scores are in the reply |
| `TIDEMARK_SKIP_RERANK_CHECK` | – | Set to skip the startup call to a reranking API |
| `TIDEMARK_PREFERENCES` | `off` | Preference extraction: `off`, `rules` or `llm` (needs `TIDEMARK_LLM_*`) |
| `TIDEMARK_PREFERENCE_SLOTS` | 0 | Final search slots for the user's most relevant preferences (0 = none); a search may override it with `preference_slots` |
| `TIDEMARK_PREFERENCE_BATCH_CHARS`, `_WORKERS`, `_CONTEXT`, `_ATTEMPTS` | 12000, 2, 50, 4 | `llm` mode: characters of user messages per model call; users processed at once; existing preferences shown to the model; failed attempts before a batch falls back to the rules (retries wait 1, 2, 4 … minutes) |
| `TIDEMARK_PREFERENCE_DELAY` | 10 | `llm` mode: seconds new records wait before the model reads them, so one call covers several Adds (0 = at once) |
| `TIDEMARK_PREFERENCE_LABELS` | `plain` | How preferences are labelled in search results: `plain` (all "[preference]"), `kind` (what the user already does as "[the user already does this]"), `novelty` (as `kind`, and "not new to them" when the question asks for something new); a search may override it with `preference_labels` |
| `TIDEMARK_PREFERENCE_NOTE` | – | An extra line added to every preference in search results |
| `TIDEMARK_PREFERENCE_PROMPT_FILE` | – | `llm` mode: replaces the extraction instructions |
| `TIDEMARK_LLM_URL`, `_MODEL`, `_API_KEY`, `_PROTOCOL` | –, –, –, from the URL | Optional language model; see [Language model](#language-model) |
| `TIDEMARK_LLM_MAX_TOKENS`, `_TEMPERATURE`, `_RETRIES`, `_NATIVE_JSON` | 4096, 0, 3, on | Reply budget, sampling temperature, retries of rate limits and server errors, use of the API's JSON mode |
| `TIDEMARK_LLM_HEADERS`, `_REQUEST_TEMPLATE`, `_RESPONSE_PATH`, `_COMMAND`, `_COMMAND_CWD`, `_USAGE_*` | – | Extra headers; custom API; command tool; where token counts are |
| `PROVIDER_EGRESS_REQUEST_TIMEOUT_MS` | 30000 | Timeout of one language-model request |
| `TIDEMARK_SKIP_LLM_CHECK` | – | Set to skip the startup call to the language model |
| `TIDEMARK_RERANK_SCOPE`, `TIDEMARK_RERANK_MAX_TOKENS` | hybrid, 512 | `all` reranks the engine's results together with the hybrid candidates (otherwise the engine's stay in front); tokens per question–record pair |
| `TIDEMARK_MAX_RESULTS` | 0 | Most results per search even if `top_k` is larger (0 = `top_k`); a search may override it with `max_results` |
| `TIDEMARK_MAX_CHARS` | 0 | Most characters of content per search; the first result is always returned (0 = no limit); a search may override it with `max_chars` |
| `TIDEMARK_USER_WEIGHT` | 1 | Weight of a second ranking over the user's own lines (`user: …`), added to the main ranking (0 = off); a search may override it with `user_weight` |
| `TIDEMARK_INSTRUCTION_SLOTS` | 0 (1.0.0: 10) | Slots at the end of the results for the user's lines that cue rules flag as standing instructions or preferences ("Always …", "I prefer …", "please remember"), most relevant first, labelled `[standing instruction, <date>]`; at most one per 10 results. **Off by default since 1.2:** the rules also flag ordinary text (pasted examples, narrative "I prefer …"), which is then shown as the user's standing instruction. Set `10` (or per search `instruction_slots: 10`) for the 1.0.0 behaviour |
| `TIDEMARK_SEARCH_RULES` | `on` | Search returns the user's active [standing rules](#standing-rules) in a separate `rules` field (absent when there are none); `off` leaves it out. Per search `rules: false/true` |
| `TIDEMARK_PASSAGES` | `off` | `on`: each long record in Search results also carries its best-matching passage (`passage`); per search `passages` |
| `TIDEMARK_PASSAGE_MAX_CHARS` | 400 | Longest passage in characters (minimum 40); per search `passage_max_chars` |
| `TIDEMARK_PASSAGE_MIN_WORDS` | 120 | Records with fewer words get no passage; per search `passage_min_words` |
| `TIDEMARK_ADOPT_EMBEDDINGS` | – | Set once to record the configured embedding model for a database made by an older version (otherwise it is only checked for matching dimensions and a warning is logged); a database whose recorded model or dimensions differ from the configuration is refused |
| `TIDEMARK_PART_SPEAKERS` | off | Experimental, not yet measured on AML's input: `on` repeats a named speaker in every part of a split message |
| `TIDEMARK_HEADING_WORDS` | off | Experimental: `on` stops "Reminder: …", "Note: …" and similar headings from being taken for a speaker's name |
| `TIDEMARK_USER_LINES` | first | Experimental: `continued` also counts the following lines of a multiline user message for the user ranking and instructions |
| `TIDEMARK_FOLLOWUPS` | 0 | Experimental: at most this many of the user's statements from other dates attached to the returned records they relate to ("I've stopped running…" under "I go running every morning"), each as a line `[said later, <date>] user: …` plus a `followups` field on the result. No result is displaced, no rank changes, stored records are untouched (0 = off); a search may override it with `followups`. See docs/EXPERIMENTS.md |
| `TIDEMARK_FOLLOWUP_SCOPE` | later | `later`: only statements dated after the returned one; `any`: earlier ones too (`followup_scope`) |
| `TIDEMARK_FOLLOWUP_RARE` | 0.05 | Share of the user's records a term may occur in and still link a follow-up (`followup_rare`) |
| `TIDEMARK_FOLLOWUP_ANCHOR_SHARE` | 0.5 | How much of the question's wording a returned sentence must cover to have follow-ups (`followup_anchor_share`) |
| `TIDEMARK_FOLLOWUP_MIN_TERMS` | 0 | When set, a follow-up may also link through this many shared distinctive terms the question does not name; finds more, attaches more unrelated lines (`followup_min_terms`) |
| `TIDEMARK_FOLLOWUP_CHARS` | 400 | Most characters of one follow-up line (`followup_chars`) |
| `TIDEMARK_APPLIES` | off | Experimental: `on` adds, for a returned user sentence that states its own period ("From next month…", "…until February", "for all of October"), the period resolved against the record's date: a line `[applies: "from next month" = from September 2023]` and an `applies` field (`from`, `until`, `until_latest`, `from_precision`, `until_precision`, `year_assumed`). A bound keeps the precision of its words (day, month or year); `until` is the last one the statement certainly applies, and where the words leave open whether the named time is included ("until February"), `until_latest` is the named time. Not recommended for use: on an independently written set its first version got 5 of 9 annotations wrong. Computed at Search; nothing is stored and no record changes; a search may override it with `applies`. See docs/EXPERIMENTS.md |
| `TIDEMARK_INSTRUCTION_DEDUP` | off | Experimental: `on` skips an instruction whose record is already among the results and gives its slot to the next record |
| `TIDEMARK_LOG_TIMING` | – | Set to log how long each step of a search takes |
| `TIDEMARK_SPEAKER_BOOST` | 0 | Extra weight for messages by a person the query names (measured: no gain on LoCoMo, so off) |

## Results

Retrieval with `text-embedding-v4`: how much of each question's gold evidence is among the top 100 results (public benchmark
data replayed through Add/Search; details and every comparison in [docs/REPORT.md](docs/REPORT.md)):

| Benchmark | Questions | Evidence in the top 100 |
|---|---|---|
| LoCoMo (all 10 conversations) | 1,977 | all of a question's evidence turns: 98.9 % of questions (99.1 % of all evidence turns) |
| LongMemEval-S (18 haystacks, 3 per type) | 18 | all answer sessions: 100 % of questions |
| BEAM 100K (20 conversations) | 355 | 88.2 % of a question's evidence messages, on average |
| BEAM 1M (5 conversations) | 89 | 70.4 % of a question's evidence messages, on average |

On the Agent Memory Leaderboard (Cycle 2), v1.0 scored **52.2** on the full textual evaluation (open-source track);
the capability breakdown is in the report.

## Technical report

`docs/REPORT.md` describes the method and every measurement behind the defaults, including ideas that did not help. `docs/ARCHITECTURE.md` describes the components, the Add and Search pipelines, and known limitations.

## Benchmarks and checks

- `bench/locomo_recall.py LOCOMO_JSON --url URL --key KEY`: replays LoCoMo through Add (20 messages per request, with session times) and reports, by category, the share of gold evidence found in the top 10, 25, 50 and 100 results.
- `bench/longmemeval_recall.py longmemeval_s_cleaned.json --url URL`: replays a sample of LongMemEval-S (one user per haystack) and reports whether the answer sessions come back in the top 10, 25, 50 and 100, by question type.
- `bench/robustness.py URL [KEY]`: checks beyond the contract. It covers text without spaces (e.g. Chinese), 6 MB requests, empty, image-only, role-less and timestamp-less messages, odd searches, concurrent Adds and Searches on one user, and isolation after eviction and reload (run with a small `TIDEMARK_CACHE_MB`).
- `bench/soak.py longmemeval_s_cleaned.json URL SERVER_PID`: AML-like load from parallel clients over many LongMemEval-S haystacks. It reports errors by status, latency percentiles and server memory over time.
- `bench/api_check.py URL KEY`: checks the memory API: list, get, correct, delete, delete user and history, user isolation, and that Search reflects every change at once.
- `bench/beam_recall.py 100K.parquet --url URL`: replays BEAM (needs `pyarrow`) and reports the share of evidence messages in the top 10, 25, 50 and 100 by question type; `--report` evaluates result caps offline.
- `bench/clbench_recall.py CL-bench.jsonl --url URL`: replays CL-bench multi-turn tasks and reports how much of each task's reference document is returned.
- `bench/rules_check.py URL KEY`: checks standing rules: add, list, edit, revoke, purge, history, search's separate `rules` field and `rule_scope`, user boundaries, user deletion, and that conflict links are kept as hints (a contradiction without a shared word is not linked).
- `bench/passages_check.py URL KEY`: checks best-matching passages: the matching paragraph of a long record, speaker, offsets that reproduce the exact text (including emoji, combining marks and CJK), short records, `passage_max_chars`, and that ranking and content are unchanged.
- `bench/client_check.py URL KEY`: rules and passages through the Python client (`pip install ./clients/python` first).
- `bench/conformance.py URL KEY`: checks the Add/Search contract. It covers credentials, echoed IDs, immediate searchability, idempotent retries, `top_k`, user isolation, options, content parts, long messages and concurrent writes.

## Origin

Tidemark was written for Unlimited Memory (2026). The name: a tidemark is the line the tide leaves behind, a dated trace of what happened. It contains no third-party code beyond the Rust crates listed in `Cargo.toml`.
