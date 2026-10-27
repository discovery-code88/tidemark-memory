# Tidemark v1.0 — technical report

Tidemark is a memory layer with two operations, Add and Search. It was built for the Agent Memory Leaderboard (AML) open-source track. AML fixes the answer model, its prompts, the judge and `top_k = 100`, so what Tidemark controls is what it stores and which records it returns.

This report describes the method and every measurement behind the v1.0 defaults, including the ideas that did not help. All numbers come from the scripts in `bench/` and can be reproduced (see the end).

> **Status of the numbers.** The measurements below use a hashed bag-of-words placeholder instead of `text-embedding-v4`, because the embedding key was not available while they were taken. They compare designs fairly with one another. Absolute values will change once real embeddings are used, and this report will be updated with them.

## 1. Architecture

```
Add  ──► durable SQLite write log ──► records (5 messages, dated, time notes)
                                      ├─► Tidemark memory engine (in memory)
                                      └─► BM25 index + embeddings
Search ─► engine ranking first, then BM25 + embedding fusion fills to top_k
```

- **Service** (`src/`): HTTP API, write log, record rendering, BM25, embedding client, rank fusion, optional ONNX reranker. Overload is answered with HTTP 429 and `Retry-After`.
- **Memory engine** (`engine/`, crate `tidemark-engine`): the memory core of Unlimited Memory, open-sourced.
  - write gating with per-memory salience, stability, utility and conflict scores;
  - lexical retrieval with query intents, temporal clauses and fact handling;
  - semantic core, procedural memory, date resolution, answer generation and verification, person profiles.

  It runs in memory. The service replays its write log into it on start. Local ONNX models in the engine are opt-in and off by default, so only `text-embedding-v4` is used.

## 2. Records and time notes

Each Add request (AML sends 20 messages) is split into non-overlapping records of up to `TIDEMARK_WINDOW = 5` consecutive messages. Each record has:
- the date of its first message as a header;
- one line per message, `Speaker: text`;
- deterministic time notes beside each message:

| Expression | Note (message dated 8 May 2023) |
|---|---|
| yesterday, three days ago | `= 7 May 2023`, `= 5 May 2023` |
| last week, two weeks ago | `= the week before 8 May 2023`, `= 2 weeks before 8 May 2023` |
| last Friday, over the weekend | `= the Friday before 8 May 2023`, `= the weekend before 8 May 2023` |
| two years ago, last year | `= 2021`, `= 2022` |
| for 3 years now | `= since about 2020` |
| next month, last summer | `= June 2023`, `= summer 2022` |

Week-, weekend- and weekday-based expressions deliberately stay relative to the message date. AML's judge marks a computed day wrong when the reference answer is relative ("no calendar math"), and its answer prompt asks to keep week-based expressions relative. A message longer than 300 words is split on its own. The notes restate what the message and its date already say; nothing depends on a later question.

## 3. Search

1. The memory engine ranks the user's records (it returns only what it judges relevant, typically 10–20 records).
2. The remaining places up to `top_k` are filled by reciprocal-rank fusion (k = 60, equal weights) of BM25 over record text (including dates and time notes) and cosine similarity of `text-embedding-v4` embeddings of the message text.
3. A second ranking over only the user's own lines (`user: …`, BM25 plus the record's embedding) is added to the fused score with weight 1 (`TIDEMARK_USER_WEIGHT`). Questions are mostly about what the user said, which long assistant replies would otherwise outweigh.
4. The user's standing instructions and preferences ("Always …", "Never …", "From now on …", "I prefer …"), ranked among themselves, take up to 10 slots at the end of the results (`TIDEMARK_INSTRUCTION_SLOTS`), each as a short item labelled `[standing instruction, <date>]`.
5. Multiple-choice `options` are appended to the query. At most `top_k` items are returned, with strictly descending scores.

BEAM (placeholder embeddings, `bench/beam_recall.py`), share of evidence messages in the top 100: 100K-token split 67% → 85% with steps 3–4, 1M-token split (5 conversations) 55% → 66%; the top 10 is unchanged. LongMemEval-S stays at 100%; LoCoMo is unaffected (its messages name their speaker, so no record has `user:` lines). Truncating the results (`TIDEMARK_MAX_RESULTS`, `TIDEMARK_MAX_CHARS`, off) only lost evidence on BEAM.

With real `text-embedding-v4` embeddings (2026-10-26), share of evidence messages in the top 100, new rankings off → defaults: BEAM 100K (20 conversations, 355 questions) 79.4% → 88.2%; BEAM 1M (5 conversations, 89 questions) 65.1% → 70.4%; LongMemEval-S (18 haystacks) 100% in both, 94.4% → 100% within the top 25. Summarization questions, which mostly need assistant replies, lose (100K 48.2% → 42.9%, 1M 18.9% → 12.0%), mostly through the user weight (with it off and the instruction slots kept: 46.6% and 17.7%); every other type gains or holds. Embedding cost: about 350 tokens per record (4.3 characters per token); about 0.15 M tokens per BEAM 100K conversation, 1.0 M per BEAM 1M conversation, 0.12 M per LongMemEval-S haystack.

**Input format caveat.** These replays send each message as `user: …` / `assistant: …`, which is what steps 3–4 detect. AML's own Add requests (seen in its smoke test) look different: `[dataset][session_n][D1:6][text] Speaker: …`, with speaker names or source labels (`Caroline`, `Christina Baker` for BEAM's user, `NARRATION`, `document`), and no role that reaches the record text. Only 98 of the 730 smoke-test records contained a `user:` line. The gains above therefore hold for the replayed format and are not evidence of gains on AML; for most AML data the user ranking and instruction slots have little to act on. A faithful replay of AML's input is the next step.

## 4. Measurements (LoCoMo, placeholder embeddings)

### 4.1 Evidence recall: record size

`bench/locomo_recall.py` replays LoCoMo as AML does and measures `all@k`: the share of questions whose gold evidence turns are all in the top k.

| Messages per record | all@10 | all@25 | all@100 | multi-hop @100 | inference @100 | text per question |
|---|---|---|---|---|---|---|
| 1 | 59.2% | 71.2% | 85.3% | 45.9% | 47.2% | 28 KB |
| 2 | 60.0% | 71.3% | 85.2% | 46.6% | 52.8% | 29 KB |
| **5** | 72.5% | 83.5% | **96.5%** | **83.6%** | **84.3%** | 70 KB |
| 10 | 79.2% | 89.7% | 100% | 100% | 100% | 93 KB |
| 20 | 85.7% | 95.6% | 100% | 100% | 100% | 92 KB |

From 10 messages per record, 100 records hold an entire LoCoMo conversation, so recall is trivially complete.

### 4.2 Answer quality: record size (proxy)

To see whether more context turns into correct answers, a stand-in answer model (agnes-2.5-flash, not AML's gpt-4o-mini) answered 313 LoCoMo questions (categories 1–4, every fifth question) from the top-100 records. It used AML's published LoCoMo answer prompt, and the answers were graded by hand against AML's published judge rules.

| Messages per record | correct | multi-hop | temporal | inference | single-hop |
|---|---|---|---|---|---|
| 1 | 67.7% | 51% | 54% | 57% | 80% |
| **5** | **71.9%** | 47% | **63%** | 57% | **86%** |
| 10 | 69.3% | 49% | 61% | 52% | 81% |

1 → 5 messages: +37 / −24 questions. 5 → 10: +20 / −28.

Handing over the whole conversation (10) did not help, so Tidemark uses 5. Multi-hop did not improve despite much higher recall: AML's judge marks list answers wrong for missing *and* extra items, which the answer model rarely gets exactly right. The differences are not statistically conclusive (sign test p ≈ 0.12 for 1 vs 5), and this is one dataset with a proxy answer model.

### 4.3 Engine ranking

Same records (5 messages), hybrid ranking alone vs engine first + hybrid fill:

| | all@10 | all@25 | all@50 | all@100 | temporal all@10 |
|---|---|---|---|---|---|
| hybrid ranking | 72.5% | 83.5% | 89.9% | 96.5% | 76.9% |
| **engine first + fill** | **74.1%** | **85.1%** | **90.3%** | **96.6%** | **85.9%** |

On its own the engine returns too few records (median 15) to use a 100-record budget. Combined, it improves the top ranks, especially for temporal questions, without losing recall.

### 4.4 Ideas that did not help

- **Speaker boost** (rank messages by a person the question names higher): multi-hop all@100 +0.4 points at either weight tried (0.5, 1.0); at 1.0 adversarial all@100 −3.3 points. Off by default.
- **One message per record:** best precision per record, but about half of each conversation never fits in 100 records; see 4.1.
- **Whole conversation** (10 or more messages per record): complete recall, but no better answers than 5; see 4.2.
- **Computed dates for week-based expressions** (an earlier version wrote `last Friday = Friday 5 May 2023`): conflicts with AML's judge rules, replaced by relative-anchored notes.

## 5. Conformance and robustness

`bench/conformance.py` checks:
- credentials in all three schemes, and health without credentials;
- echoed IDs and searchable-on-return;
- idempotent retries, `top_k`, user isolation, options;
- content parts, long messages, concurrent Adds, and 4xx for malformed requests.

All checks pass with both backends. Restarting the service keeps all data: the write log is replayed. Under overload (tested with a limit of one concurrent request) the service answers 429 with `Retry-After: 2`. Adds of one user are serialized so retries stay idempotent; different users are added in parallel.

The engine carries 291 unit tests. Tests that needed the private host application's HTTP test harness were not carried over.

## 6. Scale

**Single long history.** 25,000 messages for one user (about 1M tokens, BEAM size), short messages:
- Add stays at 2–3 ms per request;
- Search grows linearly from 10 ms at 2,500 messages to 86 ms (max 136 ms) at 25,000;
- server memory 152 MB.

**Many histories (LongMemEval-S practice, 48 haystacks of about 50 sessions each).**
- *Quadratic ingestion, found and fixed.* The first run showed loading time per haystack rising from 4.5 s to 115 s. On every write the engine copied all stored memories of all users for its episode, self-model and autobiographical builders, and re-evaluated meta-memory for every memory. All of these only depend on the memory's own conversation. Scoping them is behaviour-preserving (the engine's 291 tests pass; search results are byte-identical), and loading now stays flat at 5–9 s per haystack. Passing references instead of copies to the builders also cut a cold reload of one haystack from 5.5 s to 2.3 s.
- *Memory, found and fixed.* Holding everything in memory cost about 19 MB per haystack, which would exceed the machine's memory over AML's ~1,500 histories. SQLite now holds all records, and only recently used users are kept in memory within `TIDEMARK_CACHE_MB`. With a deliberately small 200 MB budget the server stayed at 288–311 MB after 48 haystacks, every search reloading its user. Recall was identical to the uncapped run.

**Robustness** (`bench/robustness.py`, 16 MB cache, 76 evictions during the run). All checks pass:
- a 22,000-character message without spaces (split by characters, since word counts do not apply; embedding inputs capped);
- Japanese and Chinese: partial phrases find their sentence first (keyword search indexes CJK text as overlapping character pairs);
- a 6 MB request (body limit raised to 64 MB);
- empty, image-only, role-less and timestamp-less messages; emoji and right-to-left text;
- unknown users, `top_k` 0 and 10,000, queries and options given as content parts or objects;
- 30 concurrent Adds plus 30 Searches on one user (each accepted Add stored exactly once);
- after eviction and reload, each of 40 users finds only their own data, also with 40 users added and searched by 6 parallel clients during constant eviction (198 evictions in the run).

With the embedding service unreachable, Add answers 503 after its retries and stores nothing, so a retry starts clean. With a wrong key (HTTP 401) it answers 503 at once, and the service refuses to start, since a startup check makes one embedding call. SIGTERM shuts the service down gracefully.

**Soak** (`bench/soak.py`, 4 parallel clients, 512 MB cache):
- 500 LongMemEval-S haystacks: 24,129 Adds and 663 Searches, all answered 200 (429s retried), no server errors;
- Add p50 about 70 ms (p95 about 240 ms), Search p50 about 75 ms;
- a Search for an evicted user about 2.5 s (the reload).
- *Memory:* with glibc's default allocator settings it kept growing past the budget (807 MB at haystack 300, 901 MB at 500). With `MALLOC_ARENA_MAX=2` it levelled off at 578–585 MB for the whole second half of the run, so the growth was allocator fragmentation, not a leak. The deployment sets `MALLOC_ARENA_MAX=2`.

**LongMemEval-S evidence recall** (48 questions, 8 per type, placeholder embeddings): all@25 97.9%, all@50 and all@100 100%. Preference questions are weakest at the very top (all@10 25%).

## 7. Limitations

- The recall measurements use the benchmarks' public data replayed in a `role: text` format, which differs from AML's tagged input (see section 3); answer-level comparisons use a stand-in answer model (gpt-5.6-luna) with AML's published prompts and judges.
- Long messages split into several records repeat their speaker only in the embedding input; in the stored text, later parts of a named or tagged message carry no speaker (`TIDEMARK_PART_SPEAKERS=on` changes this for named messages; not yet measured on AML's format).
- The engine's write gate rejects about half of single messages as low-signal. With 5-message records it rejects almost none, and the hybrid ranking covers what it does not keep.
- The engine includes answer generation, verification and person profiles, but this service exposes only Add and Search; those parts are library code here.
- The service runs on one machine. A user who was evicted from memory costs one reload from SQLite (about 2 s for a LongMemEval-size history) on their next request; for a 1M-token BEAM history about 21–34 s.

## 8. Reproducing

```sh
cargo build --release
# start the service (empty database), then:
python3 bench/locomo_recall.py locomo10.json --url http://127.0.0.1:18200 --out results.json
python3 bench/conformance.py http://127.0.0.1:18200 <key>
```

`TIDEMARK_WINDOW=1|5|10`, `TIDEMARK_BACKEND=builtin|engine` and the other settings in the README select the variants above.
