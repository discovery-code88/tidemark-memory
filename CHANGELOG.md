# Changelog

## 1.2.0 (unreleased)

The first release after 1.0.0. A 1.1.0 was prepared but never published; its changes (listed under "Prepared as
1.1.0" below) ship in 1.2.0. Migration from 1.0.0: `docs/MIGRATION-1.2.md`.

**Changed default:** the instruction slots are off (`TIDEMARK_INSTRUCTION_SLOTS` 0; 1.0.0: 10); see below. With
`TIDEMARK_INSTRUCTION_SLOTS=10`, Search returns what 1.0.0 returns for the same stored records.

### New in 1.2.0

- **Standing rules** (`/rules`): a user can state rules on purpose ("Don't use Git"), with an optional scope, then
  list, edit (versioned), revoke (history kept) or purge them. Search returns active rules in a separate `rules`
  field that never takes a result place; `rules: false` and `rule_scope` per search, `TIDEMARK_SEARCH_RULES=off`
  globally. Rules that may contradict each other all stay active and name each other (`possible_conflicts`);
  nothing is resolved automatically. The conflict links are a hint and may be incomplete: contradictions worded
  without a shared word are not found, and an empty list does not mean there is no conflict. Responses for users without rules are unchanged. Python client:
  `add_rule`, `rules`, `get_rule`, `update_rule`, `revoke_rule`, `purge_rule`, `rule_history`,
  `search_with_rules`. Check: `bench/rules_check.py` (in CI).
- **Best-matching passages** (opt-in; `passages: true` per search or `TIDEMARK_PASSAGES=on`): each long record in
  the results also carries the sentences within one message that best match the question, with speaker and character
  offsets into the complete `content`. Ranking and content are unchanged; no embedding calls. The record stays
  authoritative; the passage is a reading aid chosen by words and can miss a paraphrase or omit a qualification. On the 10 questions of
  the real-work trial (exposed; the word matching was adjusted after seeing them) the passage held the deciding text
  for 9, including the mixed-topic record of failure case T08; the miss matched only by meaning. Check:
  `bench/passages_check.py` (in CI).

### Prepared as 1.1.0 (never published; part of 1.2.0)

**Changed default:** the instruction slots are off (`TIDEMARK_INSTRUCTION_SLOTS` 0; 1.0.0: 10). With default
settings Search therefore no longer reserves the last places for lines labelled `[standing instruction, …]`; those
places hold ranked records instead. With `TIDEMARK_INSTRUCTION_SLOTS=10` (or per search `instruction_slots: 10`)
Search returns what 1.0.0 returns for the same stored records. Preference extraction and preference slots are off
unless configured. Add differs from 1.0.0 in the time notes it writes for new records
(two corrections below) and accepts one new optional field. Records already stored are never rewritten. See
`docs/MIGRATION-1.2.md`.

- **Time notes for durations (changes what Add writes):** a duration gets a `since about …` note only when its
  sentence clearly looks back ("I've been attending for three months", "for 3 years now"). 1.0.0 also wrote one for
  forward or ambiguous durations ("I've frozen my membership for three months" → `since about March 2023`) and for
  future wording ("I'll have been … for"); those now get no note. This applies to records added from 1.2 on.
  Records already stored keep the notes they were written with; nothing is rewritten.
- **Time notes and quotations (changes what Add writes):** a relative expression inside quotation marks gets a note
  only when the quotation is what the user says at the time of the message ("I told her today, '…'"). 1.0.0 resolved
  every expression against the message date, including fiction, lyrics and other people's words ("My script says,
  'I've been ill for a month'" → `since about …`). Expressions outside the quotation are unaffected; apostrophes are
  not quotation marks. A guard against unsupported notes, not an interpretation of reported speech. New Adds only.
- **`utc_offset_minutes` on Add messages (optional):** the sender's offset from UTC. With it the date header and
  time notes use the sender's calendar day and the header names the offset; `created_at` stays the UTC instant.
  Without it nothing changes. A fixed offset is not a time zone and cannot determine daylight-saving changes.
- **Fix (found in a 1.1.0 candidate; 1.0.0 was not affected): Add no longer crashes on curly quotation marks.** The quotation check above stepped
  one byte around neighbouring quotation marks; “ ” « » ‘ ’ take several bytes, so a message with another such
  quotation next to the one holding a time expression made Add fail (connection dropped, nothing stored). Found in
  real use of release candidate 1. Regression test `multi_byte_quotation_marks_next_to_a_quotation_do_not_panic`.
- **Instruction slots off by default:** `TIDEMARK_INSTRUCTION_SLOTS` now defaults to 0. The
  cue rules that fill these slots flag ordinary text as well, and it was then presented as the user's standing
  instruction: in a real-work trial both texts that took the slot were not instructions (a pasted code review with
  example text, a one-off request); on 51 change-over-time questions 592 of 608 slot items were narrative
  paragraphs flagged on an embedded "I prefer …". The misattribution is the reason for the change; its retrieval
  effect was small (at 10 results with the engine backend, 1 question helped and 2 harmed; no effect at 100). To keep
  the 1.0.0 behaviour set `TIDEMARK_INSTRUCTION_SLOTS=10`. The setting and the per-request override are unchanged.
- **CI (found while verifying a 1.1.0 candidate):** the workflow now builds the service binary before the live API checks; `cargo test`
  alone does not produce `target/release/tidemark`, so on a fresh runner those checks could not start.

- **Local embeddings** (`TIDEMARK_EMBED_PROTOCOL=local`, `TIDEMARK_EMBED_DIR`): a sentence-embedding model run on
  the machine with ONNX Runtime; no provider, key or bill. Tested with `intfloat/multilingual-e5-small` (ONNX, 384
  dimensions); configuration, measured CPU cost and limits in the README ("Local embeddings"). Opt-in; the default
  without embedding settings is still the offline placeholder.
- **Fused ranking** (experimental, off by default; `TIDEMARK_ENGINE_MERGE=fused`, per request `engine_merge`): with
  the default engine backend, combines the engine's ranking (depth ≥ 100) with the hybrid ranking by reciprocal rank
  instead of letting the engine's hits fill every slot. On LoCoMo with local embeddings it put all evidence in the
  first 10 for 77.2 % of questions against 74.3 % for the default (62 questions gained, 3 lost; no category worse);
  at 25 results it equals the default. **This is better top-10 evidence coverage on one dataset used in development,
  not better answers: its effect on answers has not been measured.** The default stays engine first.

- **Memory API:** `GET /memories` (list, by session, paginated), `GET /memories/{id}`, `PATCH /memories/{id}`
  (correct a memory; re-embedded, previous text kept in its history), `DELETE /memories/{id}`,
  `GET /memories/{id}/history`, `DELETE /users/{user_id}` (removes a user's memories, request log and history).
  All scoped to the requesting user; changes are durable on return and visible to the next Search.
- **Follow-ups** (experimental, off by default; `TIDEMARK_FOLLOWUPS`, per request `followups`): the user's later
  statements about the same thing attached to the returned record they relate to, without displacing results.
  Development results in `docs/EXPERIMENTS.md`.
- **When a statement applies** (experimental, off by default; `TIDEMARK_APPLIES`, per request `applies`): the period
  a returned sentence states for itself ("From next month…"), resolved against its record's day, keeping the
  precision of the words (day, month or year) and explicit boundaries (`until` certain, `until_latest` possible).
  Computed at Search; nothing stored. **Not recommended for use:** on an independently written set its first version
  got 5 of 9 annotations wrong; the defects found were fixed, but only re-checked on the same, now exposed, cases.
- **Python client** (`clients/python`, standard library only) and examples (`examples/quickstart.py`, `examples/curl.md`).
- **Docker:** `Dockerfile` (multi-stage, non-root, health check), `docker-compose.yml` with a data volume, `.env.example`.
- **CI:** GitHub Actions runs the service and engine tests and live conformance, memory-API and quickstart checks.
- **Docs:** `docs/ARCHITECTURE.md`; README with quickstart, full API reference and results.
- `bench/api_check.py` for the memory API.
- **Speed, same results:** measured on the largest history of the leaderboard run (8,458 records), with identical search results to 1.0.0 under the same settings (66 of 66 searches identical in IDs, content and scores; measured with 1.0.0's settings, instruction slots on; equivalence tests on the real records).
  - Reloading a user: 299 s → 66 s. The write gate's novelty check uses cached, interned token sets (bit-identical values).
  - Search CPU per query: ~1.15 s → ~0.19 s. The engine's per-memory text features are precomputed (Jaccard and coverage bit-identical; cosine with sums in a fixed order instead of hash-map order), and BM25 statistics are kept up to date per user instead of rebuilt per search.
  - Memory: about 20 % more for users that have been searched (cached features).
- **Any embedding provider:** native support for Azure OpenAI, Google Gemini, Cohere, Voyage AI, Jina AI, Ollama and
  Hugging Face TEI besides OpenAI-compatible endpoints, plus a JSON template for any other API
  (`TIDEMARK_EMBED_PROTOCOL`). Questions and stored texts get the provider's query and document input types, or
  configurable prefixes; extra headers, the key's header and the input cap are settings. Existing settings and
  databases work unchanged (same recorded model name).
- **Reranking APIs:** besides the local ONNX cross-encoder, `TIDEMARK_RERANK_PROTOCOL` uses Cohere's `/rerank`
  (also vLLM, SiliconFlow, Xinference), Jina, Voyage, Hugging Face TEI, DashScope, or a JSON template. Checked at
  start; a failed call during a search keeps the search's order. Reranking stays off by default.
- **Preference memory:** what users like, avoid, habitually do or want, kept as versioned statements with when they
  apply and the user's own words as evidence (`GET/POST /preferences`, `GET/PATCH/DELETE /preferences/{id}`,
  `GET /preferences/{id}/history`). Off by default. Found by rules (also in later lines of multiline messages) or by the
  language model after Add (durable queue, verified evidence, replacements as new versions, fallback to rules).
  Search adds the most relevant ones in `preference_slots` (default 0). With the model, new records wait
  `TIDEMARK_PREFERENCE_DELAY` seconds (default 10) so one call covers several Adds. Each preference has a kind (`likes`, `does`,
  `instruction`); search shows all as "[preference]" by default, and optionally labels what the user already does
  (`kind`) and marks it as not new for requests for something new (`novelty`) (`TIDEMARK_PREFERENCE_LABELS`, per
  request `preference_labels`); `TIDEMARK_PREFERENCE_NOTE` adds a line to each.
  The rules skip short reactions ("I love that!") and narration ("I often find myself …"). Inspired by M-flow's
  preference points; independent implementation. `bench/preference_check.py` checks it (also in CI).
  **Measured, not yet shown to help:** on PersonaMem 32k neither extraction mode showed an overall gain, and label
  variants demonstrated no benefit (see `docs/EXPERIMENTS.md`); hence extraction is off and labels are plain by default.
- **Optional language model** (`TIDEMARK_LLM_*`): any chat API through the engine's protocol adapters
  (OpenAI-compatible, Anthropic, Gemini, custom JSON template, local command-line tool), with JSON-mode fallback,
  retries of transient failures, token counting and an optional operator cap on attempts
  (`TIDEMARK_LLM_MAX_ATTEMPTS`). Checked at start. Add and Search do not use it; it is the basis
  for features that need a model. The engine exposes it as `tidemark_engine::ChatModel`.
- **`tidemark reembed`:** moves a database to another embedding model; resumable, swapped in atomically, with
  `--estimate` and `--abort`. The service refuses to start while a run is unfinished.
- `TIDEMARK_LOG_TIMING=1` logs the time of each step of a search.
- **Docs:** report corrected on the leaderboard's input format; leaderboard result added.

## 1.0.0 (2026-10-27)

The version evaluated on the Agent Memory Leaderboard (Cycle 2), tag `v1.0`.

- Add/Search service on SQLite with the Unlimited Memory engine, hybrid BM25 + embedding ranking,
  a ranking over the user's own lines and standing-instruction slots, date lines and resolved relative times.
- Safeguards: embedding model and dimensions recorded and checked per database; request deduplication per user and
  request ID; per-user locks; bounded in-memory cache; 429 with Retry-After beyond the concurrency limit.
- Experimental settings, off by default: speaker on every part of a split message, heading words, multiline user
  lines, instruction deduplication, combined reranking scope.
