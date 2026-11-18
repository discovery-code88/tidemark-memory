# Migrating from 1.0.0 to 1.2.0

1.2.0 is the first release after 1.0.0. A 1.1.0 was prepared but never published; everything it contained is part
of 1.2.0, so this guide covers every change since 1.0.0.

## Summary
- **Stored data:** no migration step. 1.2.0 opens a 1.0.0 database as it is and adds its new tables (preferences,
  rules) on start. No record, note or embedding is rewritten.
- **Search, one default changed:** instruction slots are off (`TIDEMARK_INSTRUCTION_SLOTS` 0; 1.0.0: 10). Default
  searches no longer end with `[standing instruction, …]` items; those places hold ranked records instead. With
  `TIDEMARK_INSTRUCTION_SLOTS=10` the same stored records give the same results as 1.0.0. Everything else Search does
  by default is unchanged. See "Instruction slots" below.
- **Search responses can carry two new fields**, both absent unless used: `rules` (top level, only when the user has
  active standing rules; README, "Standing rules") and `passage` (per result, only with `passages: true`). Clients that
  reject unknown fields need to accept them before rules or passages are used.
- **Add:** records added with 1.2.0 can carry different time notes than 1.0.0 would have written for the same message
  (two corrections), and messages may carry a new optional field (`utc_offset_minutes`). Everything else Add writes
  is unchanged. Add no longer fails on text with curly quotation marks next to a quotation (a defect of the
  unpublished 1.1.0 candidates, not of 1.0.0).
- **New endpoints** (nothing existing changes): memory management (`GET/PATCH/DELETE /memories…`,
  `DELETE /users/{id}`), preferences (`/preferences`), standing rules (`/rules`).
- **New options** are off by default, apart from `TIDEMARK_SEARCH_RULES` (on), which has no effect until a user has
  rules.

## Standing rules (new)
- Rules are stated on purpose through `/rules` ("Don't use Git", optional scope); nothing in Add, Search or
  preference extraction creates, changes or revokes one. Each change is a version; `DELETE` revokes (history kept),
  `?purge=true` erases. `DELETE /users/{id}` removes a user's rules.
- Search returns active rules in a separate `rules` field; they never take one of the `top_k` places. `rules: false`
  leaves them out; `rule_scope` limits them to one scope plus unscoped rules; `TIDEMARK_SEARCH_RULES=off` turns the
  field off.
- `possible_conflicts` on a rule lists other active rules that **may** contradict it (overlapping scope, a shared
  word, opposite negation). It is a hint and can be incomplete: contradictions worded without a shared word are not
  detected, and an empty list does not mean the rule conflicts with nothing. Nothing is resolved automatically.

## Best-matching passages (new, opt-in)
- With `passages: true` (or `TIDEMARK_PASSAGES=on`) each long record in the results also carries `passage`: the
  sentences within one message that best match the question's words, with character offsets (Unicode code points)
  into the record's `content`.
- **The record stays authoritative; the passage is a reading aid.** It is chosen by words, so it can miss a
  paraphrase or leave out a qualification stated elsewhere in the record. Ranking and `content` are unchanged.

## Python client
- `clients/python` 1.2.0 adds `add_rule`, `rules`, `get_rule`, `update_rule`, `revoke_rule`, `purge_rule`,
  `rule_history`, `search_with_rules` and the memory and preference methods; `search(..., passages=True)` returns
  results with `passage`. Existing calls are unchanged; `search` still returns the list of results only.

## What changes for records added with 1.2.0

| | 1.0.0 | 1.2.0 |
|---|---|---|
| Duration without a clear backward reading: "I've frozen my membership for three months", "I worked there for two years", "I've lived here for three years" | `for three months = since about …` | no note |
| Duration that clearly looks back: "I've been attending for three months", "for 3 years now" | note | same note |
| Future or hypothetical duration: "By December I'll have been attending for three months" | note | no note |
| Relative time inside a quotation that is not the user's present speech: "My script says, 'I've been ill for a month'", "Ana said, 'I'll call tomorrow'" | note | no note |
| The same phrase both inside such a quotation and outside it, in one message | note | no note for that phrase |
| Relative time outside a quotation, or in the user's own present speech ("I told her today, '…yesterday…'") | note | same note |
| Message with `utc_offset_minutes` | field ignored | local date header naming the offset; notes use the sender's calendar day |
| Message without `utc_offset_minutes` | UTC | UTC, byte-identical header |

## Instruction slots (changed default)

| | 1.0.0 | 1.2.0 |
|---|---|---|
| `TIDEMARK_INSTRUCTION_SLOTS` unset | 10: up to one `[standing instruction, <date>] user: …` item per 10 results, in the last places | 0: no such items; all places hold ranked records |
| `TIDEMARK_INSTRUCTION_SLOTS=10` | same | same as 1.0.0 |
| per search `instruction_slots: N` | overrides | overrides (unchanged) |

Why: the cue rules ("always", "never", "make sure", "I prefer", "please remember", …) also flag ordinary text, such
as an example inside a pasted document or a narrative paragraph with "I prefer", and the whole line was then shown
as the user's standing instruction. **To keep the previous behaviour**, set `TIDEMARK_INSTRUCTION_SLOTS=10` in the
service environment (or send `instruction_slots: 10` with each search). Nothing stored changes either way.

## What does not change
- **Records stored before the upgrade** keep the content they were written with, including time notes that 1.2.0
  would no longer write (for example `for three months = since about March 2023` on a forward duration, or a note
  inside quoted fiction). 1.2.0 does not detect or remove them. A database therefore can hold records written under
  both rules; a record's `created_at` and your upgrade time tell which.
- **`created_at`** is the UTC instant in both versions, with or without an offset.
- **Re-adding** a conversation under a new `request_id` writes new records under the 1.2.0 rules; it does not replace
  the old ones. To replace them, delete the old records (`DELETE /memories/{id}` or `DELETE /users/{user_id}`) first.
- **Request deduplication:** repeating a `request_id` already stored for a user is still a no-op, so it does not
  refresh old notes either.

## `utc_offset_minutes`
- Optional, per message, an integer from -720 to 840: the sender's offset from UTC at the time of that message
  (`120` = UTC+02:00). A value outside the range rejects the request with 400 and stores nothing.
- Each message uses its own offset. Messages with different offsets, or with and without one, are never grouped
  into one record, so no message's relative times are read with a neighbour's offset.
- A fixed offset is not a time zone. It cannot tell when daylight-saving time starts or ends; send the offset that
  was in force for each message.
- Experimental Search options that read dates (`applies`, follow-up notes) use the same local day as the record's
  header.

## Experimental options (all off by default)
- `TIDEMARK_FOLLOWUPS` / `followups`: later statements by the user attached to the record they relate to. Small
  measured gain; most attached lines were still unrelated on the development cases.
- `TIDEMARK_APPLIES` / `applies`: **not recommended.** Rejected for enabling after 5 of 9 annotations were wrong on
  an independently written set; the fixes were only re-checked on the same cases.
- `TIDEMARK_ASSEMBLY`: made the change-over-time cases worse; do not use it for that purpose.
- `TIDEMARK_ENGINE_MERGE=fused` / `engine_merge`: engine and hybrid rankings combined by reciprocal rank. Better
  top-10 evidence coverage on LoCoMo (74.3 → 77.2 %); answers not measured. The default (`first`) is unchanged.

## New opt-in embedding option
- `TIDEMARK_EMBED_PROTOCOL=local` with `TIDEMARK_EMBED_DIR`: local ONNX embeddings (tested: multilingual-e5-small).
  An existing database records its embedding model; switching to or from local embeddings needs `tidemark reembed`.

## Known limitations of time notes (kept as they are in 1.2.0)
- **Attribution:** a note restates a phrase of a message against that message's date. It does not say whom the
  phrase is about. "Ana told me she moved last month" gets `last month = …`: that dates Ana's move as the user
  reported it; it is not a fact about the user. The quotation guard only covers text inside quotation marks;
  fiction or reported speech without them is resolved as before.
- **Scope:** a note is attached to its message, after the message's text, and names its phrase; it is not placed
  after its own sentence.
- **Durations:** present-perfect durations without "been" or "now" ("I've lived here for three years") get no
  approximate start, because the same wording can look forward ("I've frozen … for three months").
- **Units:** hours, minutes and week-long durations are not resolved ("exactly 90 minutes ago", "for two weeks").
- **Ambiguous weekdays and weeks** stay relative to the message date ("next Friday = the Friday after …").
- **Evidence:** the corrections were driven by, and re-checked on, development cases that had been seen. Zero errors
  on those cases is a regression result, not an estimate of reliability on new language.
