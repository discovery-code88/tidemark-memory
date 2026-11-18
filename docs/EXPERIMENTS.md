# Experiments

Measurements made while developing Tidemark 1.1 and 1.2 (1.1 was prepared but not published; its changes ship in 1.2). They are recorded with their limits; none of them changes a default
by itself. Raw results, preregistered plans and locked analysis scripts are kept with each run (not in this repository).

## Preference memory on PersonaMem 32k

PersonaMem (32k-token histories): 37 users, 589 multiple-choice questions. Each user's conversation is replayed up to
the point of each question; Tidemark searches (10 records, plus 5 preference slots where preferences are on), and an
answer model picks one of four replies using only the retrieved memories. Harness: `bench/personamem_eval.py`.

### Run 1: extraction modes (gemini-3.8-flash, text-embedding-v4)

| | off | rules | llm |
|---|---|---|---|
| All 589 questions | 76.4% | 77.9% | 76.4% |
| vs off (better / worse) | – | 33 / 24 (sign test p = 0.29) | 37 / 37 (p = 1.0) |

Neither mode demonstrated an overall gain under this retrieval setup; this does not establish that the modes are
equivalent. With llm extraction, preference-aligned recommendations rose (70.9% → 76.4%) and "suggest new ideas" fell
(51.6% → 45.2%). One inspected failure suggests a mechanism: a remembered preference ("prefers potlucks") was taken as
an answer constraint when the user asked for something new. One example does not establish the cause of the
category-wide drop. Whether preferences matter more with longer histories is untested (planned as experiment B).

### Experiment A: label variants (agnes-2.5-flash, free tier; preregistered)

One replay with llm extraction; every question answered three times from the same stored preferences: plain labels,
"kind" labels (what the user already does), and "novelty" labels (kind, plus "not new to them" for requests for
something new). Primary comparison: kind vs plain on the 93 new-idea questions.

| Comparison (93 new-idea questions) | Better / worse | Difference [95% CI] | p (raw / Holm) |
|---|---|---|---|
| kind vs plain (primary) | 10 / 8 | +2.2 pts [−6.5, +10.8] | 0.82 / 1.00 |
| novelty vs plain | 7 / 9 | −2.2 pts [−10.8, +6.5] | 0.80 / 1.00 |
| novelty vs kind | 7 / 11 | −4.3 pts [−12.9, +4.3] | 0.48 / 1.00 |

Inconclusive; effects of up to about 10 points either way are not excluded. On the other 496 questions both label
arms scored below plain (kind −3.6 pts, p = 0.041; novelty −6.2 pts, p = 0.002), but the arms were always answered
in the same order (plain, kind, novelty), a confound; and identical prompts (kind and novelty on unflagged questions)
disagreed on 88 of 474 answers, showing large answer noise. Throttling of later calls is an untested explanation.
Overall accuracy was lower than in run 1; this coincided with the model change and A does not establish the cause.
Extraction: 2,123 preferences proposed, 27.7% rejected at verification (quote not verbatim, unknown record or missing
field), 1,535 stored; A tests labels only on preferences that survive extraction.

### Experiment A2: randomized-order repeat (agnes-2.5-flash, free tier; preregistered)

Four arms per question, all from the same stored preferences: plain, an identical repeat of plain, kind, and novelty.
Arm order was randomized per question (seed 20261102), which breaks systematic alignment between labels and answer
position. Hard caps: 2,600 answer attempts and 450 extraction attempts (retries included), $1 of embeddings.
Primary comparison: kind vs plain on the 496 questions other than new-idea questions; Holm family of four.

The run stopped at the answer-attempt cap: 425–426 of 589 questions answered per arm. Users are replayed in order, so
the missing questions belong to the last users; the completed subset may not represent all 589 questions (new-idea
questions are under-covered: 34 of 93 missing).

| Comparison | Complete pairs | Better / worse | Difference [95% CI] | p (raw / Holm) |
|---|---|---|---|---|
| kind vs plain, other 496 (primary) | 366 of 496 | 37 / 38 | −0.3 pts [−4.9, +4.6] | 1.00 / 1.00 |
| novelty vs plain, other 496 | 366 of 496 | 36 / 39 | −0.8 pts [−5.5, +3.8] | 0.82 / 1.00 |
| kind vs plain, 93 new-idea | 59 of 93 | 9 / 6 | +5.1 pts [−8.5, +18.6] | 0.61 / 1.00 |
| novelty vs plain, 93 new-idea | 59 of 93 | 10 / 7 | +5.1 pts [−8.5, +18.6] | 0.63 / 1.00 |

Conclusions:
- A's regression signal did not reproduce in A2's completed subset. Neither label variant demonstrated a benefit.
- The primary 95% interval spans roughly −5 to +5 points on the observed subset; it does not establish equivalence or
  cover the omitted users.
- Later answer positions had substantially lower accuracy (positions 1–4: 69.5%, 67.4%, 62.4%, 57.9%; position 4 vs 1:
  better on 32, worse on 81, p < 0.001). Mean attempts did not differ by position (about 1.5 at each). This makes order a
  plausible explanation for A's apparent regression, but the mechanism remains unknown.
- Identical prompts produced different answers 23% of the time (plain vs its repeat: 98 of 425, 95% CI 19–27%).

Decision: preference extraction stays off by default and labels default to `plain`; `kind` and `novelty` remain
options. Experiment B (longer histories) and the rules-cleanup rerun remain deferred.

## Follow-ups on change-over-time cases (2026-11-12; retrieval only, no model or paid embedding calls)

**Question:** when Search returns a statement, can it also return what the user said later about the same thing?

**Data:** 21 author-made cases, 51 questions about facts that change (stop, resume, replace, exception, temporary,
future-dated, correction), each inside a 344-message host conversation; top 10; offline placeholder embedder for both
backends. The settings below were chosen on these same cases, so the numbers are development results, not a test.

| Setting | Complete evidence sets (engine-first / hybrid) | Displaced corrections | Lines attached | of which the case's statement |
|---|---|---|---|---|
| off (default; identical to before in all 204 searches) | 28 / 37 of 51 | 6 / 2 | 0 | 0 |
| `TIDEMARK_ASSEMBLY=on` (the earlier opt-in) | 26 / 34 | 4 / 4 | 143 / 135 records swapped in | key passages pushed out in 10 / 12 of 102 searches |
| `followups=3` (defaults) | 31 / 38 | 4 / 1 | 9 / 12 | 3 / 1 |
| `followups=3`, scope `any` | 32 / 38 | 4 / 1 | 15 / 16 | 7 / 1 |
| `followups=3`, `min_terms=2` | 31 / 38 | 4 / 1 | 22 / 29 | 5 / 3 |
| `followups=3`, `min_terms=2`, `anchor_share=0` | 34 / 38 | 2 / 1 | 100 / 113 | 8 / 3 |

- Follow-ups never changed which records were returned or their order, and never removed a key passage.
- The gain is small: three more complete sets engine-first (running stopped; yoga resumed, two questions), one hybrid.
- Most attached lines are still unrelated: a shared word used in another sense ("host" a dinner / podcast "hosts").
- Not reached by any setting: changes that share no distinctive word with the original ("never on work nights",
  "back on my bike", "moving to Bordeaux"), and replacements linked only by words the question doesn't name unless
  `min_terms` is set ("green tea" → "black coffee").
- Evidence assembly made these cases worse and should not be used for this purpose.

Decision: follow-ups stay experimental and off by default. Not measured: answers, production embeddings, real
conversations, histories larger than about 90 records.

## When a statement applies (2026-11-12; no model or paid embedding calls)

**Question:** can Search say when a statement holds, not only when it was said?

**Data:** the same 21 cases; 10 of their 39 key passages have a gold period (`effective` in the frozen cases). The
rules were written with these cases in view, so this is a development check, not a test.

| | Count |
|---|---|
| Passages with a gold period | 10 |
| Annotation compatible with gold | 6 |
| — of which the full gold period | 4 |
| — of which partial | 2 |
| Annotation contradicting gold | 0 |
| No annotation | 4 |
| Passages without a gold period that got an annotation | 0 of 29 |
| Host-conversation sentences that got one (344 messages) | 0 |

- **Full (4):** "since the start of August" (from 1 August), "for all of October" (1 to 31 October), "From next
  month" (from 1 September), "Starting May 1st" (from 1 May).
- **Partial (2):**
  - "Doing dry January, not a drop of alcohol until February": the end only, with its boundary left open (certain
    to 31 January, possibly to 28 February); no start. Gold is 1 to 31 January.
  - "My last day at the bank will be June 30": the end of the job (30 June). Gold records the next state, retired
    from 1 July, which the annotation does not state.
- **None (4), by design:** event times ("quit … last month", "last week", "yesterday"; these keep their time notes)
  and the bare duration "frozen … for three months", which can look backwards or forwards.
- With `applies` off, the same searches return the same records and no `applies` field.
- The host conversation contains no explicit period forms, so the zero there shows the rules did not fire on
  ordinary sentences; it says nothing about precision on text that does contain such forms.

**Boundary semantics** (`src/applies.rs`):
- `from` is the first day the statement applies; `until` is the last day it certainly applies.
- "until <time>" does not say whether the named time is included. `until` is the unit before it and `until_latest`
  the named time; the note gives both ("through January 2023 (or through February 2023, if the named time is
  included)").
- A bound keeps the precision of its words: day, month or year. No day is invented for "since January 2025".
- "through …", "until the end of …", "last day … is …" and whole periods include the named time.
- One stated bound stays one bound; the other is not filled in from the record's date.
- A month without a year is read as its nearest occurrence ("since" reads backwards) and flagged `year_assumed`.

**Duration time notes corrected (default behaviour, new Adds only):** 1.0.0 annotated "I've frozen my gym
membership for three months" as `for three months = since about March 2023`. From 1.2 a duration gets a start date
only when its sentence clearly looks back ("have/has been … for", "for … now"); forward, ambiguous, future,
hypothetical and past-perfect wording get no note. Stored records are not rewritten. Cost: retrospective sentences
without "been" or "now" ("I've lived here for three years") lose their note.

Decision: `TIDEMARK_APPLIES` stays experimental and off by default. Not built: storing a period so it can be
corrected. Not measured: whether answers improve.

**Next check for applies:** a set of temporal examples and counterexamples written independently of these rules and
frozen before the rules are run on it.

## Independent temporal set (2026-11-12; no model calls)

**Set:** 60 English cases written without sight of the implementation, frozen by hash
(`temporal-validation-frozen/`). AI-authored; scored by the assistant that wrote the rules; no human adjudication.
Seven cases could not be scored as issued (their gold assumes another utterance time) and were reissued separately
as an erratum (`temporal-validation-frozen-v2-erratum/`).

**First run, rules unchanged** (53 scorable cases; clear / ambiguity-flagged):

| Category | Clear (42) | Ambiguity-flagged (11) |
|---|---|---|
| Complete / compatible | 10 | 4 |
| Compatible partial | 1 | 2 |
| Incorrect | 5 | 1 |
| Abstention, gold has a datable component | 8 | 0 |
| Abstention, date explicit in the text | 4 | 0 |
| Abstention, nothing to date | 14 | 4 |

`applies` made 9 annotations and 5 were incorrect: a stated year dropped after a comma or at the end of a range
(2), a day invented for a month or year start (2), and "six weeks from today" read as a start (1). The sixth
incorrect annotation is a default time note on quoted fiction.

**Decision after the first run:** `applies` is rejected for enabling and stays off.

**Second run, after fixing those three defects** (development regression on material already seen): clear cases
13 complete, 2 partial, 1 incorrect; ambiguity-flagged 4 compatible, 3 partial, 0 incorrect; abstentions unchanged.
`applies`: 8 annotations, 0 incorrect, 2 partial. The remaining incorrect case is the quoted-fiction time note.

**Erratum cases (7), apart:** 1 complete, 4 abstentions, 2 incorrect. The two incorrect ones ("yesterday",
"tomorrow" near midnight with a UTC offset) come from the interface: Add receives an instant without an offset and
dates it in UTC. Their timestamps are reconstructed.

**Quotation counterexamples (16, written by the assistant):** the default time notes annotate every relative
expression whoever said it: 5 of 5 where they should, 8 of 8 where they should not, 3 of 3 ambiguous. No rule was
changed.

**Kept as is:** the conservative duration rule. "I've lived here for three years" and "I've frozen my membership
for three months" need different readings; restoring approximate onsets is a separate candidate.

The scoring records of both runs (outputs and locks) are kept with the development records, not in this repository.

## Quotation guard and UTC offset (2026-11-12; development checks, no model calls)

All material here had been seen before the changes; these are regressions, not validation.

**Quotation guard for time notes** (default, new Adds only). Counterexamples written by the assistant (16):

| Expectation | Cases | Note before | Note after |
|---|---|---|---|
| Should annotate | 5 | 5 | 5 |
| Should not (fiction, lyric, third party, assistant-origin, example sentence) | 8 | 8 | 0 inside the quotation; "last week" outside one quotation is kept |
| Ambiguous | 3 | 3 | 2: an undated self-quotation loses its note; "I just told Ana, '…yesterday'" and unquoted reported speech keep theirs |

On the independent set the only change is IT-032 (quoted fiction): its note is gone, so no incorrect annotation
remains among the 53 originally scorable cases (clear: 13 complete, 2 partial, 0 incorrect, and one more abstention
with nothing to date). The guard is not a reading of reported speech: fiction or reports without quotation marks
are resolved as before.

**`utc_offset_minutes`** (optional, per message). Erratum cases, reported apart; timestamps of IT-045 and IT-046 are
chosen reconstructions:

| Case | No offset sent (as before) | Offset sent |
|---|---|---|
| IT-045 "I arrived yesterday", 00:30 at UTC+02:00 | `yesterday = 30 December 2025` | `yesterday = 31 December 2025` (gold's local day) |
| IT-046 "I'll arrive tomorrow", 23:30 at UTC−05:00 | `tomorrow = 2 January 2027` | `tomorrow = 1 January 2027` (gold's local day) |

The earlier result on these two was an interface limitation, not a parser error: Add received an instant without
an offset, and the parser cannot recover what was not sent. Without the field the outputs are byte-identical to the
previous run. A fixed offset is not a time zone.

## Temporal-parser round closed (2026-11-12)

- Release checks before closing (local, no model calls): 19 of 19 integration checks passed: each message of a mixed-offset request uses its own offset;
  Search-time `applies` reads the same local day as the Add-time note; a follow-up carries only its own sentence
  and that sentence's notes; an out-of-range offset is rejected and stores nothing.
- Two scoping defects found by those checks and fixed: a note for a phrase that also occurs inside a quotation
  that does not count is no longer written (a note names a phrase, not a place in the message); follow-ups no
  longer read Add's time notes as part of the user's sentence.
- All earlier outputs reproduce byte for byte with the final build, and default Search results on the 21
  change-over-time cases are identical to the build before this round (204 of 204 searches).
- `applies` stays off and rejected for enabling. Duration and time-unit coverage is not expanded. Remaining gaps are
  listed in `docs/MIGRATION-1.2.md`.
- Zero errors on exposed cases is a regression result, not an estimate of reliability on new language.

## Local reranker on change-over-time and ordinary questions (2026-11-12; no model API or paid calls)

**Question:** does the existing reranking option, with a free local cross-encoder, return later changes together
with the statements they change?

**Setup:** `TIDEMARK_RERANK_DIR` = BAAI/bge-reranker-base (ONNX, CPU), `rerank_scope=all`; default backend
(engine-first) and hybrid; offline placeholder embedder for both; top 10. The 21 change cases are author-made and
have been used for development all along; the reranker settings were compared on them, so this is a development
result.

Change cases, 51 questions (engine-first / hybrid):

| Reranker | Complete evidence sets | Old state without its correction | Median search time (6 CPU cores) |
|---|---|---|---|
| off (default) | 28 / 37 | 6 / 2 | under 0.1 s |
| 15 candidates | 34 / 40 | 3 / 2 | 1.0 s |
| 30 candidates | 42 / 42 | 2 / 3 | 2.0 s |
| 50 candidates | 42 / 41 | 2 / 4 | 3.3 s |
| 30 candidates, `rerank_scope=hybrid` | 28 / 42 | 6 / 3 | 1.9 s |

- Engine-first by type at 30 candidates: correction 6/6, temporary 6/6, future 6/6, stop 8/9, resume 7/9,
  replace 5/9 (1/9 without), exception 4/6.
- It changed the results of every search and lost the complete set in 3 of 102 engine-first searches, all as-of
  questions where the older statement dropped out ("What was my morning routine in April?").
- With `rerank_scope=hybrid` the engine's results stay in front, so engine-first is unchanged: the gain needs
  `all`.
- Still incomplete at 30 (9 questions): changes with no shared wording ("moving to Bordeaux", "never on work
  nights", bus → bike) and two as-of questions.
- Follow-ups on top of the reranker added no relevant line (10 attached, 0 relevant).

Ordinary questions, LoCoMo evidence recall at top 10 (conv-26 and conv-30, 301 questions, replayed as AML does, same
stored records, engine-first):

| | all evidence found | per evidence turn | p50 search |
|---|---|---|---|
| reranker off | 76.4 % | 74.7 % | 3 ms |
| 30 candidates, scope all | 82.1 % | 80.5 % | 2.6 s |

By category (all evidence found, off → on): multi-hop 23.8 → 38.1 %, temporal 96.8 → 96.8 %, inference 27.3 → 27.3 %,
single-hop 81.6 → 89.5 %, adversarial 88.7 → 91.5 %.

**Not measured:** answers; production embeddings; top_k of 100 as AML uses (reranking 30 candidates reorders the
head of the list, so it matters less there); other reranker models; a GPU.

**A sentence-level judge was tried and dropped.** The same cross-encoder, scoring candidate follow-up sentences
against the question, did not separate relevant from unrelated ones: "I've stopped running altogether…" scored −7.8
for "Do I still go running in the mornings?", below hundreds of unrelated sentences. The code was removed.

**Decision:** nothing changes by default. For deployments that return few results and can afford about 2 s per
search on CPU, `TIDEMARK_RERANK_DIR` with `TIDEMARK_RERANK_CANDIDATES=30` and `TIDEMARK_RERANK_SCOPE=all` is the
measured way to get later changes into the results; follow-ups are not needed with it.

## Reranker at 100 results (2026-11-13; protocol fixed before the run; no model or paid calls)

Same fixed reranker (bge-reranker-base, 30 candidates, scope all) against off, with the live service's settings
and query shape (`top_k` 100), placeholder embedder, private ports. Full LoCoMo (10 conversations, 1,986 questions)
and the 51 change questions.

- **Selection:** the same set of records in every search (1,986 of 1,986; 51 of 51). Only positions 1–30 are
  reordered. Tidemark truncates nothing; the text returned is identical.
- **Evidence, LoCoMo, all gold evidence found:** first 10: 74.1 → 82.2 % (190 questions gained, 29 lost); first 25:
  85.1 → 86.2 %; first 50 and first 100: unchanged (90.3 %, 96.6 %).
- **Evidence, change questions within the first 10:** current 16 → 24 of 27, historical 7 → 10 of 12, as-of 3 → 6
  of 9, scoped 2 → 2 of 3. One as-of question loses its older statement from the first 10 (rank 5 → 11). Within
  the first 25 and beyond nothing is lost and almost nothing gained.
- **Latency:** sequential p50 4 ms → 2.7 s (p95 3.3 s). The reranker serves one search at a time: 0.36 searches
  per second at any concurrency; p50 10.9 s at 4 concurrent searches and 22.0 s at 8.

**Reading:** at 100 results the reranker changes order, not content. Its benefit therefore depends on the consumer
being sensitive to order or cutting the list short, which this check cannot show, and its throughput is far below
the bursts in the live log. Not enabled; the next gate is a capped answer comparison through the consuming pipeline.
The full results are kept with the development records, not in this repository.

