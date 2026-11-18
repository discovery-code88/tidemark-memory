#!/usr/bin/env python3
"""PersonaMem (32k) through Tidemark: replays each user's conversation up to the
point of each question, searches, and lets an answer model pick one of the
options from the retrieved memories only. Measures how well memories (and
preferences) serve personalization.

usage: personamem_eval.py --url URL --key KEY --label NAME --out FILE.jsonl
                          [--data DIR] [--contexts N] [--top-k 10] [--preference-slots 5]
Answer model (OpenAI-compatible chat): EVAL_ANSWER_URL, EVAL_ANSWER_MODEL, EVAL_ANSWER_API_KEY.
Resumable: questions already in --out are skipped (their users are replayed again only up to the next question).
"""
import argparse, ast, csv, json, os, random, re, sys, time, urllib.error, urllib.request

p = argparse.ArgumentParser()
p.add_argument("--url", required=True)
p.add_argument("--key", required=True)
p.add_argument("--label", required=True)
p.add_argument("--out", required=True)
p.add_argument("--data", default="personamem", help="folder with the PersonaMem files (default: ./personamem)")
p.add_argument("--contexts", type=int, default=0, help="first N users only (0 = all 37)")
p.add_argument("--top-k", type=int, default=10, help="records per question (preferences are extra)")
p.add_argument("--preference-slots", type=int, default=0)
p.add_argument("--wait", type=int, default=1800, help="seconds to wait for queued preference extraction")
p.add_argument("--variants", default="", help="comma-separated arms NAME or NAME=LABELS (LABELS = plain|kind|novelty, default NAME); each question is answered once per arm from the same memory; results go to OUT with .NAME before .jsonl. Repeating LABELS under another name gives an identical-prompt control, e.g. plain,plain_repeat=plain")
p.add_argument("--shuffle-seed", type=int, default=None, help="answer the arms of each question in a random order (seeded per question); the position is recorded")
p.add_argument("--max-answer-calls", type=int, default=0, help="stop (exit 3) after this many answer calls in this session (0 = no cap)")
p.add_argument("--attempt-ledger", default="", help="file with one line per answer attempt (retries included), kept across sessions")
p.add_argument("--max-attempts", type=int, default=0, help="hard cap on answer attempts recorded in --attempt-ledger; at the cap the harness stops (exit 3)")
p.add_argument("--answer-retries", type=int, default=8, help="attempts per answer call; when reached, the answer is left missing (counted) for a later resume")
p.add_argument("--types", default="", help="comma-separated question types to ask (others are skipped; users are still replayed)")
p.add_argument("--part", default="0/1", help="i/n: only every n-th user starting at i (to split a run over parallel harnesses)")
p.add_argument("--done-from", action="append", default=[], help="other result files whose questions count as done")
args = p.parse_args()

H = {"Authorization": f"Token {args.key}", "Content-Type": "application/json", "User-Agent": "tidemark-bench/1.0"}
A_URL, A_MODEL, A_KEY = (os.environ.get(k, "") for k in ("EVAL_ANSWER_URL", "EVAL_ANSWER_MODEL", "EVAL_ANSWER_API_KEY"))
if not (A_URL and A_MODEL):
    sys.exit("set EVAL_ANSWER_URL and EVAL_ANSWER_MODEL")


def tm(method, path, body=None):
    req = urllib.request.Request(args.url.rstrip("/") + path, data=json.dumps(body).encode() if body is not None else None, headers=H, method=method)
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.loads(r.read() or b"null")
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < 7:
                time.sleep(2 ** attempt)
                continue
            raise
        except (urllib.error.URLError, OSError):  # connection dropped or refused
            if attempt == 7:
                raise
            time.sleep(min(60, 2 ** attempt))


usage = {"calls": 0, "input": 0, "output": 0, "retry_limit": 0, "attempts": 0}
last = {"attempts": 0, "latency": 0.0}


def answer(prompt):
    url = A_URL.rstrip("/") + ("" if A_URL.rstrip("/").endswith("/chat/completions") else "/chat/completions")
    body = {"model": A_MODEL, "temperature": 0, "messages": [
        {"role": "system", "content": "You are a personal assistant. Use only the memories given about this user. Reply with the letter of the best option only, e.g. (b)."},
        {"role": "user", "content": prompt}]}
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {A_KEY}"} if A_KEY else {})}
    t0 = time.time()
    for attempt in range(args.answer_retries):
        if args.attempt_ledger:
            n = sum(1 for _ in open(args.attempt_ledger)) if os.path.exists(args.attempt_ledger) else 0
            if args.max_attempts and n >= args.max_attempts:
                print(f"CAP reached: {n} answer attempts (cap {args.max_attempts})", flush=True)
                sys.exit(3)
            with open(args.attempt_ledger, "a") as ledger:
                ledger.write(f"{time.time():.0f}\n")
        last["attempts"] = attempt + 1
        usage["attempts"] += 1
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.loads(r.read())
            u = d.get("usage") or {}
            usage["calls"] += 1
            usage["input"] += u.get("prompt_tokens", 0)
            usage["output"] += u.get("completion_tokens", 0)
            last["latency"] = time.time() - t0
            return d["choices"][0]["message"]["content"] or ""
        except (urllib.error.URLError, OSError, KeyError) as e:  # incl. timeouts and dropped connections
            code = getattr(e, "code", 0)
            if code in (400, 401, 403, 404):
                raise
            # Rate limits (free tiers) can last minutes: wait longer for them.
            time.sleep(min(300, (5 if code == 429 else 1) * 2 ** attempt))
    usage["retry_limit"] += 1
    print(f"RETRY_LIMIT reached after {args.answer_retries} attempts", flush=True)
    return None


ctx = {}
for line in open(os.path.join(args.data, "personamem_shared_contexts_32k.jsonl")):
    ctx.update(json.loads(line))
rows = list(csv.DictReader(open(os.path.join(args.data, "personamem_questions_32k.csv"))))
order = list(dict.fromkeys(r["shared_context_id"] for r in rows))
if args.contexts:
    order = order[: args.contexts]
part, parts = (int(x) for x in args.part.split("/"))
order = order[part::parts]
arms = [a for a in args.variants.split(",") if a]
labels_of = {a.split("=")[0]: (a.split("=", 1)[1] if "=" in a else a) for a in arms}
variants = list(labels_of) or [None]
outs = {v: (args.out if v is None else args.out.replace(".jsonl", f".{v}.jsonl")) for v in variants}
done = set()
# A question counts as done only when every variant has answered it.
per = [{json.loads(l)["question_id"] for l in open(f)} if os.path.exists(f) else set() for f in outs.values()]
done = set.intersection(*per) if per else set()
for f in ([args.out] if variants == [None] else []) + args.done_from:
    if os.path.exists(f):
        done |= {json.loads(l)["question_id"] for l in open(f)}
files = {v: open(f, "a") for v, f in outs.items()}
answered = {v: p for v, p in zip(outs, per)}
T0 = 1683554160000

for cid in order:
    user = f"pm:{args.label}:{cid[:16]}"
    messages = ctx[cid]
    questions = sorted((r for r in rows if r["shared_context_id"] == cid), key=lambda r: int(r["end_index_in_shared_context"]))
    if all(q["question_id"] in done for q in questions):
        continue
    fed = 0
    for q in questions:
        end = int(q["end_index_in_shared_context"]) + 1
        if end > fed:
            batch = []
            for i in range(fed, end):
                m = messages[i]
                if m["role"] not in ("user", "assistant"):
                    continue  # persona descriptions are not part of the conversation
                text = re.sub(r"^(User|Assistant):\s*", "", m["content"].strip())
                batch.append({"role": m["role"], "timestamp": T0 + i * 60000, "content": text})
            for j in range(0, len(batch), 20):
                tm("POST", "/add", {"request_id": f"{cid}:{fed}:{j}", "user_id": user, "session_id": cid[:16], "messages": batch[j:j + 20]})
            fed = end
        if q["question_id"] in done or (args.types and q["question_type"] not in args.types.split(",")):
            continue
        # Model extraction runs after Add: wait until this user's queue is empty.
        t = time.time()
        while time.time() - t < args.wait and tm("GET", f"/preferences?user_id={urllib.request.quote(user)}").get("pending_records", 0):
            time.sleep(2)
        order = list(variants)
        if args.shuffle_seed is not None:
            random.Random(f"{args.shuffle_seed}:{q['question_id']}").shuffle(order)
        for position, v in enumerate(order, 1):
            if q["question_id"] in answered[v]:
                continue
            if args.max_answer_calls and usage["calls"] + usage["retry_limit"] >= args.max_answer_calls:
                print(f"CAP reached: {args.max_answer_calls} answer calls", flush=True)
                sys.exit(3)
            # Preferences come in addition to the --top-k records, so every run sees as many records.
            body = {"query": q["user_question_or_message"], "user_id": user, "top_k": args.top_k + args.preference_slots, "preference_slots": args.preference_slots}
            if v:
                body["preference_labels"] = labels_of[v]
            hits = tm("POST", "/search", body)["data"]
            memories = "\n\n".join(f"[{i + 1}] {h['content']}" for i, h in enumerate(hits))
            options = ast.literal_eval(q["all_options"]) if q["all_options"].startswith("[") else q["all_options"]
            prompt = f"Memories about the user:\n{memories}\n\nThe user now says: {q['user_question_or_message']}\n\nWhich reply fits this user best?\n" + "\n".join(options if isinstance(options, list) else [options])
            reply = answer(prompt)
            if reply is None:
                continue  # left missing; a later run answers it
            picked = (re.search(r"\(([a-d])\)", reply) or re.search(r"\b([a-d])\b", reply.lower()) or [None, ""])[1]
            rec = {"question_id": q["question_id"], "type": q["question_type"], "context": cid, "picked": f"({picked})", "correct": q["correct_answer"].strip(), "variant": v,
                   "ok": f"({picked})" == q["correct_answer"].strip(), "position": position, "attempts": last["attempts"], "latency_s": round(last["latency"], 2), "preferences_shown": sum(1 for h in hits if h.get("kind") == "preference"), "reply": reply[:200]}
            files[v].write(json.dumps(rec) + "\n")
            files[v].flush()
        print(f"{args.label} {q['question_id'][:12]} {q['question_type']:45s}  answers={usage['calls']} retry_limit={usage['retry_limit']}  tokens in={usage['input']} out={usage['output']}", flush=True)

print(json.dumps({"label": args.label, "answer_usage": usage}))
