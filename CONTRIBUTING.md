# Contributing

Issues and pull requests are welcome.

**Build and test**
```bash
cargo test --release                              # service
(cd engine && cargo test --release --lib)          # memory engine
TIDEMARK_API_KEYS=dev TIDEMARK_DB=/tmp/dev.sqlite ./target/release/tidemark &
python3 bench/conformance.py http://127.0.0.1:18200 dev
python3 bench/api_check.py  http://127.0.0.1:18200 dev
```
Without an embedding key the server uses offline placeholder embeddings, so these checks need no account.

**Changes to ranking or stored text** should come with a measurement: a recall run (`bench/*_recall.py`) before and
after, on the same data. `docs/REPORT.md` records how earlier decisions were measured, including ideas that did not
help; please add yours there.

**Style:** keep behaviour configurable (a setting, overridable per request where it makes sense) rather than
hard-wired, and keep defaults unchanged unless a measurement supports the change.

By contributing you agree that your contribution is licensed under the Apache License 2.0.
