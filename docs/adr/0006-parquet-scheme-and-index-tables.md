# 0006 Scheme table and index table as Parquet, texts separate from vectors

Status: accepted, 2026-09-09. Supersedes the CSV layout of ADR 0002.

## Context

The CSV index repeated the rendered text of every entry next to its vector, so the
same 80k texts were stored once per model, the files were 330 to 630 MB, and the text
shown to users came from the model artefact rather than from the scheme.

## Decision

Two tables per IPC version, both Parquet (zstd):

- **Scheme table**, one per version and language:
  `scheme/ipc_<version>_<lang>.parquet` with `symbol, level, depth, parent, title,
  heading, path`. `path` is the `|`-joined chain of canonical symbols from the section
  to the entry, e.g. `A|A01|A01B|A01B0001000000|A01B0001020000|A01B0001040000`.
- **Index table**, one per version, language and model:
  `index/ipc_<version>_<lang>_<model>.parquet` with `symbol, path, text_hash` and the
  vector as float32 columns `e000 .. eNNN`. Version, language, model, dimension, row
  count and build time live in the Parquet schema metadata under the key `text2ipc`.

The text of an entry is rendered from the scheme table by joining the titles along
`path` top-down (section > class > subclass > main group > subgroups), with the
guidance heading placed before a main group that sits under one. **The vector is the
embedding of that full text**, never of the entry's own title alone; `path` records
which symbols contributed. `text_hash` is the hash of the rendered text and drives
vector reuse across versions.

## Consequences

- Index files shrink about 2.8x (PT e5-base: 630 MB CSV to 227 MB Parquet) and load in
  well under a second with pyarrow; `pandas`, DuckDB and Hugging Face `datasets` read
  them directly.
- Titles are stored once per language, so the same scheme table serves every model.
- Reading an index requires its scheme table; `t2ipc download`, `t2ipc manifest` and
  `t2ipc hf-export` always move the pair.
- The text style experiment (lower-casing, dropping levels, child-first) is gone; the
  assembly order is fixed and documented. The measurements that motivated the removal
  are kept in `docs/evals.md`.
- Legacy CSV indexes were converted with `t2ipc migrate-csv`, which verifies that the
  re-rendered text hashes match the ones the vectors were built from.
