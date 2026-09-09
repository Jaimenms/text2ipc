# 0005 Where indexes live, and how they reach users

Status: accepted, 2026-09-09

## Context

An index is 300 to 650 MB and takes 20 to 40 minutes plus a model download to build.
Neither belongs in git, and a PyPI user should not have to build one. While studying,
artefacts next to the code are easier to see, measure and delete.

## Decision

1. **Location.** `TEXT2IPC_HOME` wins. Without it, inside the study repository (a
   `pyproject.toml` naming `text2ipc` plus an existing `data/` directory) the home is
   `./data`, which is gitignored. Anywhere else it is `~/.cache/text2ipc`.
2. **Distribution.** The Hugging Face Hub is the only channel (an S3 bucket was
   planned and dropped on 2026-09-09: the Hub already hosts the same Parquet files
   for the endpoint, with LFS, versioning and checksums). `scripts/publish_hf.sh`
   exports and uploads; `t2ipc download [repo]` copies `index/` and `scheme/` from a
   repository into the local home. Users choose between downloading and building;
   nothing is fetched implicitly.
3. **Same files everywhere.** The CSV contract of ADR 0002 is the interchange format;
   there is no separate "published" format to keep in sync.

## Consequences

- Raw sources stay out of publication: `wipo/` (master XML), `inpi/` (JSON cache),
  `rpi/` (eval issues). Only `manifest.json`, `scheme/*.parquet` and `index/*.parquet`
  are published; they are enough to answer a query.
- Hugging Face: `t2ipc hf-export <dir>` assembles a model repository for **Inference
  Endpoints**: `handler.py` (an `EndpointHandler` taking `{"inputs", "parameters"}` and
  returning JSON matches), the index under `index/`, the package vendored under
  `text2ipc/` so no PyPI release is needed, `text2ipc.json` naming the index, a model
  card with `pipeline_tag: text-classification`, and `.gitattributes` routing the CSV
  through LFS. Published with `hf upload` to `jaimenms/text2ipc-pt` (IPC 20260101, PT,
  multilingual-e5-base) on 2026-09-09. The serverless free API does not run custom
  handlers; a dedicated endpoint does.
- A dataset-repo release is straightforward too: the same CSV (or a Parquet twin
  with the vector as a list column) as a `dataset`, keyed by version, language, model
  and style; the embedder is already a Hub model, so a `model card` would only need
  to point at it and document the query prefix and text style. A `sentence-transformers`
  compatible "retriever" packaging is not needed for the CSV to be usable.
- Anyone republishing must rerun `t2ipc manifest`; a stale manifest fails checksum
  verification on download rather than silently loading the wrong file.
