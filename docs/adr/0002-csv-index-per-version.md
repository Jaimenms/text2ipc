# 0002 One CSV per (IPC version, scheme language, embedding model)

Status: superseded by ADR 0006 (Parquet, texts separate from vectors), 2026-09-09

## Context

Indexes must be inspectable, diffable across versions and reusable when a new IPC
version appears. Users asked for plain CSV.

## Decision

`ipc_<version>_<lang>_<model-slug>.csv` with columns
`symbol, level, depth, parent, title, text, text_hash, embedding`, where `embedding`
is a space-separated list of floats printed with six decimals. A `.meta.json` sidecar
records version, language, model, dimension, row count and build time. Rows are in
scheme depth-first order.

Building version N looks for the newest built index of the same language and model
with an older version and copies vectors whose `text_hash` is unchanged.

## Consequences

- A 384-dimensional index is ~250 MB as CSV. Acceptable for a study; a binary sidecar
  (`.npy`) can be added later without changing the CSV contract.
- Reuse is exact (bit-identical copies), so two builds of the same version from
  different starting points are identical apart from `built_at`.
- Different models never share files; the model name is part of the path and the meta.
