# 0003 Scheme language and query language are independent

Status: accepted, 2026-09-09

## Context

Queries will be Portuguese first, other languages later. WIPO publishes authentic
master files in English and French only; for other languages its IPC Publication
bridges to national offices running the same IPCPUB software. INPI Brazil serves the
Portuguese scheme that way (`ipc.inpi.gov.br`), as per-subclass JSON rather than a
master XML.

## Decision

- The scheme language is a build parameter (`--lang EN|FR`) mapped to the WIPO file.
- Any other language is applied as a `symbol,title` overlay on top of the English
  hierarchy. Missing rows keep the English title, so partial translations work.
- Portuguese is built in: `--lang PT` pulls the translation WIPO's IPC Publication
  "bridges" to, served by INPI Brazil's IPCPUB instance as static JSON (one file per
  subclass plus an index). The titles are cached as a CSV under the text2ipc home. Any
  other translation can be supplied by hand with `--titles-csv`.
- Query language is handled by the embedder. The default is a multilingual model so a
  Portuguese query is matched against the English scheme cross-lingually.

## Consequences

- No Portuguese scheme is needed to start; cross-lingual retrieval is the baseline.
- A Portuguese overlay, once obtained, can be evaluated against that baseline on the
  same RPI cases to decide whether it pays for itself.
- Index files are keyed by scheme language, and the overlay path is recorded in the
  meta sidecar.
