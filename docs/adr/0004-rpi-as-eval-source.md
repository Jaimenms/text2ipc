# 0004 Use the Revista da Propriedade Industrial for evals

Status: accepted, 2026-09-09

## Context

Gold labels must come from real examiner decisions and be Portuguese, matching the
first target language. INPI Brazil publishes a weekly zip
(`https://revistas.inpi.gov.br/txt/P<issue>.zip`) with an XML listing every dispatch.

## Decision

Dispatches 1.3 (PCT admitted to national phase) and 3.1 (application published) are
parsed for INID 21 (number), 51 (IPC), 54 (title) and 57 (abstract when present).
Cases are stored as JSONL under `evals/` and committed, since they derive from public
data and are small.

Two file layouts exist: recent issues ship an XML (`Patente_<issue>_<date>.xml`);
issues from around 2011 (e.g. 2100) ship only a Latin-1 text file with one
`(INID) value` per line. Both are parsed. In the text layout the abstract repeats the
title, which is stripped.

## Consequences

- Abstracts exist only in the older text issues (roughly 2100 to 2329, 2011 to 2015);
  everything later is title-only. Cases are tagged so results can be split. 29 issues
  are committed: 2100, 2905 and 27 sampled at random with seed 42 (list in
  `docs/evals.md`).
- Many cases are PCT national-phase entries whose titles are literal translations;
  they are representative of what an INPI user will paste.
- Gold symbols follow the IPC edition printed in the issue (mostly 2006.01 and later),
  not necessarily the version being evaluated. Truncated levels absorb most of the drift.
