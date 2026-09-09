# 0001 Embed the full ancestor path, not the entry title

Status: accepted, 2026-09-09

## Context

IPC titles are written to be read in context. Subgroup titles are fragments ("with
teeth", "characterised by the shape of the rotor") and even main-group titles assume
the subclass title above them.

## Decision

The embedded text of an entry is `ancestor titles > guidance heading > own title`,
top-down. Cross references inside titles are stripped. Notes and catchword indexes are
not embedded.

## Consequences

- Vectors are self-contained and comparable across levels.
- A title change anywhere invalidates the vectors of the whole subtree below it; the
  incremental build handles this by hashing the full text.
- Texts get long (average ~390 characters, max ~1500). Models with short context
  windows truncate the tail, which is the most specific part. Prefer models with at
  least 512 tokens.
