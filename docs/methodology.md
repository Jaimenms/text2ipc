# Methodology

## Problem

Given a text, return IPC symbols ranked by how well they describe it, at a requested
level of the hierarchy (section, class, subclass, main group, subgroup) or at the
deepest level the evidence supports.

## 1. The scheme as a set of texts

WIPO's `ipc_scheme_<version>.xml` nests entries physically. Only five kinds are
classification targets: sections, classes, subclasses, main groups and subgroups
(dot levels 1 to 9). Guidance headings, notes, indexes and subsection titles are not.

An entry's title alone is not enough to embed. A subgroup reads "with teeth"; only the
path makes it "Hand tools > Spades; Shovels > with teeth". So the embedded text of an
entry is the concatenation of every ancestor title, the guidance heading it sits under
(if any), and its own title, top-down and joined with `>`.

Cross references inside titles ("edge trimmers A01G 3/06") are dropped: they describe
what belongs *elsewhere*, and embedding them attracts precisely the wrong texts.

Subgroups nest to nine dot levels, so the parent of a subgroup is usually another
subgroup, not the main group. The path is walked parent by parent from the entry up to
the section; nothing assumes a fixed number of levels.

## 2. Two tables per version: scheme and index

The parsed scheme is written once per version and language as a Parquet **scheme
table** (`symbol, level, depth, parent, title, heading, path`); `path` is the chain
of symbols from the section to the entry. The text of an entry is rendered from it by
joining the titles along the path, top-down, and that full text is what gets embedded.
Vectors go to a separate Parquet **index table** per model (`symbol, path, text_hash,
e000..eNNN`), so titles are stored once and every model reuses them (ADR 0006). Rows
keep the scheme's depth-first order, so a parent always precedes its children; the
search code exploits that to aggregate scores in one pass in each direction.

Assembly order matters less than completeness: lower-casing the scheme titles or
dropping the section and class titles were both measured to hurt (`docs/evals.md`),
so the full path in its original case is the only assembly.

When a new IPC version is built, entries whose text hash matches the previous index for
the same language and model reuse the stored vector. Because the text includes the
ancestor path, a renamed group invalidates its whole subtree, which is what you want.

## 3. Scoring a query

Flat cosine similarity between the query and all 80k entry vectors is computed first;
it is cheap. Three heuristics then reshape it:

| Component | Definition | What it fixes |
|---|---|---|
| own | cosine(query, entry) | the baseline |
| path support | mean own-similarity of the entry's ancestors | an isolated hit under an unrelated subclass |
| subtree support | max own-similarity over the entry and its descendants | a class whose title is generic but whose children match |

`score = 0.6 * own + 0.15 * path + 0.25 * subtree` (all three are `Weights` fields).

**Beam descent.** Candidates at a level are restricted to children of the best
parents at the level above (defaults: 3 sections, 6 classes, 10 subclasses, 25 main
groups). This removes the specificity bias, where long specific subgroup texts win on
word overlap alone, and it walks the tree rather than ranking a flat list.

**Auto level.** Starting from the top subclasses, descend while the best child's score
is within `auto_margin` of its parent. The walk stops where the evidence stops.

**Gap.** Optionally drop results more than `gap` below the best score, giving a
multi-label answer whose length depends on confidence.

**Distinct branches.** The top-k results are then walked in rank order and a result
is discarded when a better-ranked one is its ancestor or its descendant. Nested
subgroups are common (up to nine dot levels), and a subgroup inside the best answer
adds nothing that its path does not already show. The list keeps only distinct
branches and may hold fewer than k rows (`dedupe_branches=False` restores the raw
top-k).

## 4. Languages

The query language and the scheme language are independent. The default embedder is
multilingual (`intfloat/multilingual-e5-base`), so a Portuguese abstract is matched
against the English scheme cross-lingually. Titles printed in capitals are lower-cased
before embedding; that alone lifted subclass recall by ten points. WIPO ships EN and FR master files. Portuguese comes from the translation WIPO bridges
to, INPI Brazil's IPCPUB, fetched as JSON and applied as a title overlay on the English
hierarchy; any other translation can be supplied as a `symbol,title` CSV (ADR 0003).

## 5. Evaluation

INPI Brazil's weekly *Revista da Propriedade Industrial* lists, for every published
application (dispatches 1.3 and 3.1), the Portuguese title, the abstract when the issue
includes it, and the IPC symbols assigned by examiners. `t2ipc rpi <issue>` turns that
into JSONL cases; `t2ipc eval` reports hit@k at every level, hit@1 on the office's main
symbol, and MRR. Multi-label gold sets count a hit if any gold symbol is found.

Recent XML issues (2905, 2026-09-08) carry titles only; older text issues (2100,
2011) carry abstracts. Both are in `evals/`. Current numbers are in `docs/evals.md`:
with the default model, the correct subclass is first about one time in five and
within the top ten about two times in five. The model, not the hierarchy heuristics,
is the limiting factor; a stronger multilingual embedder or an LLM rerank over the
surviving candidates is the obvious next step.
