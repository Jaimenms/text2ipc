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

`score = own_w * own + path_w * path + subtree_w * subtree` (the `Weights` fields).
The defaults are `1.0, 0.0, 0.0`: on the Portuguese scheme, path support lowers every
hit rate for both e5 models, because section and class titles are generic and sit
close to every query, so a branch with bland ancestors (A62C, fire fighting) collects
support for texts about anything (`docs/evals.md`, 2026-10-08). It helped on the
English scheme with e5-small, which is why it stays available.

**Beam descent.** Candidates at a level are restricted to children of the best
parents at the level above (defaults: 5 sections, 10 classes, 20 subclasses, 40 main
groups), ranked by the best score in their subtree. Measured on the Portuguese
scheme the width makes no difference between the defaults and no beam at all; it is
kept because it walks the tree rather than ranking a flat list and costs nothing.

**Auto level.** Starting from the top subclasses, descend while the best child's score
is within `auto_margin` of its parent. The walk stops where the evidence stops. Each
starting subclass yields at most one result, so the number of starts (`auto_roots`)
follows `top_k` (at least 5) unless set explicitly.

**Gap.** Optionally drop results more than `gap` below the best score, giving a
multi-label answer whose length depends on confidence.

**Distinct branches.** The top-k results are then walked in rank order and a result
is discarded when a better-ranked one is its ancestor or its descendant. Nested
subgroups are common (up to nine dot levels), and a subgroup inside the best answer
adds nothing that its path does not already show. The list keeps only distinct
branches and may hold fewer than k rows (`dedupe_branches=False` restores the raw
top-k).

### Long texts

Paragraphs (blank-line separated) are embedded separately: on 500 INPI cases the
mean of the title vector and the abstract vector gives subclass@1 26.6% against
22.0% for the abstract alone and 22.0% for the two concatenated (`docs/evals.md`),
so a title above an abstract is two pieces, not one string. The embedders also have
a token limit (512 for the e5 family) and truncate silently beyond it, so a whole
description would be judged by its first page: a paragraph over the limit is cut at
sentence ends into chunks that fit (`chunking.split_text`, one sentence of overlap).
Every piece is embedded as a query and the vectors are combined
(`IpcClassifier.classify(..., chunking=...)`):

| `chunking` | Query | Use |
|---|---|---|
| `mean` (default) | unit-length mean of the chunk vectors | one topic spread over many pages |
| `max` | all chunk vectors; an entry scores by its best chunk | a text that covers several inventions |
| `truncate` | the embedder's own cut | the old behaviour, for comparison |

A single paragraph within the limit is embedded whole. Query normalisation
(whitespace, lower-casing of shouting text) runs per paragraph. The browser demo
applies the same split with the model's tokenizer (`splitText` in `scorer.js`) and
the `mean` policy.

### Reranking

The first stage orders candidates badly more than it misses them: with the top 50
at the subgroup level the office's subclass is in the list 64% of the time but
first only 24% of the time (`docs/evals.md`). A second stage therefore judges each
candidate against the text with a cross-encoder, a model that reads the pair
together instead of comparing two vectors: `rerank=True` keeps
`candidates = max(5 * top_k, 25)` entries from the first stage, scores every
(text, path text) pair with `BAAI/bge-reranker-v2-m3` (568M parameters,
multilingual), and fuses the two scores:

`score = cosine * (0.5 + 0.5 * sigmoid(logit))`

so the judge can at most halve a score, cosine still orders ties, and reranked
scores stay comparable with plain ones. `fusion="judge"` and `"product"` are the
alternatives measured; `Match.judge` carries the verdict in 0..1. On 991 INPI
cases the default lifts subclass@1 from 23.6% to 31.6% and group@1 from 11.2% to
16.6%, for about 0.7 s per query on an Apple M-series GPU. It is off by default in
the package because of that cost and the 2.2 GB model; `t2ipc classify --rerank`
and `t2ipc eval --rerank` switch it on.

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
2011) carry abstracts. Both are in `evals/`. Besides the two standard files, the
notebooks and the sweeps use a fixed sample of 1,000 cases drawn from all 29 issues
(`load_many(..., sample=1000, seed=0)`, half with abstracts). The eval text is the
abstract when the issue printed one, else the title.

Current numbers are in `docs/evals.md`. With e5-base on the Portuguese scheme the
office's subclass is first about one time in four and within the top ten about two
times in five; the cross-encoder second stage takes rank 1 to about one time in
three. The first stage, not the hierarchy heuristics, is the limiting factor: with 50
candidates the subclass is present 64% of the time. For the second stage,
`text2ipc.eval.judge_candidates` runs the reranker once per case and
`fusion_results` evaluates every fusion rule from that run, so comparing fusions
costs one judged pass (notebook 02).
