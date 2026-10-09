// JavaScript port of text2ipc/search/scorer.py: hierarchical scoring on top of
// flat cosine similarity. Pure functions, no DOM, no fetch: the browser page
// (app.js) and the Node parity test (tests/js_parity.mjs) both import it.
//
// Keep in step with the Python module. Every heuristic here mirrors a function
// there by name: pathSupport, subtreeSupport, beamDescend, autoDescend,
// distinctBranches. tests/test_web.py runs both on the mini scheme and compares.

export const LEVELS = ["section", "class", "subclass", "group", "subgroup"];
export const AUTO_LEVEL = "auto";
export const PATH_SEPARATOR = " > ";

export const DEFAULT_WEIGHTS = { own: 1.0, path: 0.0, subtree: 0.0 }; // see Weights in scorer.py
export const DEFAULT_BEAM = { section: 5, class: 10, subclass: 20, group: 40 };
export const DEFAULT_PARAMS = {
  level: "subgroup",
  topK: 10,
  gap: null, // drop results scoring more than this below the best one
  autoMargin: 0.02, // auto level: descend while best child >= parent - margin
  autoRoots: null, // auto level: subclasses to start from; null = max(5, topK)
  dedupeBranches: true, // drop ancestors/descendants of a higher-ranked result
};

// -- index ------------------------------------------------------------------

/**
 * Build the in-memory index from the exported columns and vectors.
 * scheme: {symbol: string[], level: int[], depth: int[], parent: int[], title: string[], heading: string[]}
 * vectors: {encoding: "int8"|"float32", rows, dim, data: Int8Array|Float32Array, scales?: Float32Array}
 */
export function buildIndex(scheme, vectors) {
  const n = scheme.symbol.length;
  if (vectors.rows !== n) {
    throw new Error(`vectors have ${vectors.rows} rows, scheme has ${n}`);
  }
  const parent = Int32Array.from(scheme.parent);
  const children = Array.from({ length: n }, () => []);
  for (let i = 0; i < n; i++) {
    const p = parent[i];
    if (p >= i) throw new Error("Index rows must be in depth-first order (parents before children)");
    if (p >= 0) children[p].push(i);
  }
  return {
    rows: n,
    dim: vectors.dim,
    symbol: scheme.symbol,
    level: Int8Array.from(scheme.level),
    depth: Int8Array.from(scheme.depth),
    parent,
    children,
    title: scheme.title,
    heading: scheme.heading,
    vectors,
  };
}

/** Parse vectors.bin as written by text2ipc.web.export.write_vectors. */
export function parseVectors(buffer, meta) {
  const { rows, dim, encoding } = meta;
  if (encoding === "int8") {
    const scales = new Float32Array(buffer, 0, rows);
    const data = new Int8Array(buffer, rows * 4, rows * dim);
    return { encoding, rows, dim, data, scales };
  }
  if (encoding === "float32") {
    return { encoding, rows, dim, data: new Float32Array(buffer, 0, rows * dim) };
  }
  throw new Error(`unknown vector encoding ${encoding}`);
}

/** Canonical symbol to human form: A01B0001020000 -> "A01B 1/02". */
export function formatSymbol(symbol) {
  if (symbol.length !== 14) return symbol;
  const sub = symbol.slice(0, 4);
  const group = symbol.slice(4, 8);
  let subgroup = symbol.slice(8).replace(/0+$/, "");
  if (subgroup.length < 2) subgroup = subgroup.padEnd(2, "0");
  return `${sub} ${parseInt(group, 10)}/${subgroup}`;
}

/** Symbol chain from the section down to row i. */
export function chainOf(index, i) {
  const chain = [];
  for (let node = i; node >= 0; node = index.parent[node]) chain.push(node);
  return chain.reverse();
}

/** The text an entry was embedded with: titles top-down, heading before a group. */
export function textAt(index, i) {
  const parts = [];
  for (const s of chainOf(index, i)) {
    if (index.heading[s]) parts.push(index.heading[s]);
    parts.push(index.title[s]);
  }
  return parts.filter((p) => p).join(PATH_SEPARATOR);
}

// -- query normalisation (text2ipc/textnorm.py, classifier.normalize_query) ---

const PARAGRAPH_RE = /\n\s*\n/;

/** Non-empty parts separated by a blank line; single newlines do not split. */
export function paragraphs(text) {
  return text.split(PARAGRAPH_RE).filter((p) => p.trim());
}

/** Collapse runs of whitespace inside each paragraph; keep blank lines as breaks. */
export function collapseWhitespace(text) {
  return paragraphs(text)
    .map((p) => p.split(/\s+/).filter(Boolean).join(" "))
    .join("\n\n");
}

export function lowercaseIfShouting(text, threshold = 0.6) {
  const letters = Array.from(text).filter((c) => /\p{L}/u.test(c));
  if (!letters.length) return text;
  const upper = letters.filter((c) => c !== c.toLowerCase() && c === c.toUpperCase()).length;
  return upper / letters.length > threshold ? text.toLowerCase() : text;
}

export function normalizeQuery(text) {
  return collapseWhitespace(text).split("\n\n").map(lowercaseIfShouting).join("\n\n");
}

// -- scoring ------------------------------------------------------------------

function similarities(index, query) {
  const { rows, dim, data, scales, encoding } = index.vectors;
  // an array of vectors comes from a long text: an entry scores by its best chunk
  const queries = Array.isArray(query) ? query : [query];
  const sims = new Float64Array(rows).fill(-Infinity);
  for (const q of queries) {
    if (q.length !== dim) throw new Error(`query has ${q.length} dimensions, index has ${dim}`);
    for (let i = 0; i < rows; i++) {
      const off = i * dim;
      let s = 0;
      for (let j = 0; j < dim; j++) s += data[off + j] * q[j];
      s = encoding === "int8" ? s * scales[i] : s;
      if (s > sims[i]) sims[i] = s;
    }
  }
  return sims;
}

// -- long texts (text2ipc/chunking.py) -----------------------------------------

const SENTENCE_RE = /(?<=[.!?;:])\s+(?=\S)/;

/**
 * One chunk per paragraph (blank-line separated, never merged), cut further into
 * sentence chunks of at most maxTokens tokens when a paragraph is too long; a single
 * over-long sentence is cut by words. `overlap` trailing sentences of a chunk are
 * repeated at the start of the next. countTokens(text) must count a text as the
 * embedder will see it; maxTokens null means no limit.
 */
export function splitText(text, maxTokens, countTokens, overlap = 1) {
  return paragraphs(text).flatMap((p) => pack(p.trim(), maxTokens, countTokens, overlap));
}

function pack(paragraph, maxTokens, countTokens, overlap) {
  if (maxTokens === null || maxTokens === undefined) {
    return [paragraph.split(/\s+/).filter(Boolean).join(" ")];
  }
  let units = paragraph.split(SENTENCE_RE).map((s) => s.trim()).filter(Boolean);
  units = units.flatMap((u) => cutByWords(u, maxTokens, countTokens));
  const chunks = [];
  let current = [];
  for (const unit of units) {
    if (current.length && countTokens([...current, unit].join(" ")) > maxTokens) {
      chunks.push(current);
      const carry = overlap > 0 ? current.slice(-overlap) : [];
      current = [...carry, unit];
      while (current.length > 1 && countTokens(current.join(" ")) > maxTokens) {
        current = current.slice(1);
      }
    } else {
      current.push(unit);
    }
  }
  if (current.length) chunks.push(current);
  return chunks.map((c) => c.join(" "));
}

function cutByWords(unit, maxTokens, countTokens) {
  if (countTokens(unit) <= maxTokens) return [unit];
  const out = [];
  let piece = [];
  for (const w of unit.split(/\s+/).filter(Boolean)) {
    if (piece.length && countTokens([...piece, w].join(" ")) > maxTokens) {
      out.push(piece.join(" "));
      piece = [];
    }
    piece.push(w);
  }
  if (piece.length) out.push(piece.join(" "));
  return out;
}

/** Unit-length mean of several unit vectors (the "mean" chunking policy). */
export function meanVector(vectors) {
  const dim = vectors[0].length;
  const out = new Float32Array(dim);
  for (const v of vectors) for (let j = 0; j < dim; j++) out[j] += v[j];
  let norm = 0;
  for (let j = 0; j < dim; j++) norm += out[j] * out[j];
  norm = Math.sqrt(norm) || 1;
  for (let j = 0; j < dim; j++) out[j] /= norm;
  return out;
}

/** Mean similarity of the ancestors; equals own similarity for roots. */
function pathSupport(index, sims) {
  const n = index.rows;
  const total = new Float64Array(n);
  const count = new Int32Array(n);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const p = index.parent[i]; // parents precede children
    if (p >= 0) {
      total[i] = total[p] + sims[p];
      count[i] = count[p] + 1;
    }
    out[i] = count[i] > 0 ? total[i] / count[i] : sims[i];
  }
  return out;
}

/** Max value over the node and everything below it. */
function subtreeSupport(index, values) {
  const best = Float64Array.from(values);
  for (let i = index.rows - 1; i >= 0; i--) {
    const p = index.parent[i]; // children precede parents in reverse
    if (p >= 0 && best[i] > best[p]) best[p] = best[i];
  }
  return best;
}

/** Stable descending sort of row indices by a value array (Python's sorted(key=-v)). */
function sortDesc(items, values) {
  return items.slice().sort((a, b) => values[b] - values[a]);
}

function* descendants(index, roots) {
  const stack = roots.slice();
  while (stack.length) {
    const i = stack.pop();
    for (const c of index.children[i]) {
      yield c;
      stack.push(c);
    }
  }
}

/** Walk down from the sections, keeping the parents whose subtree scores best. */
function beamDescend(index, bestBelow, target, beam) {
  const targetRank = LEVELS.indexOf(target);
  let frontier = [];
  for (let i = 0; i < index.rows; i++) if (index.level[i] === 0) frontier.push(i);
  for (let rank = 0; rank < LEVELS.length; rank++) {
    if (rank === targetRank) return frontier;
    const level = LEVELS[rank];
    frontier = sortDesc(frontier, bestBelow).slice(0, beam[level]);
    const next = LEVELS[rank + 1];
    if (next === "subgroup") {
      frontier = Array.from(descendants(index, frontier));
    } else {
      frontier = frontier.flatMap((i) => index.children[i]);
    }
  }
  return frontier;
}

function argmaxFirst(items, values) {
  let best = items[0];
  for (const i of items) if (values[i] > values[best]) best = i;
  return best;
}

/**
 * From the best subclasses, walk down the branch with the best subtree score while
 * that subtree still promises something within autoMargin of the current node, then
 * answer with the node on the walked path whose own similarity is highest (deepest
 * on ties).
 */
function autoDescend(index, sims, score, bestBelow, p) {
  const nRoots = p.autoRoots ?? Math.max(5, p.topK);
  let roots = beamDescend(index, bestBelow, "subclass", p.beam);
  roots = sortDesc(roots, bestBelow).slice(0, nRoots);
  const out = [];
  for (const i of roots) {
    const path = [i];
    let node = i;
    while (index.children[node].length) {
      const bestChild = argmaxFirst(index.children[node], bestBelow);
      if (bestBelow[bestChild] < score[node] - p.autoMargin) break;
      node = bestChild;
      path.push(node);
    }
    const best = argmaxFirst(path.slice().reverse(), sims);
    if (!out.includes(best)) out.push(best);
  }
  return sortDesc(out, score).slice(0, p.topK);
}

function isAncestorOrSelf(index, a, i) {
  for (let node = i; node >= 0; node = index.parent[node]) if (node === a) return true;
  return false;
}

/** Keep a result only if no accepted result sits on its path or below it. */
function distinctBranches(index, ranked) {
  const accepted = [];
  for (const i of ranked) {
    const sameBranch = accepted.some(
      (a) => isAncestorOrSelf(index, a, i) || isAncestorOrSelf(index, i, a),
    );
    if (!sameBranch) accepted.push(i);
  }
  return accepted;
}

/**
 * Rank IPC entries for a unit query vector. Returns matches ordered by score with the
 * same fields as text2ipc.search.Match plus `pretty` and `index` (the row).
 */
export function search(index, query, params = {}) {
  return rank(index, query, params).matches;
}

/**
 * Like search, but also returns the per-row arrays the matches were computed from
 * (`sims`, `scores`, `pathSupport`, `subtreeSupport`), so a caller can show how the
 * ancestors of a result scored. The Python package has no counterpart; `search`
 * is the function the parity test covers.
 */
export function rank(index, query, params = {}) {
  const p = {
    ...DEFAULT_PARAMS,
    ...params,
    weights: { ...DEFAULT_WEIGHTS, ...(params.weights || {}) },
    beam: { ...DEFAULT_BEAM, ...(params.beam || {}) },
  };
  if (p.level !== AUTO_LEVEL && !LEVELS.includes(p.level)) {
    throw new Error(`level must be one of ${LEVELS.join(", ")} or ${AUTO_LEVEL}`);
  }
  const sims = similarities(index, query);
  const path = pathSupport(index, sims);
  const subtree = subtreeSupport(index, sims);
  const w = p.weights;
  const score = new Float64Array(index.rows);
  for (let i = 0; i < index.rows; i++) {
    score[i] = w.own * sims[i] + w.path * path[i] + w.subtree * subtree[i];
  }
  const bestBelow = subtreeSupport(index, score);

  let chosen;
  if (p.level === AUTO_LEVEL) {
    chosen = autoDescend(index, sims, score, bestBelow, p);
  } else {
    chosen = beamDescend(index, bestBelow, p.level, p.beam);
    chosen = sortDesc(chosen, score).slice(0, p.topK);
  }
  if (p.gap !== null && p.gap !== undefined && chosen.length) {
    const best = score[chosen[0]];
    chosen = chosen.filter((i) => score[i] >= best - p.gap);
  }
  if (p.dedupeBranches) chosen = distinctBranches(index, chosen);

  const matches = chosen.map((i) => ({
    index: i,
    symbol: index.symbol[i],
    pretty: formatSymbol(index.symbol[i]),
    level: LEVELS[index.level[i]],
    depth: index.depth[i],
    title: index.title[i],
    text: textAt(index, i),
    score: score[i],
    similarity: sims[i],
    pathSupport: path[i],
    subtreeSupport: subtree[i],
  }));
  return { matches, sims, scores: score, pathSupport: path, subtreeSupport: subtree };
}

// -- second stage (text2ipc/rerank/base.py) ------------------------------------

export function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

/**
 * Combine first-stage scores with a judge's logits: "blend" (default) is
 * score * (0.5 + 0.5 * sigmoid(logit)); "judge" is sigmoid(logit) alone;
 * "product" is score * sigmoid(logit).
 */
export function fuse(scores, logits, how = "blend") {
  return scores.map((s, i) => {
    const j = sigmoid(logits[i]);
    if (how === "blend") return s * (0.5 + 0.5 * j);
    if (how === "judge") return j;
    if (how === "product") return s * j;
    throw new Error(`fusion must be blend, judge or product, got ${how}`);
  });
}

/** Reorder matches by the fused score and attach each judge verdict (0..1). */
export function rerankMatches(matches, logits, how = "blend", topK = matches.length) {
  const fused = fuse(matches.map((m) => m.score), logits, how);
  const order = matches.map((_, i) => i).sort((a, b) => fused[b] - fused[a]);
  return order.slice(0, topK).map((i) => ({ ...matches[i], score: fused[i], judge: sigmoid(logits[i]) }));
}
