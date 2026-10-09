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

export const DEFAULT_WEIGHTS = { own: 0.7, path: 0.3, subtree: 0.0 };
export const DEFAULT_BEAM = { section: 5, class: 10, subclass: 20, group: 40 };
export const DEFAULT_PARAMS = {
  level: "subgroup",
  topK: 10,
  gap: null, // drop results scoring more than this below the best one
  autoMargin: 0.02, // auto level: descend while best child >= parent - margin
  autoRoots: 5, // auto level: how many subclasses to start descending from
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

export function collapseWhitespace(text) {
  return text.split(/\s+/).filter(Boolean).join(" ");
}

export function lowercaseIfShouting(text, threshold = 0.6) {
  const letters = Array.from(text).filter((c) => /\p{L}/u.test(c));
  if (!letters.length) return text;
  const upper = letters.filter((c) => c !== c.toLowerCase() && c === c.toUpperCase()).length;
  return upper / letters.length > threshold ? text.toLowerCase() : text;
}

export function normalizeQuery(text) {
  return lowercaseIfShouting(collapseWhitespace(text));
}

// -- scoring ------------------------------------------------------------------

function similarities(index, query) {
  const { rows, dim, data, scales, encoding } = index.vectors;
  const sims = new Float64Array(rows);
  for (let i = 0; i < rows; i++) {
    const off = i * dim;
    let s = 0;
    for (let j = 0; j < dim; j++) s += data[off + j] * query[j];
    sims[i] = encoding === "int8" ? s * scales[i] : s;
  }
  return sims;
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
  let roots = beamDescend(index, bestBelow, "subclass", p.beam);
  roots = sortDesc(roots, bestBelow).slice(0, p.autoRoots);
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
  if (query.length !== index.dim) {
    throw new Error(`query has ${query.length} dimensions, index has ${index.dim}`);
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
