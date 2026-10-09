// Browser glue for the static demo: fetch manifest + index files, load the ONNX
// embedder with transformers.js, embed the query, rank with scorer.js, and draw the
// merged paths of the results as a tree. Everything stays in the visitor's browser;
// no request carries the text anywhere.
import { env, pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1";
import {
  AUTO_LEVEL,
  LEVELS,
  buildIndex,
  chainOf,
  formatSymbol,
  meanVector,
  normalizeQuery,
  parseVectors,
  rank,
  rerankMatches,
  splitText,
} from "./scorer.js";

env.allowLocalModels = false;

const $ = (id) => document.getElementById(id);
const ui = {
  text: $("text"),
  lang: $("lang"),
  level: $("level"),
  topK: $("topk"),
  rerank: $("rerank"),
  rerankLabel: $("rerank-label"),
  run: $("run"),
  status: $("status"),
  graph: $("graph"),
  legend: $("legend"),
  tip: $("tip"),
  results: $("results"),
  gold: $("gold"),
  examples: $("examples"),
  meta: $("meta"),
};


const state = {
  manifest: null,
  examples: [], // real applications from the evals: title, abstract, office IPC
  gold: null, // the example whose text is in the box, if unchanged
  indexes: new Map(), // lang -> built index
  indexLoads: new Map(), // lang -> promise of the load in flight (clicks during loading reuse it)
  embedder: null,
  embedderLoad: null, // promise of the model load in flight
  rerankWorker: null, // Web Worker running the cross-encoder, created on first use
  rerankRequests: new Map(), // request id -> {resolve, reject}
  rerankSeq: 0,
  progress: new Map(), // label -> {loaded, total}
  pending: 0, // loads in flight that have not reported progress yet
  error: null,
};

const LEVEL_LEN = { section: 1, class: 3, subclass: 4 };

/** Canonical symbol cut to a level, as text2ipc.eval.harness.truncate does. */
function truncateSymbol(symbol, level) {
  if (level in LEVEL_LEN) return symbol.slice(0, LEVEL_LEN[level]);
  if (level === "group") return symbol.length === 14 ? symbol.slice(0, 8) + "000000" : symbol;
  return symbol;
}

/** Does a result agree with an office-assigned symbol at the result's own level? */
function agreesWithGold(m) {
  if (!state.gold) return false;
  return state.gold.ipc.some((g) => truncateSymbol(g, m.level) === truncateSymbol(m.symbol, m.level));
}

function exampleText(ex) {
  return `${ex.title}\n\n${ex.abstract}`;
}

// -- status -------------------------------------------------------------------

function fmtMB(bytes) {
  return `${(bytes / 1e6).toFixed(bytes < 10e6 ? 1 : 0)} MB`;
}

function fmtProgress({ loaded, total, unit }) {
  if (unit === "pairs") return `${loaded} / ${total} pairs`;
  return total ? `${fmtMB(loaded)} / ${fmtMB(total)}` : fmtMB(loaded);
}

function renderStatus(message) {
  ui.status.innerHTML = "";
  if (state.error) {
    ui.status.className = "status error";
    ui.status.textContent = state.error;
    return;
  }
  ui.status.className = "status";
  if (!state.progress.size) {
    ui.status.textContent =
      message || (state.pending ? "Loading…" : "Ready. Everything runs in this browser tab.");
    return;
  }
  for (const [label, entry] of state.progress) {
    const { loaded, total } = entry;
    const row = document.createElement("div");
    row.className = "progress";
    const pct = total ? Math.min(100, Math.round((100 * loaded) / total)) : 0;
    row.innerHTML = `<span></span><div class="bar"><i style="width:${pct}%"></i></div><small></small>`;
    row.querySelector("span").textContent = label;
    row.querySelector("small").textContent = fmtProgress(entry);
    ui.status.appendChild(row);
  }
}

function setProgress(label, loaded, total, unit) {
  state.progress.set(label, { loaded, total, unit });
  renderStatus();
}

function doneProgress(label) {
  state.progress.delete(label);
  renderStatus();
}

// -- loading ------------------------------------------------------------------

async function fetchWithProgress(url, label) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body) return res.arrayBuffer();
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    setProgress(label, loaded, total);
  }
  const out = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  doneProgress(label);
  return out.buffer;
}

function entryFor(lang) {
  return state.manifest.indexes.find((e) => e.lang === lang) || state.manifest.indexes[0];
}

function loadIndex(entry) {
  if (state.indexes.has(entry.lang)) return Promise.resolve(state.indexes.get(entry.lang));
  if (!state.indexLoads.has(entry.lang)) {
    state.pending += 1;
    const load = fetchIndex(entry)
      .catch((err) => {
        state.indexLoads.delete(entry.lang); // let a later click retry
        throw err;
      })
      .finally(() => {
        state.pending -= 1;
        renderStatus();
      });
    state.indexLoads.set(entry.lang, load);
  }
  return state.indexLoads.get(entry.lang);
}

async function fetchIndex(entry) {
  const label = `IPC ${entry.version} ${entry.lang}`;
  const [schemeBuf, vecBuf] = await Promise.all([
    fetchWithProgress(entry.scheme, `${label} scheme`),
    fetchWithProgress(entry.vectors, `${label} vectors`),
  ]);
  const scheme = JSON.parse(new TextDecoder().decode(schemeBuf));
  const vectors = parseVectors(vecBuf, {
    rows: entry.rows,
    dim: state.manifest.dim,
    encoding: entry.encoding,
  });
  const index = buildIndex(scheme, vectors);
  state.indexes.set(entry.lang, index);
  return index;
}

function loadEmbedder() {
  if (state.embedder) return Promise.resolve(state.embedder);
  if (!state.embedderLoad) {
    const { web_model: model, web_dtype: dtype } = state.manifest;
    const label = `model ${model} (${dtype})`;
    state.pending += 1;
    state.embedderLoad = pipeline("feature-extraction", model, {
      dtype,
      progress_callback: (e) => {
        if (e.status === "progress" && e.file && e.file.endsWith(".onnx")) {
          setProgress(label, e.loaded, e.total);
        }
      },
    })
      .then((p) => {
        state.embedder = p;
        return p;
      })
      .catch((err) => {
        state.embedderLoad = null; // let a later click retry
        throw err;
      })
      .finally(() => {
        state.pending -= 1;
        doneProgress(label);
      });
  }
  return state.embedderLoad;
}

/** The reranker runs in a Web Worker (rerank-worker.js) so the page keeps painting. */
function rerankWorker() {
  if (!state.rerankWorker) {
    const worker = new Worker("rerank-worker.js", { type: "module" });
    const { web_model: model, dtype } = state.manifest.reranker;
    const loadLabel = `reranker ${model} (${dtype})`;
    const judgeLabel = "judging candidates";
    worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === "progress") setProgress(loadLabel, m.loaded, m.total);
      else if (m.type === "ready") doneProgress(loadLabel);
      else if (m.type === "judging") setProgress(judgeLabel, m.done, m.total, "pairs");
      else if (m.id !== undefined && state.rerankRequests.has(m.id)) {
        const { resolve, reject } = state.rerankRequests.get(m.id);
        state.rerankRequests.delete(m.id);
        doneProgress(loadLabel);
        doneProgress(judgeLabel);
        if (m.type === "error") reject(new Error(m.message));
        else resolve(m);
      }
    };
    worker.onerror = (e) => {
      for (const { reject } of state.rerankRequests.values()) reject(new Error(e.message || "reranker worker failed"));
      state.rerankRequests.clear();
      doneProgress(loadLabel);
      doneProgress(judgeLabel);
    };
    state.rerankWorker = worker;
  }
  return state.rerankWorker;
}

function askWorker(message) {
  const worker = rerankWorker();
  const id = ++state.rerankSeq;
  return new Promise((resolve, reject) => {
    state.rerankRequests.set(id, { resolve, reject });
    worker.postMessage({ id, ...message });
  });
}

/** Cross-encoder logits for (text, path text) pairs; progress shows in the status. */
async function judge(text, matches) {
  const { web_model: modelId, dtype, max_length: maxLength } = state.manifest.reranker;
  setProgress("judging candidates", 0, matches.length, "pairs");
  const reply = await askWorker({
    type: "judge",
    modelId,
    dtype,
    text,
    texts: matches.map((m) => m.text),
    maxLength,
    batch: 4,
  });
  return reply.logits;
}

/**
 * Query vector for a text of any length. Paragraphs are embedded separately and a
 * paragraph over the model's token limit is cut into sentence chunks (scorer.js
 * splitText, a port of text2ipc/chunking.py); the unit-length mean of the pieces is
 * the query, as IpcClassifier does with chunking="mean".
 */
async function embedText(text) {
  const { query_prefix: prefix, max_tokens: limit } = state.manifest;
  const countTokens = (t) => state.embedder.tokenizer(prefix + t).input_ids.dims[1];
  const options = { pooling: "mean", normalize: true };
  const chunks = splitText(text, limit || null, countTokens);
  if (chunks.length <= 1) {
    const out = await state.embedder(prefix + text, options);
    return { query: out.data, chunks: 1 };
  }
  const out = await state.embedder(chunks.map((c) => prefix + c), options);
  const dim = out.dims[out.dims.length - 1];
  const vectors = chunks.map((_, i) => out.data.subarray(i * dim, (i + 1) * dim));
  return { query: meanVector(vectors), chunks: chunks.length };
}

// -- classify -----------------------------------------------------------------

// One classification at a time. The ONNX session must not run twice concurrently,
// and the page must end up showing the results of the latest request: a request
// made while one is running waits for it and then runs with the current inputs.
let running = false;
let queued = false;

async function classify() {
  if (running) {
    queued = true;
    return;
  }
  running = true;
  ui.run.disabled = true;
  try {
    do {
      queued = false;
      await classifyOnce();
    } while (queued);
  } finally {
    running = false;
    ui.run.disabled = false;
  }
}

async function classifyOnce() {
  const text = normalizeQuery(ui.text.value || "");
  if (!text) return;
  if (state.gold && (ui.text.value || "").trim() !== exampleText(state.gold).trim()) state.gold = null;
  try {
    const entry = entryFor(ui.lang.value);
    const [index] = await Promise.all([loadIndex(entry), loadEmbedder()]);
    renderStatus("Embedding...");
    const t0 = performance.now();
    const { query, chunks } = await embedText(text);
    const t1 = performance.now();
    const topK = Number(ui.topK.value);
    const rerank = Boolean(state.manifest.reranker) && ui.rerank.checked;
    const candidates = rerank ? Math.max(state.manifest.reranker.candidates, topK) : topK;
    let { matches, sims } = rank(index, query, { level: ui.level.value, topK: candidates });
    const t2 = performance.now();
    let judged = "";
    if (rerank && matches.length) {
      renderStatus();
      const logits = await judge(text, matches);
      matches = rerankMatches(matches, logits, state.manifest.reranker.fusion, topK);
      judged = ` · judged ${logits.length} in ${Math.round(performance.now() - t2)} ms`;
    }
    renderGraph(index, matches, sims, entry);
    renderResults(matches, rerank);
    renderGold(matches);
    ui.meta.textContent =
      `IPC ${entry.version} ${entry.lang} · ${entry.rows.toLocaleString()} entries · ` +
      `${state.manifest.web_model} ${state.manifest.web_dtype} · ` +
      `${chunks > 1 ? `${chunks} chunks · ` : ""}` +
      `embed ${Math.round(t1 - t0)} ms · score ${Math.round(t2 - t1)} ms${judged}`;
    renderStatus();
    updateUrl(text, entry.lang);
  } catch (err) {
    state.error = `Something failed: ${err.message || err}`;
    renderStatus();
    console.error(err);
  }
}

function renderResults(matches, reranked = false) {
  ui.results.innerHTML = "";
  if (!matches.length) {
    ui.results.textContent = "No result.";
    return;
  }
  const table = document.createElement("table");
  table.innerHTML =
    "<thead><tr><th>#</th><th>Symbol</th><th>Level</th><th>Score</th><th>Sim.</th>" +
    (reranked ? "<th>Judge</th>" : "") +
    "<th>Section &gt; … &gt; entry</th></tr></thead>";
  const body = document.createElement("tbody");
  matches.forEach((m, i) => {
    const tr = document.createElement("tr");
    const cells = [
      String(i + 1),
      m.pretty,
      m.level,
      m.score.toFixed(3),
      m.similarity.toFixed(3),
      ...(reranked ? [m.judge.toFixed(2)] : []),
      m.text,
    ];
    cells.forEach((v, j) => {
      const td = document.createElement("td");
      td.textContent = v;
      if (j === 1) td.className = "symbol";
      if (j === cells.length - 1) td.className = "path";
      tr.appendChild(td);
    });
    if (agreesWithGold(m)) {
      tr.className = "gold";
      const tick = document.createElement("span");
      tick.className = "tick";
      tick.textContent = " ✓";
      tick.title = "Agrees with the symbol INPI assigned, at this level";
      tr.children[1].appendChild(tick);
    }
    body.appendChild(tr);
  });
  table.appendChild(body);
  ui.results.appendChild(table);
}

// -- graph: the results' paths merged into one tree ----------------------------

const G = { rowH: 34, gap: 96, padX: 14, padY: 16, nodeH: 24, charW: 7.4, badgeW: 34 };
const SVG = "http://www.w3.org/2000/svg";

function svgEl(tag, attrs = {}, text) {
  const el = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (text !== undefined) el.textContent = text;
  return el;
}

/** What a node shows: A, 62, C, 25/00, 25/01 along the path A62C 25/01. */
function nodeLabel(index, i) {
  const sym = index.symbol[i];
  const lvl = index.level[i];
  if (lvl === 0) return sym;
  if (lvl === 1) return sym.slice(1);
  if (lvl === 2) return sym.slice(3);
  return formatSymbol(sym).split(" ")[1];
}

const ROOT = -1; // virtual level-0 node above the sections, so section edges carry a value

/** Merge the result paths into one tree under a virtual root; assign a row to every node. */
function buildGraph(index, matches) {
  const nodes = new Map();
  nodes.set(ROOT, { i: ROOT, col: 0, parent: null, children: [], minRank: 1 });
  matches.forEach((m, r) => {
    const chain = chainOf(index, m.index);
    chain.forEach((i, d) => {
      if (!nodes.has(i)) {
        nodes.set(i, { i, col: d + 1, parent: d ? chain[d - 1] : ROOT, children: [], minRank: r + 1 });
      }
    });
    const leaf = nodes.get(m.index);
    leaf.match = m;
    leaf.rank = r + 1;
  });
  for (const n of nodes.values()) if (n.parent !== null) nodes.get(n.parent).children.push(n.i);
  const byRank = (a, b) => nodes.get(a).minRank - nodes.get(b).minRank || a - b;
  for (const n of nodes.values()) n.children.sort(byRank);
  let row = 0;
  const place = (n) => {
    if (!n.children.length) {
      n.row = row++;
      return;
    }
    n.children.forEach((c) => place(nodes.get(c)));
    const rows = n.children.map((c) => nodes.get(c).row);
    n.row = (Math.min(...rows) + Math.max(...rows)) / 2;
  };
  place(nodes.get(ROOT));
  return { nodes, rows: row };
}

function renderGraph(index, matches, sims, entry) {
  ui.graph.innerHTML = "";
  hideTip();
  if (!matches.length) {
    ui.legend.hidden = true;
    return;
  }
  const { nodes, rows } = buildGraph(index, matches);
  for (const n of nodes.values()) {
    n.label = n.i === ROOT ? "IPC" : nodeLabel(index, n.i);
    n.w = Math.max(28, 12 + n.label.length * G.charW);
  }
  const cols = Math.max(...[...nodes.values()].map((n) => n.col)) + 1;
  const colW = new Array(cols).fill(0);
  for (const n of nodes.values()) colW[n.col] = Math.max(colW[n.col], n.w);
  const colX = [];
  let x = G.padX;
  for (let c = 0; c < cols; c++) {
    colX.push(x);
    x += colW[c] + G.gap;
  }
  const width = x - G.gap + G.badgeW + G.padX;
  const height = rows * G.rowH + 2 * G.padY;
  const yOf = (row) => G.padY + row * G.rowH + G.rowH / 2;

  const svg = svgEl("svg", {
    width,
    height,
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": "Paths of the results through the IPC hierarchy",
  });
  const edges = svgEl("g", { class: "edges" });
  const labels = svgEl("g", { class: "edge-labels" });
  const nodeLayer = svgEl("g", { class: "nodes" });
  svg.append(edges, labels, nodeLayer);

  for (const n of nodes.values()) {
    const nx = colX[n.col];
    const ny = yOf(n.row);
    const isResult = n.match !== undefined;
    if (n.parent !== null) {
      const p = nodes.get(n.parent);
      const x1 = colX[p.col] + p.w;
      const y1 = yOf(p.row);
      const xm = (x1 + nx) / 2;
      const d = `M${x1},${y1} C${xm},${y1} ${xm},${ny} ${nx},${ny}`;
      const value = isResult ? n.match.score : sims[n.i];
      const edge = svgEl("path", { d, class: `edge${isResult ? " result" : ""}` });
      const hit = svgEl("path", { d, class: "hit", tabindex: "0" });
      edges.append(edge, hit);
      const lx = (x1 + nx) / 2;
      const ly = (y1 + ny) / 2;
      const txt = value.toFixed(3);
      const lw = 10 + txt.length * 6.6;
      const pill = svgEl("g", { class: `edge-label${isResult ? " result" : ""}` });
      pill.append(
        svgEl("rect", { x: lx - lw / 2, y: ly - 9, width: lw, height: 18, rx: 9 }),
        svgEl("text", { x: lx, y: ly + 4, "text-anchor": "middle" }, txt),
      );
      labels.append(pill);
      const info = tipInfo(index, n, sims, entry);
      hit.setAttribute("aria-label", info.aria);
      bindTip(hit, info, edge);
    }
    const g = svgEl("g", {
      class: `node${isResult ? " result" : ""}${n.i === ROOT ? " root" : ""}`,
      transform: `translate(${nx},${ny - G.nodeH / 2})`,
      tabindex: "0",
    });
    g.append(
      svgEl("rect", { width: n.w, height: G.nodeH, rx: 6 }),
      svgEl("text", { x: n.w / 2, y: G.nodeH / 2 + 4, "text-anchor": "middle" }, n.label),
    );
    if (isResult) {
      const badge = `#${n.rank}${agreesWithGold(n.match) ? " ✓" : ""}`;
      g.append(svgEl("text", { class: "badge", x: n.w + 8, y: G.nodeH / 2 + 4 }, badge));
    }
    const info = tipInfo(index, n, sims, entry);
    g.setAttribute("aria-label", info.aria);
    bindTip(g, info, g);
    nodeLayer.append(g);
  }
  ui.graph.appendChild(svg);
  ui.legend.hidden = false;
}

function tipInfo(index, n, sims, entry) {
  if (n.i === ROOT) {
    const strong = `IPC ${entry.version} ${entry.lang}`;
    const title = "Level 0: the whole scheme. Each edge to a section carries that section's similarity.";
    return {
      lines: [{ strong, rest: ` · ${entry.rows.toLocaleString()} entries` }],
      symbol: "root",
      title,
      aria: `${strong}. ${title}`,
    };
  }
  const sym = formatSymbol(index.symbol[n.i]);
  const level = LEVELS[index.level[n.i]];
  const title = index.title[n.i] || index.heading[n.i] || "";
  const lines = [];
  if (n.match) {
    lines.push({ strong: `score ${n.match.score.toFixed(3)}`, rest: ` · result #${n.rank}` });
    const judged = n.match.judge === undefined ? "" : ` · judge ${n.match.judge.toFixed(2)}`;
    lines.push({
      rest: `similarity ${n.match.similarity.toFixed(3)} · path support ${n.match.pathSupport.toFixed(3)}${judged}`,
    });
  } else {
    lines.push({ strong: `similarity ${sims[n.i].toFixed(3)}`, rest: "" });
  }
  return {
    lines,
    symbol: `${sym} · ${level}`,
    title,
    aria: `${sym}, ${level}: ${title}. ${lines.map((l) => (l.strong || "") + l.rest).join(". ")}`,
  };
}

function bindTip(target, info, lift) {
  const show = (e) => {
    lift.classList.add("hover");
    showTip(info, e);
  };
  const hide = () => {
    lift.classList.remove("hover");
    hideTip();
  };
  target.addEventListener("pointerenter", show);
  target.addEventListener("pointermove", (e) => positionTip(e.clientX, e.clientY));
  target.addEventListener("pointerleave", hide);
  target.addEventListener("focus", (e) => {
    const r = e.target.getBoundingClientRect();
    lift.classList.add("hover");
    showTip(info, null);
    positionTip(r.left + r.width / 2, r.top + r.height / 2);
  });
  target.addEventListener("blur", hide);
}

function showTip(info, e) {
  ui.tip.innerHTML = "";
  for (const l of info.lines) {
    const p = document.createElement("div");
    if (l.strong) {
      const b = document.createElement("strong");
      b.textContent = l.strong;
      p.appendChild(b);
    }
    p.appendChild(document.createTextNode(l.rest));
    ui.tip.appendChild(p);
  }
  const sym = document.createElement("div");
  sym.className = "tip-symbol";
  sym.textContent = info.symbol;
  const title = document.createElement("div");
  title.className = "tip-title";
  title.textContent = info.title;
  ui.tip.append(sym, title);
  ui.tip.hidden = false;
  if (e) positionTip(e.clientX, e.clientY);
}

function positionTip(cx, cy) {
  const pad = 14;
  const w = ui.tip.offsetWidth;
  const h = ui.tip.offsetHeight;
  let left = cx + pad;
  let top = cy + pad;
  if (left + w > window.innerWidth - 8) left = cx - w - pad;
  if (top + h > window.innerHeight - 8) top = cy - h - pad;
  ui.tip.style.left = `${Math.max(8, left)}px`;
  ui.tip.style.top = `${Math.max(8, top)}px`;
}

function hideTip() {
  ui.tip.hidden = true;
}

function renderGold(matches) {
  ui.gold.innerHTML = "";
  if (!state.gold) return;
  const g = state.gold;
  const hits = matches.filter(agreesWithGold).length;
  const strong = document.createElement("strong");
  strong.textContent = `INPI assigned: ${g.ipc.map(formatSymbol).join(", ")}`;
  ui.gold.appendChild(strong);
  ui.gold.appendChild(
    document.createTextNode(
      ` (${g.source}). ${hits ? `${hits} of ${matches.length} results agree at their level (✓).` : "No result agrees at its level."}`,
    ),
  );
}

function updateUrl(text, lang) {
  const params = new URLSearchParams({ lang, level: ui.level.value, q: text });
  history.replaceState(null, "", `?${params}`);
}

// -- setup --------------------------------------------------------------------

function fillSelect(select, values, labels) {
  select.innerHTML = "";
  values.forEach((v, i) => {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = labels ? labels[i] : v;
    select.appendChild(o);
  });
}

async function loadExamples() {
  try {
    const res = await fetch("examples.json");
    if (res.ok) state.examples = await res.json();
  } catch {
    state.examples = [];
  }
}

function renderExamples() {
  ui.examples.innerHTML = "";
  for (const ex of state.examples) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "chip";
    b.textContent = ex.title.length > 56 ? `${ex.title.slice(0, 56)}…` : ex.title;
    b.title = `${ex.abstract.slice(0, 160)}…  INPI: ${ex.ipc.map(formatSymbol).join(", ")}`;
    b.addEventListener("click", () => {
      ui.text.value = exampleText(ex); // filled in only; Classify runs it
      state.gold = ex;
      ui.text.focus();
    });
    ui.examples.appendChild(b);
  }
}

async function main() {
  try {
    const res = await fetch("manifest.json");
    if (!res.ok) throw new Error(`manifest.json: HTTP ${res.status}`);
    state.manifest = await res.json();
  } catch (err) {
    state.error = `Could not load the index manifest: ${err.message || err}`;
    renderStatus();
    return;
  }
  const m = state.manifest;
  fillSelect(
    ui.lang,
    m.indexes.map((e) => e.lang),
    m.indexes.map((e) => `${e.lang} scheme (IPC ${e.version})`),
  );
  fillSelect(ui.level, [AUTO_LEVEL, ...LEVELS]);
  ui.level.value = "group";
  if (!m.reranker) ui.rerankLabel.hidden = true;
  else ui.rerankLabel.title = `${m.reranker.web_model} (${m.reranker.dtype}) re-judges the top ${m.reranker.candidates} when you click Classify; 280 MB once, then about a second per candidate`;
  await loadExamples();
  renderExamples();

  const params = new URLSearchParams(location.search);
  if (params.get("lang") && m.indexes.some((e) => e.lang === params.get("lang"))) {
    ui.lang.value = params.get("lang");
  }
  if (params.get("level") && [AUTO_LEVEL, ...LEVELS].includes(params.get("level"))) {
    ui.level.value = params.get("level");
  }
  if (params.get("q")) ui.text.value = params.get("q");

  ui.run.addEventListener("click", classify);
  ui.lang.addEventListener("change", () => loadIndex(entryFor(ui.lang.value)).catch(fail));
  window.addEventListener("scroll", hideTip, { passive: true });

  try {
    await Promise.all([loadIndex(entryFor(ui.lang.value)), loadEmbedder()]);
    renderStatus(); // nothing runs until Classify is clicked
  } catch (err) {
    fail(err);
  }
}

function fail(err) {
  state.error = `Could not load the model or the index: ${err.message || err}`;
  renderStatus();
  console.error(err);
}

main();
