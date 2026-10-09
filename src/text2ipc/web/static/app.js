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
  normalizeQuery,
  parseVectors,
  rank,
} from "./scorer.js";

env.allowLocalModels = false;

const $ = (id) => document.getElementById(id);
const ui = {
  text: $("text"),
  lang: $("lang"),
  level: $("level"),
  topK: $("topk"),
  run: $("run"),
  status: $("status"),
  graph: $("graph"),
  legend: $("legend"),
  tip: $("tip"),
  results: $("results"),
  examples: $("examples"),
  meta: $("meta"),
};

const EXAMPLES = [
  { lang: "PT", text: "Aparelho para combate a incêndios com mangueira flexível reforçada" },
  {
    lang: "PT",
    text: "Composição farmacêutica compreendendo um anticorpo monoclonal para o tratamento de câncer de mama",
  },
  { lang: "EN", text: "Hand-operated hoe with two blades for weeding between rows of plants" },
  {
    lang: "EN",
    text: "Method for transmitting data packets between nodes of a wireless mesh network with adaptive routing",
  },
];

const state = {
  manifest: null,
  indexes: new Map(), // lang -> built index
  embedder: null,
  progress: new Map(), // label -> {loaded, total}
  error: null,
};

// -- status -------------------------------------------------------------------

function fmtMB(bytes) {
  return `${(bytes / 1e6).toFixed(bytes < 10e6 ? 1 : 0)} MB`;
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
    ui.status.textContent = message || "Ready. Everything runs in this browser tab.";
    return;
  }
  for (const [label, { loaded, total }] of state.progress) {
    const row = document.createElement("div");
    row.className = "progress";
    const pct = total ? Math.min(100, Math.round((100 * loaded) / total)) : 0;
    row.innerHTML = `<span></span><div class="bar"><i style="width:${pct}%"></i></div><small></small>`;
    row.querySelector("span").textContent = label;
    row.querySelector("small").textContent = total ? `${fmtMB(loaded)} / ${fmtMB(total)}` : fmtMB(loaded);
    ui.status.appendChild(row);
  }
}

function setProgress(label, loaded, total) {
  state.progress.set(label, { loaded, total });
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

async function loadIndex(entry) {
  if (state.indexes.has(entry.lang)) return state.indexes.get(entry.lang);
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

async function loadEmbedder() {
  if (state.embedder) return state.embedder;
  const { web_model: model, web_dtype: dtype } = state.manifest;
  const label = `model ${model} (${dtype})`;
  state.embedder = await pipeline("feature-extraction", model, {
    dtype,
    progress_callback: (e) => {
      if (e.status === "progress" && e.file && e.file.endsWith(".onnx")) {
        setProgress(label, e.loaded, e.total);
      }
    },
  });
  doneProgress(label);
  return state.embedder;
}

async function embed(text) {
  const out = await state.embedder(state.manifest.query_prefix + text, {
    pooling: "mean",
    normalize: true,
  });
  return out.data;
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
  try {
    const entry = entryFor(ui.lang.value);
    const [index] = await Promise.all([loadIndex(entry), loadEmbedder()]);
    renderStatus("Embedding...");
    const t0 = performance.now();
    const query = await embed(text);
    const t1 = performance.now();
    const { matches, sims } = rank(index, query, {
      level: ui.level.value,
      topK: Number(ui.topK.value),
    });
    const t2 = performance.now();
    renderGraph(index, matches, sims, entry);
    renderResults(matches);
    ui.meta.textContent =
      `IPC ${entry.version} ${entry.lang} · ${entry.rows.toLocaleString()} entries · ` +
      `${state.manifest.web_model} ${state.manifest.web_dtype} · ` +
      `embed ${Math.round(t1 - t0)} ms · score ${Math.round(t2 - t1)} ms`;
    renderStatus();
    updateUrl(text, entry.lang);
  } catch (err) {
    state.error = `Something failed: ${err.message || err}`;
    renderStatus();
    console.error(err);
  }
}

function renderResults(matches) {
  ui.results.innerHTML = "";
  if (!matches.length) {
    ui.results.textContent = "No result.";
    return;
  }
  const table = document.createElement("table");
  table.innerHTML =
    "<thead><tr><th>#</th><th>Symbol</th><th>Level</th><th>Score</th><th>Sim.</th><th>Section &gt; … &gt; entry</th></tr></thead>";
  const body = document.createElement("tbody");
  matches.forEach((m, i) => {
    const tr = document.createElement("tr");
    const cells = [
      String(i + 1),
      m.pretty,
      m.level,
      m.score.toFixed(3),
      m.similarity.toFixed(3),
      m.text,
    ];
    cells.forEach((v, j) => {
      const td = document.createElement("td");
      td.textContent = v;
      if (j === 1) td.className = "symbol";
      if (j === 5) td.className = "path";
      tr.appendChild(td);
    });
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
      g.append(svgEl("text", { class: "badge", x: n.w + 8, y: G.nodeH / 2 + 4 }, `#${n.rank}`));
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
    lines.push({
      rest: `similarity ${n.match.similarity.toFixed(3)} · path support ${n.match.pathSupport.toFixed(3)}`,
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

function renderExamples() {
  ui.examples.innerHTML = "";
  const langs = new Set(state.manifest.indexes.map((e) => e.lang));
  for (const ex of EXAMPLES.filter((e) => langs.has(e.lang))) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "chip";
    b.textContent = `${ex.lang}: ${ex.text.slice(0, 48)}${ex.text.length > 48 ? "…" : ""}`;
    b.title = ex.text;
    b.addEventListener("click", () => {
      ui.text.value = ex.text;
      ui.lang.value = ex.lang;
      classify();
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
  ui.text.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) classify();
  });
  ui.lang.addEventListener("change", () => loadIndex(entryFor(ui.lang.value)).catch(fail));
  window.addEventListener("scroll", hideTip, { passive: true });

  try {
    await Promise.all([loadIndex(entryFor(ui.lang.value)), loadEmbedder()]);
    renderStatus();
    if (ui.text.value.trim()) classify();
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
