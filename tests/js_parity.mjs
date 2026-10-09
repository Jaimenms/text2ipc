// Run the JavaScript scorer on an exported Space directory for a list of cases and
// print the matches as JSON. Driven by tests/test_web.py; needs Node >= 18.
//   node tests/js_parity.mjs <export_dir> <cases.json>
// cases.json: [{"lang": "EN", "query": [floats], "params": {...search params...}}]
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const [exportDir, casesPath] = process.argv.slice(2);
const { buildIndex, parseVectors, search } = await import(
  pathToFileURL(join(exportDir, "scorer.js")).href
);
const manifest = JSON.parse(readFileSync(join(exportDir, "manifest.json"), "utf8"));
const cases = JSON.parse(readFileSync(casesPath, "utf8"));

const indexes = new Map();
function indexFor(lang) {
  if (!indexes.has(lang)) {
    const entry = manifest.indexes.find((e) => e.lang === lang);
    const scheme = JSON.parse(readFileSync(join(exportDir, entry.scheme), "utf8"));
    const raw = readFileSync(join(exportDir, entry.vectors));
    const buffer = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
    const vectors = parseVectors(buffer, {
      rows: entry.rows,
      dim: manifest.dim,
      encoding: entry.encoding,
    });
    indexes.set(lang, buildIndex(scheme, vectors));
  }
  return indexes.get(lang);
}

const out = cases.map((c) =>
  search(indexFor(c.lang), Float32Array.from(c.query), c.params).map((m) => ({
    symbol: m.symbol,
    pretty: m.pretty,
    level: m.level,
    depth: m.depth,
    title: m.title,
    text: m.text,
    score: m.score,
    similarity: m.similarity,
    path_support: m.pathSupport,
    subtree_support: m.subtreeSupport,
  })),
);
process.stdout.write(JSON.stringify(out));
