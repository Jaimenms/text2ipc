// Web Worker for the second stage: loads the cross-encoder and judges (text, path)
// pairs off the main thread, so the page stays responsive and can show progress.
// Messages in: {id, type: "load", modelId, dtype} and
//              {id, type: "judge", text, texts, maxLength, batch}.
// Messages out: {type: "progress", loaded, total} while the model downloads,
//               {type: "judging", done, total} after every batch,
//               {id, type: "loaded"} | {id, type: "done", logits} | {id, type: "error", message}.
import {
  AutoTokenizer,
  XLMRobertaModel,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1";

env.allowLocalModels = false;

let tokenizer = null;
let model = null;
let loading = null;

self.onmessage = async (e) => {
  const { id, type } = e.data;
  try {
    if (type === "load") {
      await load(e.data);
      self.postMessage({ id, type: "loaded" });
    } else if (type === "judge") {
      await load(e.data);
      self.postMessage({ id, type: "done", logits: await judge(e.data) });
    }
  } catch (err) {
    self.postMessage({ id, type: "error", message: (err && err.message) || String(err) });
  }
};

function load({ modelId, dtype }) {
  if (model) return Promise.resolve();
  if (!loading) {
    loading = (async () => {
      const progress = (p) => {
        if (p.status === "progress" && p.file && p.file.endsWith(".onnx")) {
          self.postMessage({ type: "progress", loaded: p.loaded, total: p.total });
        }
      };
      // jina's config carries no model_type, so the class is named, as its model card does
      const attempt = async () => {
        tokenizer = await AutoTokenizer.from_pretrained(modelId);
        model = await XLMRobertaModel.from_pretrained(modelId, { dtype, progress_callback: progress });
      };
      try {
        await attempt();
        self.postMessage({ type: "ready" });
      } catch (err) {
        // Some browsers cannot store a 280 MB response in the Cache API and the
        // loader fails with a network error; retry without the cache.
        console.warn("reranker load failed, retrying without the browser cache", err);
        env.useBrowserCache = false;
        try {
          await attempt();
          self.postMessage({ type: "ready" });
        } finally {
          env.useBrowserCache = true;
        }
      }
    })().catch((err) => {
      loading = null;
      throw err;
    });
  }
  return loading;
}

async function judge({ text, texts, maxLength, batch }) {
  const logits = [];
  const size = batch || 4;
  for (let i = 0; i < texts.length; i += size) {
    const part = texts.slice(i, i + size);
    const inputs = tokenizer(part.map(() => text), {
      text_pair: part,
      padding: true,
      truncation: true,
      max_length: maxLength,
    });
    const out = await model(inputs);
    for (let j = 0; j < part.length; j++) logits.push(Number(out.logits.data[j]));
    self.postMessage({ type: "judging", done: logits.length, total: texts.length });
  }
  return logits;
}
