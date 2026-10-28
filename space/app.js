// The page: loads the tokenizer and the two ONNX graphs once, then translates on this device.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.min.mjs";
import { Tokenizer } from "./tokenizer.js";
import { translateIds } from "./translate.js";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
ort.env.wasm.numThreads = 1;  // a static page isn't cross-origin isolated, so one thread

const $ = (id) => document.getElementById(id);
const FILES = ["model/encoder.onnx", "model/decoder.onnx"];

async function download(url, onBytes) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  const reader = response.body.getReader();
  const chunks = [];
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    onBytes(value.length);
  }
  const bytes = new Uint8Array(chunks.reduce((n, c) => n + c.length, 0));
  let at = 0;
  for (const chunk of chunks) { bytes.set(chunk, at); at += chunk.length; }
  return bytes;
}

async function load() {
  const tokenizer = new Tokenizer(await (await fetch("model/tokenizer.json")).json());
  let loaded = 0;
  const total = 56.0e6;  // the two graphs, about 56 MB
  const progress = (n) => {
    loaded += n;
    $("status").textContent = `Loading the model into your browser: ${Math.min(99, Math.round(100 * loaded / total))}%`;
  };
  const [encoderBytes, decoderBytes] = await Promise.all(FILES.map((f) => download(f, progress)));
  const encoder = await ort.InferenceSession.create(encoderBytes);
  const decoder = await ort.InferenceSession.create(decoderBytes);
  const bos = tokenizer.vocab.get("<bos>");
  const eos = tokenizer.vocab.get("<eos>");
  return async (text) => {
    const clean = text.split(/\s+/u).filter(Boolean).join(" ");  // like perevod: whitespace collapsed
    if (!clean) return "";  // an empty input never reaches the model
    return tokenizer.decode(await translateIds(ort, encoder, decoder, tokenizer.encode(clean), bos, eos));
  };
}

const ready = load();
window.perevod = { ready, translate: async (text) => (await ready)(text) };  // for tests

ready.then(() => {
  $("status").textContent = "The model runs on this device; nothing you type leaves it.";
  $("translate").disabled = false;
}).catch((error) => {
  $("status").textContent = `The model did not load: ${error.message}`;
});

async function run() {
  const button = $("translate");
  button.disabled = true;
  $("output").value = "…";
  const start = performance.now();
  try {
    $("output").value = await window.perevod.translate($("input").value);
    $("timing").textContent = `${Math.round(performance.now() - start)} ms on this device`;
  } finally {
    button.disabled = false;
  }
}

$("translate").addEventListener("click", run);
$("input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) run();
});
for (const example of document.querySelectorAll(".example")) {
  example.addEventListener("click", () => {
    $("input").value = example.textContent;
    if (!$("translate").disabled) run();
  });
}
