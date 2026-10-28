// Beam search over the model's two ONNX graphs, step for step like perevod.decoding.beam_search (lesson 3's
// algorithm with lesson 7's length penalty); checked translation for translation against the Python package.
// The decoder graph returns the log-probabilities of the next token for every hypothesis.

const MAX_LEN = 256;  // the model's positions
const BEAM_SIZE = 4;
const ALPHA = 0.6;
const MAX_TOKENS = 128;

function lengthPenalty(length) {
  return ((5 + length) / 6) ** ALPHA;  // Wu et al. (2016)
}

function int64(rows, width) {
  const data = new BigInt64Array(rows.length * width);
  rows.forEach((row, r) => row.forEach((value, c) => { data[r * width + c] = BigInt(value); }));
  return data;
}

function topK(row, offset, size, k) {
  // the k largest log-probabilities of one row, best first; ties go to the smaller token id
  const best = [];
  for (let id = 0; id < size; id++) {
    const value = row[offset + id];
    if (best.length < k || value > best[best.length - 1][1]) {
      let at = best.length;
      while (at > 0 && value > best[at - 1][1]) at--;
      best.splice(at, 0, [id, value]);
      if (best.length > k) best.pop();
    }
  }
  return best;
}

export async function translateIds(ort, encoder, decoder, srcIds, bos, eos) {
  const src = srcIds.slice(0, MAX_LEN);
  const srcTensor = new ort.Tensor("int64", int64([src], src.length), [1, src.length]);
  const memory = (await encoder.run({ src: srcTensor })).memory;
  const [, length, width] = memory.dims;
  let beams = [[[bos], 0]];
  const finished = [];
  for (let step = 0; step < Math.min(MAX_TOKENS, MAX_LEN - 1); step++) {
    const n = beams.length;
    const tgtLength = beams[0][0].length;
    const repeated = new Float32Array(n * length * width);
    for (let i = 0; i < n; i++) repeated.set(memory.data, i * length * width);
    const feeds = {
      tgt: new ort.Tensor("int64", int64(beams.map((b) => b[0]), tgtLength), [n, tgtLength]),
      memory: new ort.Tensor("float32", repeated, [n, length, width]),
      src: new ort.Tensor("int64", int64(Array(n).fill(src), src.length), [n, src.length]),
    };
    const logp = (await decoder.run(feeds)).logp;
    const vocab = logp.dims[1];
    const candidates = [];
    beams.forEach(([tokens, total], row) => {
      for (const [id, value] of topK(logp.data, row * vocab, vocab, BEAM_SIZE)) {
        candidates.push([[...tokens, id], total + value]);
      }
    });
    candidates.sort((a, b) => b[1] - a[1]);  // stable, like Python's sort
    beams = [];
    for (const candidate of candidates) {
      if (candidate[0][candidate[0].length - 1] === eos) finished.push(candidate);
      else beams.push(candidate);
      if (beams.length === BEAM_SIZE) break;
    }
    if (finished.length >= BEAM_SIZE || beams.length === 0) break;
  }
  const pool = finished.length ? finished : beams;
  let best = pool[0];
  for (const candidate of pool) {
    const score = candidate[1] / lengthPenalty(candidate[0].length - 1);
    if (score > best[1] / lengthPenalty(best[0].length - 1)) best = candidate;
  }
  return best[0].slice(1).filter((id) => id !== eos);
}
