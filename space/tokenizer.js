// The model's tokenizer (tokenizer.json from Hugging Face `tokenizers`), rewritten for the browser:
// the same normalizer, pre-tokenizer, BPE and decoder, checked token for token against the original.

const QUOTES = [["«", '"'], ["»", '"'], ["„", '"'], ["“", '"'], ["”", '"'], ["‘", "'"], ["’", "'"], ["–", "—"]];
// Rust's \w is Unicode-aware; JavaScript's isn't, even with the u flag, so it is spelled out
const WORD = "\\p{Alphabetic}\\p{M}\\p{Nd}\\p{Pc}\\p{Join_Control}";
const PIECE = new RegExp(` ?[${WORD}]+| ?[^${WORD}\\s]+`, "gu");

export class Tokenizer {
  constructor(spec) {
    this.vocab = new Map(Object.entries(spec.model.vocab));
    this.tokens = [];
    for (const [token, id] of this.vocab) this.tokens[id] = token;
    this.ranks = new Map();
    spec.model.merges.forEach((pair, rank) => {
      const [a, b] = Array.isArray(pair) ? pair : pair.split(" ");
      this.ranks.set(a + "\u0000" + b, rank);
    });
    this.special = new Set(spec.added_tokens.filter((t) => t.special).map((t) => t.id));
    this.unk = this.vocab.get(spec.model.unk_token);
  }

  normalize(text) {
    let s = text.normalize("NFKC").replace(/\p{Mn}/gu, "");
    for (const [from, to] of QUOTES) s = s.split(from).join(to);
    s = s.replace(/\s+-\s+/gu, " — ").replace(/\s+/gu, " ").trim();
    return s ? " " + s : "";  // Prepend leaves an empty text empty
  }

  pieces(normalized) {
    // Split, isolated: the matches, and whatever lies between them, each a piece of its own
    const out = [];
    let last = 0;
    for (const match of normalized.matchAll(PIECE)) {
      if (match.index > last) out.push(normalized.slice(last, match.index));
      out.push(match[0]);
      last = match.index + match[0].length;
    }
    if (last < normalized.length) out.push(normalized.slice(last));
    return out.map((piece) => piece.replaceAll(" ", "▁"));  // Metaspace, nothing prepended
  }

  bpe(piece) {
    let symbols = Array.from(piece);
    while (symbols.length > 1) {
      let best = -1;
      let bestRank = Infinity;
      for (let i = 0; i + 1 < symbols.length; i++) {
        const rank = this.ranks.get(symbols[i] + "\u0000" + symbols[i + 1]);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          best = i;
        }
      }
      if (best < 0) break;
      symbols = [...symbols.slice(0, best), symbols[best] + symbols[best + 1], ...symbols.slice(best + 2)];
    }
    return symbols.map((symbol) => this.vocab.get(symbol) ?? this.unk);
  }

  encode(text) {
    const ids = [];
    for (const piece of this.pieces(this.normalize(text))) ids.push(...this.bpe(piece));
    return ids;
  }

  decode(ids) {
    const words = ids.filter((id) => !this.special.has(id)).map((id) => this.tokens[id].replaceAll("▁", " "));
    if (words.length && words[0].startsWith(" ")) words[0] = words[0].slice(1);
    return words.join("");
  }
}
