// Compares tokenizer.js with the original tokenizer on the cases check_tokenizer.py writes:
//   node check_tokenizer.mjs model/tokenizer.json cases.json
import { readFileSync } from "node:fs";
import { Tokenizer } from "./tokenizer.js";

const [tokenizerPath, casesPath] = process.argv.slice(2);
const tokenizer = new Tokenizer(JSON.parse(readFileSync(tokenizerPath, "utf8")));
const cases = JSON.parse(readFileSync(casesPath, "utf8"));
let encodeDiffers = 0;
for (const [text, ids] of cases.encode) {
  if (JSON.stringify(tokenizer.encode(text)) !== JSON.stringify(ids)) {
    if (encodeDiffers++ < 5) console.log("encode differs:", JSON.stringify(text));
  }
}
let decodeDiffers = 0;
for (const [ids, text] of cases.decode) {
  if (tokenizer.decode(ids) !== text) {
    if (decodeDiffers++ < 5) console.log("decode differs:", JSON.stringify(text));
  }
}
console.log(`encode: ${cases.encode.length - encodeDiffers} of ${cases.encode.length} identical`);
console.log(`decode: ${cases.decode.length - decodeDiffers} of ${cases.decode.length} identical`);
process.exit(encodeDiffers + decodeDiffers > 0 ? 1 : 0);
