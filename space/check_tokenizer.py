# Checks the page's JavaScript tokenizer against the original, id for id: every English sentence of the
# test set and a few awkward inputs are encoded by both, and the Russian references decoded by both.
# Needs Node.js. Usage: python check_tokenizer.py --model model
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

DATA_REVISION = "540be5b93bf77cd0846b02d55f2558a5596887a9"
AWKWARD = ["", "   ", "naïve café", "Москва", "It's 5 o'clock — isn't it?", "“Quotes” and «guillemets»",
           "tabs\tand\nnew lines", "emoji 🙂 and CJK 漢字", "é combining", "a - b – c"]


def main():
    parser = argparse.ArgumentParser(description="Compare tokenizer.js with the original")
    parser.add_argument("--model", default="model")
    tokenizer_path = Path(parser.parse_args().model) / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    test_path = hf_hub_download("xmarva/tfs-tatoeba-en-ru", "data/test.parquet", repo_type="dataset",
                                revision=DATA_REVISION)
    test = pd.read_parquet(test_path)
    encode = []
    for text in test["en"].tolist() + AWKWARD:
        encode.append([text, tokenizer.encode(text).ids])
    decode = []
    for text in test["ru"].tolist():
        ids = tokenizer.encode(text).ids
        decode.append([ids, tokenizer.decode(ids)])
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as cases:
        json.dump({"encode": encode, "decode": decode}, cases, ensure_ascii=False)
    here = Path(__file__).parent
    subprocess.run(["node", str(here / "check_tokenizer.mjs"), str(tokenizer_path), cases.name], check=True)


if __name__ == "__main__":
    main()
