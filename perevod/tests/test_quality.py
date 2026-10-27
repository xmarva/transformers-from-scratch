import json
from pathlib import Path

import pandas as pd
import pytest
import sacrebleu
from huggingface_hub import hf_hub_download

from perevod import Translator

# these tests download the published model: `pytest -m "not slow"` skips them
pytestmark = pytest.mark.slow

GOLDEN = Path(__file__).parent / "golden.json"
DATA = "xmarva/tfs-tatoeba-en-ru"
DATA_REVISION = "540be5b93bf77cd0846b02d55f2558a5596887a9"
CHRF_FLOOR = 57.0  # three points under what the published model scores on this sample


@pytest.fixture(scope="module")
def published():
    return Translator.from_pretrained(device="cpu")


def test_the_published_model_translates_as_it_did(published):
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert published.translate(list(golden)) == list(golden.values())


def test_chrf_on_a_sample_of_the_test_set(published):
    path = hf_hub_download(DATA, "data/test.parquet", repo_type="dataset",
                           revision=DATA_REVISION)
    test = pd.read_parquet(path).sample(500, random_state=0)
    chrf = sacrebleu.corpus_chrf(published.translate(list(test.en)), [list(test.ru)]).score
    assert chrf >= CHRF_FLOOR
