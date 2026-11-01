import json
import sys
from pathlib import Path

import nbformat
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import build  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "lessons"


@pytest.fixture
def built(tmp_path):
    files = build.build_lesson(FIXTURES / "99_fixture", tmp_path, ref="v9.9.9")
    kaggle = nbformat.reads(files[tmp_path / "kaggle" / "99_fixture" / "notebook.ipynb"], as_version=4)
    colab = nbformat.reads(files[tmp_path / "colab" / "99_fixture.ipynb"], as_version=4)
    meta = json.loads(files[tmp_path / "kaggle" / "99_fixture" / "kernel-metadata.json"])
    return kaggle, colab, meta


def sources(nb):
    return [c.source for c in nb.cells]


def test_each_target_keeps_only_its_own_cells(built):
    kaggle, colab, _ = built
    assert 'platform = "kaggle"' in sources(kaggle) and 'platform = "colab"' not in sources(kaggle)
    assert 'platform = "colab"' in sources(colab) and 'platform = "kaggle"' not in sources(colab)


def test_include_is_expanded_and_ref_substituted(built):
    kaggle, _, _ = built
    assert "## Setup" in sources(kaggle)
    assert 'REF = "v9.9.9"' in sources(kaggle)
    assert not any("{{TFS_REF}}" in s for s in sources(kaggle))


def test_build_tags_are_removed_but_others_kept(built):
    kaggle, _, _ = built
    tags = [t for c in kaggle.cells for t in c.metadata.get("tags", [])]
    assert "solution" in tags
    assert not any(t.startswith(("only-", "include:")) for t in tags)


def test_header_links_to_the_other_platform(built):
    kaggle, colab, _ = built
    assert "colab.research.google.com/github/xmarva/transformers-from-scratch" in kaggle.cells[0].source
    assert "kaggle.com/code/qmarva/transformers-from-scratch-99-fixture" in colab.cells[0].source
    assert "Lesson 99 of" in kaggle.cells[0].source


def test_math_survives_untouched(built):
    kaggle, _, _ = built
    assert r"$$\sum_{i=1}^{n} i = \frac{n(n+1)}{2}$$" in kaggle.cells[1].source


def test_metadata_matches_lesson_needs(built):
    _, colab, meta = built
    assert meta["id"] == "qmarva/transformers-from-scratch-99-fixture"
    assert meta["enable_gpu"] is True and meta["is_private"] is True
    assert colab.metadata["accelerator"] == "GPU"


def test_check_mode_detects_stale_output(tmp_path):
    args = ["--lessons-dir", str(FIXTURES), "--out-dir", str(tmp_path)]
    assert build.main(args) == 0
    assert build.main([*args, "--check"]) == 0
    (tmp_path / "colab" / "99_fixture.ipynb").write_text("{}", encoding="utf-8")
    assert build.main([*args, "--check"]) == 1


def test_ref_moves_images_and_helpers_to_the_same_branch(built):
    kaggle, _, _ = built
    text = "\n".join(sources(kaggle))
    assert f"{build.RAW}/v9.9.9/assets/figures/99/x-light.png" in text
    assert f"{build.RAW}/v9.9.9/assets/covers/99-en.png" in kaggle.cells[0].source
    assert f"{build.RAW}/main/" not in text  # the Colab badge may still point at main: Colab opens released notebooks


def test_kaggle_titles_fit_kaggles_limit():
    for folder in build.lesson_dirs():  # every course's lessons
        meta = build.tomllib.loads((folder / "lesson.toml").read_text(encoding="utf-8"))
        assert len(build.kaggle_title(meta, build.course_of(folder))) <= build.KAGGLE_TITLE_MAX
    with pytest.raises(ValueError, match="at most 50"):
        build.kaggle_title({"number": 3, "short_title": "A title far too long for a Kaggle notebook"})
