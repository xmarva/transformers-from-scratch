import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import export_post  # noqa: E402


def test_a_labelled_quotation_is_a_callout_of_its_kind():
    key = export_post.components("> **Key point.** Attention mixes;\n> it doesn't pick.")
    assert key.startswith('<div class="callout callout-key" markdown="1">\n<p class="callout-label">Key point</p>')
    assert "Attention mixes;\nit doesn't pick.\n</div>" in key
    assert 'class="callout callout-warn"' in export_post.components("> **Trap.** A loss is not a metric.")
    plain = export_post.components("> **For engineers.** Pin the version.")
    assert '<div class="callout" markdown="1">' in plain and ">For engineers</p>" in plain


def test_a_quotation_that_names_its_author_is_a_pull_quote():
    pull = export_post.components("> Two in one.\n>\n> — Robertson")
    assert pull == '<blockquote class="pull" markdown="1">\nTwo in one.\n\n<cite>Robertson</cite>\n</blockquote>'
    assert export_post.components("> An example sentence.") == "> An example sentence."


def test_takeaway_and_flow():
    assert export_post.components("→ It finds 51%.") == '<p class="takeaway" markdown="1">It finds 51%.</p>'
    flow = export_post.components("tokens → vectors → scores")
    assert flow == '<p class="flow"><span>tokens</span><span>vectors</span><span>scores</span></p>'


def test_arrows_inside_a_sentence_or_code_stay():
    menu = "On Kaggle, *Settings → Accelerator → GPU*; then run."
    assert export_post.components(menu) == menu
    code = "```\n> **Key point.** no\n\na → b → c\n```"
    assert export_post.components(code) == code
