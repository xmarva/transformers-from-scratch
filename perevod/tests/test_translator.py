import pytest


def test_an_empty_input_has_an_empty_translation(translator):
    assert translator.translate(["", "   ", "tom reads"])[:2] == ["", ""]


def test_a_string_gives_a_string_and_a_list_gives_a_list(translator):
    assert isinstance(translator.translate("tom reads"), str)
    assert isinstance(translator.translate(["tom reads"]), list)


def test_spaces_and_line_breaks_change_nothing(translator):
    expected = translator.translate("tom reads a book")
    assert translator.translate("tom  reads\na book") == expected


def test_a_long_input_is_cut_with_a_warning(translator):
    long_text = " ".join(["tom reads a book ."] * 10)  # 50 tokens, the model has 32 places
    with pytest.warns(UserWarning, match="cut"):
        translation = translator.translate(long_text)
    assert isinstance(translation, str)
