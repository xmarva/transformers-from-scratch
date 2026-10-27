from perevod import decoding

SENTENCES = ["i think therefore i am .", "tom reads a book", "i am ?", "tom"]


def test_a_beam_of_one_is_greedy_decoding(translator):
    model, tokenizer = translator.model, translator.tokenizer
    greedy = decoding.greedy(model, tokenizer, SENTENCES, max_tokens=10)
    beam = decoding.beam_search(model, tokenizer, SENTENCES, beam_size=1, max_tokens=10)
    assert beam == greedy


def test_a_batch_translates_like_one_sentence_at_a_time(translator):
    alone = []
    for sentence in SENTENCES:
        alone.append(translator.translate(sentence))
    assert translator.translate(SENTENCES) == alone


def test_the_same_input_gives_the_same_translation(translator):
    assert translator.translate(SENTENCES) == translator.translate(SENTENCES)
