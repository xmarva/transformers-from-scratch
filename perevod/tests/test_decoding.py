from perevod import decoding


def test_a_beam_of_one_is_greedy_decoding(translator, pairs):
    model, tokenizer = translator.model, translator.tokenizer
    greedy = decoding.greedy(model, tokenizer, list(pairs), max_tokens=10)
    beam = decoding.beam_search(model, tokenizer, list(pairs), beam_size=1, max_tokens=10)
    assert beam == greedy


def test_a_batch_translates_like_one_sentence_at_a_time(translator, pairs):
    alone = []
    for sentence in pairs:
        alone.append(translator.translate(sentence))
    assert len(set(alone)) == len(alone)  # otherwise the order of the results can't be seen
    assert translator.translate(list(pairs)) == alone


def test_the_same_input_gives_the_same_translation(translator, pairs):
    assert translator.translate(list(pairs)) == translator.translate(list(pairs))


def test_the_tiny_model_translates_what_it_learned(translator, pairs):
    assert translator.translate(list(pairs)) == list(pairs.values())
