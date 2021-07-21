from skills_taxonomy_v2.pipeline.skills_extraction.cleaning_sentences import (
    separate_camel_case,
)


def is_token_word(token, token_len_threshold, stopwords):
    """
            Returns true if the token:
            - Doesn't contain 'www'
            - Isn't too long (if it is it is usually garbage)
    - Isn't a proper noun/number/quite a few other word types
    - Isn't a word with numbers in (these are always garbage)
    """

    return (
        ("www" not in token.text)
        and (len(token) < token_len_threshold)
        and (
            token.pos_
            not in [
                "PROPN",
                "NUM",
                "SPACE",
                "X",
                "PUNCT",
                "ADP",
                "AUX",
                "CONJ",
                "DET",
                "PART",
                "PRON",
                "SCONJ",
            ]
        )
        and (not re.search("\d", token.text))
        and (not token.text in stopwords)
    )


def process_sentence(sentence, nlp, stopwords):

    lemma_sentence_words = []
    tokvecs_i = []

    sentence = separate_camel_case(sentence)

    # Get word embeddings for all words
    doc = nlp(sentence)
    tokvecs = doc._.trf_data.tensors[0][0]

    for i, token in enumerate(doc):
        if is_token_word(token, token_len_threshold, stopwords):
            lemma_sentence_words.append(token.lemma_.lower())
            # The spacy tokens don't always align to the trf data tokens
            # These are the indices of tokvecs that match to this token
            # (it is in the form array([[18],[19]], dtype=int32)) so needs to be flattened)
            trf_alignment_indices = [
                index for sublist in doc._.trf_data.align[i].data for index in sublist
            ]
            tokvecs_i += trf_alignment_indices
    if lemma_sentence_words:
        return (lemma_sentence_words, tokvecs[[tokvecs_i]].tolist())


output_tuple_list = []
for job_id, sentences in data.items():
    for sentence in sentences:
        sentence_hash = hash(sentence)
        if sentence_hash not in sentence_hash_set:
            sentence_hash_set.add(sentence_hash)

            lemma_sentence_words = []
            tokvecs_i = []

            sentence = separate_camel_case(sentence)

            # Get word embeddings for all words
            doc = nlp(sentence)
            tokvecs = doc._.trf_data.tensors[0][0]

            for i, token in enumerate(doc):
                if is_token_word(token, token_len_threshold, stopwords.words()):
                    lemma_sentence_words.append(token.lemma_.lower())
                    # The spacy tokens don't always align to the trf data tokens
                    # These are the indices of tokvecs that match to this token
                    # (it is in the form array([[18],[19]], dtype=int32)) so needs to be flattened)
                    trf_alignment_indices = [
                        index
                        for sublist in doc._.trf_data.align[i].data
                        for index in sublist
                    ]
                    tokvecs_i += trf_alignment_indices
            if lemma_sentence_words:
                output_tuple_list += [
                    (lemma_sentence_words, tokvecs[[tokvecs_i]].tolist())
                ]
