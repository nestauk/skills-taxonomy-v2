import re

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
    not_skills_words = [
        "job",
        "number",
        "apply",
        "experience",
        "work",
        "detail",
        "full",
        "skill",
        "_",
        "click",
        "hour",
        "contact",
        "about",
        "permalink",
        "excellent",
        "good",
        "strong",
        "title",
        "description",
        "login",
        "register",
        "cv",
        "upload",
        "knowledge",
        "ensure",
        "possible",
        "ability",
    ]

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
        and (not token.text.lower() in stopwords + not_skills_words)
        and (not token.lemma_.lower() in stopwords + not_skills_words)
    )


def process_sentence(sentence, nlp, token_len_threshold, stopwords):

    lemma_sentence_words = []
    tokvecs_i = []

    sentence = separate_camel_case(sentence)

    # Get word embeddings for all words
    doc = nlp(sentence)
    tokvecs = doc._.trf_data.tensors[0][0]
    # See https://github.com/explosion/spaCy/issues/7032 for possible situation
    # where sentence is really long. We have filtered out to not include
    # really long sentences anyway.
    if doc._.trf_data.tensors[0].shape[0] != 1:
        print(
            "Long sentence alert! Tensor has been split so your embedding for this sentence isn't correct"
        )

    for i, token in enumerate(doc):
        if is_token_word(token, token_len_threshold, stopwords):
            # The spacy tokens don't always align to the trf data tokens
            # These are the indices of tokvecs that match to this token
            # (it is in the form array([[18],[19]], dtype=int32)) so needs to be flattened)
            trf_alignment_indices = [
                index for sublist in doc._.trf_data.align[i].data for index in sublist
            ]
            if len(trf_alignment_indices) == 1:
                # I make an assumption that if this is more than 1 it signifies
                # that the word doesnt exist in the vocab, so including
                # the pieces may introduce noise
                # See https://huggingface.co/transformers/glossary.html#input-ids
                lemma_sentence_words.append(token.lemma_.lower())
                tokvecs_i += trf_alignment_indices
    # Sometimes the same index is repeated, e.g. the token 'understanding'
    # maps to the 'understanding' and 'of' trf data, and then 'of' just
    # maps to the 'of' trf data. Hence tokvecs_i may contain repeats.
    tokvecs_i = list(set(tokvecs_i))
    if lemma_sentence_words:
        return lemma_sentence_words, tokvecs[[list(tokvecs_i)]].tolist()
    else:
        return None, None


def process_sentence_mask(
    sentence, nlp, bert_vectorizer, token_len_threshold, stopwords
):
    """
    Mask sentence of stopwords etc, then get sentence embedding
    """

    sentence = separate_camel_case(sentence)

    doc = nlp(sentence)
    masked_sentence = ""
    for i, token in enumerate(doc):
        if is_token_word(token, token_len_threshold, stopwords):
            masked_sentence += " " + token.text
        else:
            masked_sentence += " [MASK]"

    return masked_sentence
