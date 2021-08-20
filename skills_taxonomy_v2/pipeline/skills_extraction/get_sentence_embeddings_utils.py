"""
Functions to mask sentences of undesirable words (stopwords, punctuation etc).
Used in get_sentence_embeddings.py to process sentences before finding embeddings.
"""
import re

from skills_taxonomy_v2.pipeline.skills_extraction.cleaning_sentences import (
    separate_camel_case,
)


def is_token_word(token, token_len_threshold, stopwords, custom_stopwords):
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
        and (not token.text.lower() in stopwords + custom_stopwords)
        and (not token.lemma_.lower() in stopwords + custom_stopwords)
    )


def process_sentence_mask(
    sentence, nlp, bert_vectorizer, token_len_threshold, stopwords, custom_stopwords
):
    """
    Mask sentence of stopwords etc, then get sentence embedding
    """

    sentence = separate_camel_case(sentence)

    doc = nlp(sentence)
    masked_sentence = ""
    for i, token in enumerate(doc):
        if is_token_word(token, token_len_threshold, stopwords, custom_stopwords):
            masked_sentence += " " + token.text
        else:
            masked_sentence += " [MASK]"

    return masked_sentence
