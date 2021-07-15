"""
Input sentences output word embeddings for each of them
"""

import pickle
import re
import json
import os

from tqdm import tqdm

# from transformers import BertTokenizer, BertModel

from skills_taxonomy_v2.pipeline.skills_extraction.cleaning_sentences import (
    separate_camel_case,
    deduplicate_sentences,
)

import cupy
import spacy
import torch
import numpy
from thinc.api import set_gpu_allocator, require_gpu
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

if __name__ == "__main__":

    # Use the GPU, with memory allocations directed via PyTorch.
    # This prevents out-of-memory errors that would otherwise occur from competing
    # memory pools.
    set_gpu_allocator("pytorch")
    require_gpu(0)

    config_name = "2021.07.09.small"

    with open(
        f"outputs/sentence_classifier/data/skill_sentences/{config_name}_jobsnew1_predictions.pkl",
        "rb",
    ) as file:
        sentences_pred = pickle.load(file)

    with open(
        f"outputs/sentence_classifier/data/skill_sentences/{config_name}_jobsnew1_sentences.pkl",
        "rb",
    ) as file:
        sentences = pickle.load(file)

    with open(
        f"outputs/sentence_classifier/data/skill_sentences/{config_name}_jobsnew1_job_ids.pkl",
        "rb",
    ) as file:
        job_ids = pickle.load(file)

    token_len_threshold = 20  # To adjust. I had a look and > this number seems to all be not words (urls etc)

    # Filter out the non-skill sentences
    sentences = [
        sentences[i] for i, p in enumerate(sentences_pred.astype(bool)) if p == 1
    ]

    # Deduplicate sentences
    sentences = deduplicate_sentences(sentences)

    # Get embeddings for each token in the sentence
    nlp = spacy.load("en_core_web_trf")

    output_directory = "outputs/skills_extraction/data/"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    print(f"Finding word embeddings for {len(sentences)} sentences ...")
    for job_id, sentence in tqdm(zip(job_ids, sentences)):
        sentence = separate_camel_case(sentence)
        doc = nlp(sentence)
        # Ignore start and end padding vectors of sentence words
        tokvecs = doc._.trf_data.tensors[0][0][1:-1]
        lemma_sentence = []
        clean_sentence = []
        tokvecs_i = []
        for i, token in enumerate(doc):
            # Don't include very long words
            # or proper nouns/numbers
            # or words with numbers in (these are always garbage)
            # You generally take out a lot of the urls by having a token_len_threshold but not always
            if (
                ("www" not in token.text)
                and (len(token) < token_len_threshold)
                and (token.pos_ not in ["PROPN", "NUM", "SPACE"])
                and (not re.search("\d", token.text))
                and (not token.text in stopwords.words())
            ):
                lemma_sentence.append(token.lemma_.lower())
                clean_sentence.append(token.text.lower())
                tokvecs_i.append(i)
        if clean_sentence:
            output_line = {
                "job_id": job_id,
                "cleaned sentence": clean_sentence,
                "lemmatized sentence": lemma_sentence,
                "word embeddings": tokvecs[tokvecs_i].tolist(),
            }
            with open(
                os.path.join(
                    output_directory,
                    f"{config_name}_jobsnew1_transformers_word_embeddings.jsonl",
                ),
                "a",
            ) as file:
                file.write(json.dumps(output_line))
                file.write("\n")
