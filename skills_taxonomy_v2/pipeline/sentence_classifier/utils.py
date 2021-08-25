# File: sentence_classifier/utils.py

"""
Module for 1) splitting and preprocessing sentences, 2) loading training data and 3) generating
additional features.

"""
# ---------------------------------------------------------------------------------
import string
import nltk
import pickle
from toolz import pipe
from collections import Counter
import re
import numpy as np
import os
import boto3
from s3fs.core import S3FileSystem
from functools import lru_cache

import nltk

nltk.download("averaged_perceptron_tagger")

from skills_taxonomy_v2 import get_yaml_config, Path, PROJECT_DIR, BUCKET_NAME

# ---------------------------------------------------------------------------------

skills_config = get_yaml_config(
    Path(str(PROJECT_DIR) + "/skills_taxonomy_v2/config/base.yaml")
)
S3_PATH = "inputs/labelled_data/"


def text_cleaning(text):
    # Cleaning where it doesnt matter if you mess up the indices
    text = re.sub(r"[#]+", " NUMBER ", text)
    text = re.sub(
        "C NUMBER", "C#", text
    )  # Some situations you shouldn't have removed the numbers
    text = text.replace("\n", " ")  # This could be mid sentences
    # Clean out punctuation - some of this should be cleaned out anyway
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join(text.split())  # get rid of multiple whitespaces
    return text.lower()


def split_sentence(data, nlp, min_length=30):
    """
    Split and clean one sentence.
    Output is two lists, a list of each sentence and a list of the job_ids they are from.
    This has to be in utils.py and not predict_sentence_class.py so it can be used
    with multiprocessing (see https://stackoverflow.com/questions/41385708/multiprocessing-example-giving-attributeerror/42383397)
    """
    # Occassionally this may be filled in as None
    text = data.get("full_text")
    if text:
        sentences = []
        job_id = data.get("job_id")
        # Split up sentences
        doc = nlp(text)
        for sent in doc.sents:
            sentence = text_cleaning(sent.text)
            if len(sentence) > min_length:
                sentences.append(sentence)
        return job_id, sentences
    else:
        return None, None

@lru_cache(maxsize=None)
def load_training_data_from_s3(prefix="final_training_data"):
    """loads data as pickle from S3"""
    s3_file = S3FileSystem()
    file_path = S3_PATH + f"{prefix}.pickle"
    return pickle.load(s3_file.open("{}/{}".format(BUCKET_NAME, file_path)))


def verb_features(sents):
    """
    Generates two additional features.
    Output is numpy array where first value is the number of verbs present in a sentence
    normalised by length of sentence and the second value is a one hot encoding of whether
    the sentence begins with a verb or not.
    """
    verb_feats = []
    for text in sents:
        pos_tags = nltk.pos_tag(nltk.word_tokenize(text))
        verbs = [tag[1].count("VB") for tag in pos_tags]
        starts_with_verbs = np.where(verbs[0] == 1, 1, 0)
        verb_feats.append((sum(verbs) / len(verbs), starts_with_verbs))
    return np.array(verb_feats)
