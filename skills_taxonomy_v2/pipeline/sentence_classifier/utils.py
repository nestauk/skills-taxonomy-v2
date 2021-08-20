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

from skills_taxonomy_v2 import get_yaml_config, Path, PROJECT_DIR

# ---------------------------------------------------------------------------------

skills_config = get_yaml_config(
    Path(str(PROJECT_DIR) + "/skills_taxonomy_v2/config/base.yaml")
)
training_data_path = str(PROJECT_DIR) + skills_config["TRAINING_DATA_PATH"]


def lowercase(text):
    """Converts all text to lowercase"""
    return text.lower()


def mask_numbers(text):
    """masks numbers in sentence as NUMBER"""
    return re.sub(r"[#]+", " NUMBER ", text)


def remove_punctuation(text):
    """remove sentence punctuation"""
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)


def remove_trailing_space(text):
    """get rid of multiple whitespaces"""
    return " ".join(text.split())


def remove_short_sents(text):
    """keep sents within a range of 30 - 100 chars"""
    if 30 < len(text) < 100:
        return text


def clean_text(text, training=False):
    """
    Pipeline for preprocessing already split skill and non-skill sentences.
    remove short sents in training data otherwise follow Liz's pipeline for
    removing sents of a certain length. 
    
    """
    if training is True:
        return pipe(
            text,
            lowercase,
            mask_numbers,
            remove_trailing_space,
            remove_punctuation,
            remove_short_sents,
        )

    elif training is False:
        return pipe(
            text, lowercase, mask_numbers, remove_trailing_space, remove_punctuation,
        )


def split_sentence(data, nlp, min_length=15, max_length=100):
    """
    Split and clean one sentence. 
    Output is two lists, a list of each sentence and a list of the job_ids they are from.
    This has to be in utils.py and not predict_sentence_class.py so it can be used
    with multiprocessing (see https://stackoverflow.com/questions/41385708/multiprocessing-example-giving-attributeerror/42383397)
    """
    text = data.get("full_text")
    # Occassionally this may be filled in as None
    if text:
        sentences = []
        job_id = data.get("job_id")
        # Split up sentences
        doc = nlp(text)
        for sent in doc.sents:
            sentence = clean_text(sent.text, training=False)
            if len(sentence) in range(min_length, max_length):
                sentences.append(sentence)
        return job_id, sentences
    else:
        return None, None


def load_training_data(training_data_file_name):
    """
    loads updated training data file and prints number of skill and non-skill sentences.
    """
    with open(
        training_data_path + "/" + training_data_file_name + ".pickle", "rb"
    ) as h:
        training_data = pickle.load(h)
    print(Counter([label[2] for label in training_data]))

    return training_data


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
