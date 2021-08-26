# File: sentence_classifier/create_training_data.py

"""
Module to create training data using karlis's and label studio's data and
update some bad labels.

NOTE: I manually updated some labels on multiple occasions (at one point using model predictions to assist in relabelling) but struggled to find all CSVs.
I have used one new_labels.csv to illustrate the pipeline.

Example Usage:
	python skills_taxonomy_v2/pipeline/sentence_classifier/create_training_data.py

"""
# ---------------------------------------------------------------------------------
import spacy
from tqdm import tqdm
import os
import pandas as pd

import pickle
import json
import re
from pigeonXT import annotate
import boto3
from s3fs.core import S3FileSystem
from functools import lru_cache

from skills_taxonomy_v2 import get_yaml_config, Path, PROJECT_DIR, BUCKET_NAME

from skills_taxonomy_v2.pipeline.sentence_classifier.utils import (
    text_cleaning,
)

# ---------------------------------------------------------------------------------

skills_config = get_yaml_config(
    Path(str(PROJECT_DIR) + "/skills_taxonomy_v2/config/base.yaml")
)
training_data_path = str(PROJECT_DIR) + skills_config["TRAINING_DATA_PATH"]

S3_PATH = "inputs/labelled_data/"


@lru_cache(maxsize=None)
def load_raw_data_from_s3():
    s3_file = S3FileSystem()
    filenames = [
        "label_studio_labelled_data.json",
        "OJO_test_labelling_April2021_jobs.jsonl",
    ]
    data = []
    for file in filenames:
        file_path = S3_PATH + file
        data.append(
            [
                json.loads(line)
                for line in s3_file.open("{}/{}".format(BUCKET_NAME, file_path))
            ]
        )
    return data


def split_labelled_sentences(nlp, text, skills_annotations, skill_label_ids=[1, 5]):
    """
    Tag the sentences within one job advert text as having skills in (1) or not (0).
    - Get job advert text and corresponding annotations of skills within it
    - Split up sentences and check if they contained any skill labels
    - Create list of sentences with corresponding list of tags
    """

    skill_spans = [
        (label["start_offset"], label["end_offset"])
        for label in skills_annotations
        if label["label"] in skill_label_ids
    ]

    # Split up sentences
    skill_span_sets = [set(range(s, e)) for s, e in skill_spans]
    doc = nlp(text)
    sentences = []
    sentences_label = []

    for sent in doc.sents:
        sentence_set = set(range(sent.start_char, sent.end_char))
        split_i = [0] + [
            m.start(0) + 1 for m in re.finditer(r"[a-z][A-Z][a-z]", sent.text)
        ]
        for i, j in zip(split_i, split_i[1:] + [len(sent.text)]):
            sentences.append(sent.text[i:j])
            # Shift i and j by startchar
            sentence_set = set(range(i + sent.start_char, j + sent.start_char))
            # Is there overlap between the sentence range and any of the entity ranges?
            if any(
                [entity_set.issubset(sentence_set) for entity_set in skill_span_sets]
            ):
                sentences_label.append(1)
            else:
                sentences_label.append(0)

    return sentences, sentences_label


def split_labelled_sentences_label_studio(nlp, text, skills_annotations):
    """
    Tag the sentences within one job advert text as having skills in (1) or not (0).

    Modified slightly to account for the structure of Label Studio data.

    - Get job advert text and corresponding annotations of skills within it
    - Split up sentences and check if they contained any skill labels
    - Create list of sentences with corresponding list of tags
    """

    skill_spans = [
        (skills_span["value"]["start"], skills_span["value"]["end"])
        for index, skills_span in enumerate(skills_annotations)
        if "skill" in skills_span["value"]["labels"]
    ]

    # Split up sentences
    skill_span_sets = [set(range(s, e)) for s, e in skill_spans]
    doc = nlp(text)
    sentences = []
    sentences_label = []

    for sent in doc.sents:
        sentence_set = set(range(sent.start_char, sent.end_char))
        split_i = [0] + [
            m.start(0) + 1 for m in re.finditer(r"[a-z][A-Z][a-z]", sent.text)
        ]
        for i, j in zip(split_i, split_i[1:] + [len(sent.text)]):
            sentences.append(sent.text[i:j])
            # Shift i and j by startchar
            sentence_set = set(range(i + sent.start_char, j + sent.start_char))
            # Is there overlap between the sentence range and any of the entity ranges?
            if any(
                [entity_set.issubset(sentence_set) for entity_set in skill_span_sets]
            ):
                sentences_label.append(1)
            else:
                sentences_label.append(0)

    return sentences, sentences_label


def create_training_data(
    nlp, jobs_data, sentence_train_threshold=10, skill_label_ids=[1, 5]
):
    """
    Label all job adverts and add them all to a list of training data.
    sentence_train_threshold: A threshold of how big the sentence has to be in order to include it in the training/test data
    """

    # Create training dataset

    karlis_data = jobs_data[1]
    ls_data = jobs_data[0][0]

    training_data = []
    for job_info in karlis_data:
        text = job_info["text"]
        skills_annotations = job_info["annotations"]
        sentences, sentences_label = split_labelled_sentences(
            nlp, text, skills_annotations, skill_label_ids=skill_label_ids
        )
        for sentence, sentence_label in zip(sentences, sentences_label):
            if len(sentence) > sentence_train_threshold:
                training_data.append((text_cleaning(sentence), sentence_label))

    for job_info in ls_data:
        text = job_info["data"]["full_text"]
        skills_annotations = job_info["annotations"][0]["result"]
        sentences, sentences_label = split_labelled_sentences_label_studio(
            nlp, text, skills_annotations
        )
        for sentence, sentence_label in zip(sentences, sentences_label):
            if len(sentence) > sentence_train_threshold:
                training_data.append((text_cleaning(sentence), sentence_label))

    return training_data


def update_bad_training_data_labels(
    training_data,
):
    """
    Update bad labels in training data using a jupyter notebook widget.

    Saves csv to s3.
    """
    file_path = S3_PATH + "new_labels.csv"
    new_labels = annotate(training_data, options=["1", "0"])
    new_labels.to_csv(os.path.join("s3://", BUCKET_NAME, file_path), index=False)


def update_training_data(training_data, new_labels="new_labels.csv"):
    """
    Removes bad labels in training data and adds updated labels.
    """
    s3_file = S3FileSystem()
    file_path = S3_PATH + new_labels
    new_labels = pd.read_csv(s3_file.open("{}/{}".format(BUCKET_NAME, file_path)))

    updated_labels = []
    for index, row in new_labels.iterrows():
        if row["changed"] == True:
            updated_labels.append((row["example"].lower(), int(row["label"])))

    updated_training = []
    for old_label in training:
        for new_label in updated_labels:
            if old_label[0] == new_label[0]:
                updated_training.append(new_label)
                break
        else:
            updated_training.append(old_label)

    return updated_training


def save_training_data(training_data, output_file):

    training_path = os.path.join(training_data_path, output_file + ".pickle")

    with open(training_path, "wb") as file:
        pickle.dump(training_data, file)


if __name__ == "__main__":

    jobs_data = load_raw_data_from_s3()
    nlp = spacy.load("en_core_web_sm")
    training = create_training_data(nlp, jobs_data, sentence_train_threshold=10)
    updated_training = update_training_data(training)
    # save_training_data(updated_training, 'prelim_labels')
