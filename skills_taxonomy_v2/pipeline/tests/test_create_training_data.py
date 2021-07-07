import spacy

import pytest

from skills_taxonomy_v2.pipeline.sentence_classifier.create_training_data import (
    mask_text,
    text_cleaning,
    split_labelled_sentences,
    create_training_data,
)

nlp = spacy.load("en_core_web_sm")


def test_mask_text():

    text = "The £1000 which is 23 times the money made at 8am, 8:00am, and 08:00. And now • bullet 1 • bullet 2"
    expected_text = "The £#### which is ## times the money made at ###, ######, and #####. And now , bullet 1 , bullet 2"

    masked_text = mask_text(
        nlp,
        text,
        spacy_ner_types=["DATE", "MONEY", "CARDINAL", "TIME", "ORDINAL", "QUANTITY"],
    )

    assert len(masked_text) == len(text)
    assert masked_text == expected_text


def test_text_cleaning():

    text = "This is #### a ## and ######"
    cleaned_text = text_cleaning(text)
    expected_text = "This is NUMBER a NUMBER and NUMBER"

    assert expected_text == expected_text


def test_split_labelled_sentences():

    text = "this is a skill sentence. and another skill here too. here is a dog though."
    skills_annotations = [
        {"start_offset": 10, "end_offset": 15, "label": 1},
        {"start_offset": 38, "end_offset": 43, "label": 1},
        {"start_offset": 64, "end_offset": 67, "label": 2},
    ]
    sentences, sentences_label = split_labelled_sentences(
        nlp, text, skills_annotations, skill_label_ids=[1]
    )

    assert len(sentences) == len(sentences_label)
    assert len(sentences) == 3
    assert sentences_label == [1, 1, 0]


def test_create_training_data():

    jobs_data = [
        {
            "text": "hi. this is a skill.",
            "annotations": [{"start_offset": 14, "end_offset": 19, "label": 1}],
        },
        {
            "text": "and another skill. hello.",
            "annotations": [{"start_offset": 12, "end_offset": 17, "label": 1}],
        },
    ]
    training_data = create_training_data(
        nlp, jobs_data, sentence_train_threshold=3, skill_label_ids=[1]
    )
    expected_training_data = [
        ("this is a skill.", 1),
        ("and another skill.", 1),
        ("hello.", 0),
    ]

    assert sorted(expected_training_data, key=lambda x: x[0]) == sorted(
        training_data, key=lambda x: x[0]
    )
