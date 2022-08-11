"""
For a particular S3 location of skill sentences found, load the sentences, and for each sentence
find its words embeddings and output these.
- Don't include very long words
- Don't include proper nouns/numbers/quite a few other word types
- Don't include words with numbers in (these are always garbage)
- You generally take out a lot of the urls by having a token_len_threshold but not always

From this point we don't really care which job adverts the sentences come from,
there will also be repeated sentences to remove. Although we will process in batches
of the original data files, so repeats won't be removed cross files.

python -i skills_taxonomy_v2/pipeline/skills_extraction/get_sentence_embeddings.py
--config_path 'skills_taxonomy_v2/config/skills_extraction/2021.08.02.yaml'

"""

import pickle
import re
import json
import time
import os
from argparse import ArgumentParser
import yaml
import logging
from functools import partial

from tqdm import tqdm
import boto3
import spacy
import torch
import numpy
import nltk
from nltk.corpus import stopwords
import numpy as np

from skills_taxonomy_v2.getters.s3_data import (
    get_s3_resource,
    get_s3_data_paths,
    save_to_s3,
    load_s3_data,
)

from skills_taxonomy_v2.pipeline.skills_extraction.get_sentence_embeddings_utils import (
    process_sentence_mask,
)
from skills_taxonomy_v2.pipeline.sentence_classifier.sentence_classifier import (
    BertVectorizer,
)

from skills_taxonomy_v2 import BUCKET_NAME, custom_stopwords_dir

nltk.download("stopwords")

logger = logging.getLogger(__name__)

# Get the custom stop words - words which shouldn't be included
# when finding skills, e.g. 'skill', 'job'
with open(custom_stopwords_dir) as file:
    custom_stopwords = file.read().splitlines()

def get_embeddings(data, nlp, bert_vectorizer, token_len_threshold=20, stopwords=stopwords.words(), custom_stopwords=custom_stopwords):
    logger.info(f"Processing {len(data)} sentences...")
    start_time = time.time()

    # For each sentence mask out stop words, proper nouns etc.
    masked_sentences = []
    sentence_job_ids = []
    sentence_hashes = []
    original_sentences = {}

    for job_id, sentences in tqdm(data.items()):
        for sentence in sentences:
            masked_sentence = process_sentence_mask(
                sentence,
                nlp,
                bert_vectorizer,
                token_len_threshold,
                stopwords=stopwords,
                custom_stopwords=custom_stopwords,
            )
            if masked_sentence.replace("[MASK]", "").replace(" ", ""):
                # Don't include sentence if it only consists of masked words
                masked_sentences.append(masked_sentence)
                sentence_job_ids.append(job_id)
                # Keep a record of the original sentence via a hashed id
                original_sentence_id = hash(sentence)
                sentence_hashes.append(original_sentence_id)
                original_sentences[original_sentence_id] = sentence
    logger.info(f"Processing sentences took {time.time() - start_time} seconds")

    logger.info(f"Getting embeddings for {len(masked_sentences)} sentences...")
    start_time = time.time()
    # Find sentence embeddings in bulk for all masked sentences
    masked_sentence_embeddings = bert_vectorizer.transform(masked_sentences)
    output_tuple_list = [
        (job_id, sent_id, sent, emb.tolist())
        for job_id, sent_id, sent, emb in zip(
            sentence_job_ids,
            sentence_hashes,
            masked_sentences,
            masked_sentence_embeddings,
        )
    ]
    logger.info(f"Getting embeddings took {time.time() - start_time} seconds")
    return output_tuple_list, original_sentences

def parse_arguments(parser):
    parser.add_argument(
        "--config_path",
        help="Path to config file",
        default="skills_taxonomy_v2/config/skills_extraction/2021.11.05.yaml",
    )
    return parser.parse_args()


def try_less_data(s3, bucket_name, data, output_file_dir, stop_recursion=False):
    """
    Not an elegant solution - but if the data is too large (i think >5GB) you get
    an error:
    botocore.exceptions.ClientError: An error occurred (EntityTooLarge) when
    calling the PutObject operation: Your proposed upload exceeds the maximum allowed size

    So this function will cut down your data if this happens.
    50000 data points of the type in this function should be ok, and if not
    it will try with 10000 too (which should definitely work).
    """

    data = data[0:50000]
    try:
        save_to_s3(s3, bucket_name, data, output_file_dir)
    except:
        data = data[0:10000]
        if not stop_recursion:
            try_less_data(s3, bucket_name, data, output_file_dir, stop_recursion=True)
        else:
            logger.info(f"Wasn't able to save {output_file_dir}")


if __name__ == "__main__":

    # Load config variables

    parser = ArgumentParser()
    args = parse_arguments(parser)

    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    FLOW_ID = "word_embeddings_flow"
    flow_config = config["flows"][FLOW_ID]
    params = flow_config["params"]

    skill_sentences_dir = params["skill_sentences_dir"]
    token_len_threshold = params["token_len_threshold"]
    # Use config file name as a date stamp for the output dir
    output_dir = os.path.join(
        params["output_dir"], os.path.basename(args.config_path).replace(".yaml", "")
    )

    nlp = spacy.load("en_core_web_sm")
    bert_vectorizer = BertVectorizer(
        bert_model_name="sentence-transformers/all-MiniLM-L6-v2",
        multi_process=True,
    )
    bert_vectorizer.fit()

    # Get data paths in the location
    s3 = boto3.resource("s3")
    data_paths = get_s3_data_paths(
        s3, BUCKET_NAME, skill_sentences_dir, file_types=["*.json"]
    )

    # For loop through each data path
    logger.info(f"Running predictions on {len(data_paths)} data files ...")

    for data_path_i, data_path in enumerate(data_paths):
        logger.info(
            f"Loading data for {data_path} ({data_path_i} of {len(data_paths)}) ..."
        )
        data = load_s3_data(s3, BUCKET_NAME, data_path)

        output_tuple_list, original_sentences = get_embeddings(data, nlp, bert_vectorizer, token_len_threshold=token_len_threshold, stopwords=stopwords.words(), custom_stopwords=custom_stopwords)

        # Save the output in a folder with a similar naming structure to the input
        data_dir = os.path.relpath(data_path, skill_sentences_dir)
        output_file_dir = os.path.join(
            output_dir, data_dir.split(".json")[0] + "_embeddings.json"
        )
        try:
            save_to_s3(s3, BUCKET_NAME, output_tuple_list, output_file_dir)
        except:
            try_less_data(s3, BUCKET_NAME, output_tuple_list, output_file_dir)

        sent_id_dir = os.path.join(
            output_dir, data_dir.split(".json")[0] + "_original_sentences.json"
        )
        save_to_s3(s3, BUCKET_NAME, original_sentences, sent_id_dir)

        logger.info(f"Saved output to {output_file_dir}")
