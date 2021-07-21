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

The indices need special attention because the output tokens of doc._.trf_data are
different from the tokens in doc. You can use doc._.trf_data.align[i].data to find
how they relate.
https://stackoverflow.com/questions/66150469/spacy-3-transformer-vector-token-alignment

python -i skills_taxonomy_v2/pipeline/skills_extraction/get_word_embeddings.py --config_path 'skills_taxonomy_v2/config/skills_extraction/word_embeddings/2021.07.21.yaml'

"""

import pickle
import re
import json
import os
from argparse import ArgumentParser
import yaml
import logging
from multiprocessing import Pool
from functools import partial

from tqdm import tqdm
import boto3
import cupy
import spacy
import torch
import numpy
from thinc.api import set_gpu_allocator, require_gpu
import nltk
from nltk.corpus import stopwords

from skills_taxonomy_v2.getters.s3_data import (
    get_s3_resource,
    get_s3_data_paths,
    save_to_s3,
    load_s3_data,
)

from skills_taxonomy_v2.pipeline.skills_extraction.get_word_embeddings_utils.py import (
    process_sentence,
)

nltk.download("stopwords")

logger = logging.getLogger(__name__)


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


def parse_arguments(parser):
    parser.add_argument(
        "--config_path",
        help="Path to config file",
        default="skills_taxonomy_v2/config/skills_extraction/word_embeddings/2021.07.21.yaml",
    )
    return parser.parse_args()


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
    output_dir = params["output_dir"]

    # Use the GPU, with memory allocations directed via PyTorch.
    # This prevents out-of-memory errors that would otherwise occur from competing
    # memory pools.
    set_gpu_allocator("pytorch")
    require_gpu(0)

    # Get embeddings for each token in the sentence
    nlp = spacy.load("en_core_web_trf")

    # Get data paths in the location
    bucket_name = "skills-taxonomy-v2"
    s3 = boto3.resource("s3")
    data_paths = get_s3_data_paths(
        s3, bucket_name, skill_sentences_dir, file_types=["*.json"]
    )

    # For loop through each data path
    logger.info(f"Running predictions on {len(data_paths)} data files ...")

    for data_path in data_paths:
        logger.info(f"Loading data for {data_path} ...")
        data = load_s3_data(s3, bucket_name, data_path)
        output_tuple_list = []
        for job_id, sentences in data.items():
            with Pool(4) as pool:  # 4 cpus
                partial_process_sentence = partial(
                    process_sentence, nlp=nlp, stopwords=stopwords.words()
                )
                process_sentence_pool_output = pool.map(
                    partial_process_sentence, sentences
                )
            output_tuple_list += process_sentence_pool_output

        # Save the output in a folder with a similar naming structure to the input
        data_dir = os.path.relpath(data_path, skill_sentences_dir)
        output_file_dir = os.path.join(
            output_dir, data_dir.split(".json")[0] + "_embeddings.json"
        )
        save_to_s3(s3, bucket_name, output_tuple_list, output_file_dir)
        print(f"Saved output to {output_file_dir}")
