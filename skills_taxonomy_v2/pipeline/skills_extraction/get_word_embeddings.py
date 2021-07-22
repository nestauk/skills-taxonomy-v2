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
import numpy as np

from skills_taxonomy_v2.getters.s3_data import (
    get_s3_resource,
    get_s3_data_paths,
    save_to_s3,
    load_s3_data,
)

from skills_taxonomy_v2.pipeline.skills_extraction.get_word_embeddings_utils import (
    process_sentence,
)

nltk.download("stopwords")

logger = logging.getLogger(__name__)


def parse_arguments(parser):
    parser.add_argument(
        "--config_path",
        help="Path to config file",
        default="skills_taxonomy_v2/config/skills_extraction/word_embeddings/2021.07.21.yaml",
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
        logger.info(f"Predicting embeddings for {len(data)} sentences...")
        output_tuple_list = []
        for job_id, sentences in tqdm(data.items()):
            for sentence in sentences:
                clean_sentences, sentence_embeddings = process_sentence(
                    sentence,
                    nlp=nlp,
                    token_len_threshold=token_len_threshold,
                    stopwords=stopwords.words(),
                )
                if clean_sentences:
                    mean_sentence_embeddings = np.mean(
                        np.array(sentence_embeddings), axis=0
                    )
                    output_tuple_list.append(
                        (clean_sentences, mean_sentence_embeddings)
                    )

        # Save the output in a folder with a similar naming structure to the input
        data_dir = os.path.relpath(data_path, skill_sentences_dir)
        output_file_dir = os.path.join(
            output_dir, data_dir.split(".json")[0] + "_embeddings.json"
        )
        try:
            save_to_s3(s3, bucket_name, output_tuple_list, output_file_dir)
        except:
            try_less_data(s3, bucket_name, output_tuple_list, output_file_dir)
        print(f"Saved output to {output_file_dir}")
