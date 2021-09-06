"""
After extracting skills by clustering skill sentences, in this script names and
examples are given to each skill.

The skills_data outputed is a dictionary with the following fields for each skill number:
    'Skills name' : The closest single ngram to the centroid of all the 
        sentence embeddings which were clustered to create the skill using cosine similarity.
    'Examples': The original sentences which are closest to the centroid of the skill cluster.
    'Texts': All the cleaned sentences that went into creating the skill cluster.
"""

from argparse import ArgumentParser
import logging
import yaml
import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm
import boto3

from sklearn.metrics.pairwise import cosine_similarity

from skills_taxonomy_v2.getters.s3_data import load_s3_data
from skills_taxonomy_v2 import BUCKET_NAME
from skills_taxonomy_v2.pipeline.sentence_classifier.sentence_classifier import (
    BertVectorizer,
)
from skills_taxonomy_v2.pipeline.skills_extraction.skills_naming_utils import (
    clean_cluster_descriptions, get_clean_ngrams, get_skill_info
)

logger = logging.getLogger(__name__)
        

def parse_arguments(parser):

    parser.add_argument(
        "--config_path",
        help="Path to config file",
        default="skills_taxonomy_v2/config/skills_extraction/2021.08.02.yaml",
    )

    return parser.parse_args()


if __name__ == "__main__":

    parser = ArgumentParser()
    args = parse_arguments(parser)

    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    FLOW_ID = "name_skills"

    flow_config = config["flows"][FLOW_ID]
    params = flow_config["params"]

    s3 = boto3.resource("s3")

    # Load data
    sentence_skills = load_s3_data(s3, BUCKET_NAME, params["sentence_skills_path"])
    sentence_skills = pd.DataFrame(sentence_skills)
    sentence_embs = load_s3_data(s3, BUCKET_NAME, params["embedding_sample_path"])

    # Find n-grams and get skill information
    clean_ngrams = get_clean_ngrams(sentence_skills, params["ngram"], params["min_count"], params["threshold"])
    skills_data = get_skill_info(clean_ngrams, sentence_skills, sentence_embs, cluster_descriptions, params["num_top_sent"])

    # Save skill information
    skills_data_output_path = get_output_config_stamped(
        args.config_path, extracted_skills_larger, "skills_data.json"
    )
    save_to_s3(s3, BUCKET_NAME, skills_data, skills_data_output_path)

