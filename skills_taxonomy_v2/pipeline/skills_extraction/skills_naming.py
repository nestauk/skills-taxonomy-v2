"""
After extracting skills by clustering skill sentences, in this script names and
examples are given to each skill.

The skills_data outputed is a dictionary with the following fields for each skill number:
    'Skills name' : The closest ngram to the centroid of all the
        sentence embeddings which were clustered to create the skill using cosine similarity,
        or the shortest skill cluster description.
    'Examples': The original sentences which are closest to the centroid of the skill cluster.
    'Texts': All the cleaned sentences that went into creating the skill cluster.

Usage:

    python -i skills_taxonomy_v2/pipeline/skills_extraction/skills_naming.py --config_path 'skills_taxonomy_v2/config/skills_extraction/2021.12.07.yaml'
"""

from argparse import ArgumentParser
import logging
import yaml
import itertools
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
import boto3

from sklearn.metrics.pairwise import cosine_similarity

from skills_taxonomy_v2.getters.s3_data import load_s3_data, save_to_s3
from skills_taxonomy_v2 import BUCKET_NAME

from skills_taxonomy_v2.pipeline.skills_extraction.skills_naming_utils import (
    get_new_skills_embeds,
    clean_cluster_description,
    get_clean_ngrams,
    get_skill_info,
)
from skills_taxonomy_v2.pipeline.skills_extraction.extract_skills_utils import (
    get_output_config_stamped,
)

logger = logging.getLogger(__name__)


def parse_arguments(parser):

    parser.add_argument(
        "--config_path",
        help="Path to config file",
        default="skills_taxonomy_v2/config/skills_extraction/2021.12.07.yaml",
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
    skill_sentences = load_s3_data(s3, BUCKET_NAME, params["new_skill_sentences_path"])
    skills_embeds = get_new_skills_embeds(params["new_skills_embeds_path"], BUCKET_NAME)
    sent_cluster_embeds = load_s3_data(
        s3, BUCKET_NAME, params["mean_skills_embeds_path"]
    )
    skills = load_s3_data(s3, BUCKET_NAME, params["new_skills_path"])

    # wrangle data in the format needed
    skills_embeds_df = pd.DataFrame(skills_embeds)[
        ["original sentence", "sentence id", "embedding"]
    ]
    skill_sentences_df = pd.DataFrame(skill_sentences)[
        ["sentence id", "Cluster number predicted"]
    ]
    merged_sents_embeds = pd.merge(
        skills_embeds_df, skill_sentences_df, on="sentence id"
    )
    merged_sents_embeds = merged_sents_embeds[
        merged_sents_embeds["Cluster number predicted"] != -2
    ]

    skills_df = pd.DataFrame(skills).T
    skills_df["Mean embedding"] = sent_cluster_embeds.values()
    skills_df["Sentence embeddings"] = list(
        merged_sents_embeds.groupby("Cluster number predicted")["embedding"].apply(list)
    )

    # generate skills names
    skills_data = get_skill_info(
        skills_df,
        params["num_top_sent"],
        params["ngram"],
        params["min_count"],
        params["threshold"],
    )

    # Save skill information
    skills_data_output_path = get_output_config_stamped(
        args.config_path, params["output_dir"], "skills_data.json"
    )

    save_to_s3(s3, BUCKET_NAME, skills_data, skills_data_output_path)
