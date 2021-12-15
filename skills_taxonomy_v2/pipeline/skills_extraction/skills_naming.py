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

    python -i skills_taxonomy_v2/pipeline/skills_extraction/skills_naming.py --config_path 'skills_taxonomy_v2/config/skills_extraction/2021.11.05.yaml'
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

from skills_taxonomy_v2.getters.s3_data import load_s3_data, save_to_s3, get_s3_data_paths
from skills_taxonomy_v2 import BUCKET_NAME

from skills_taxonomy_v2.pipeline.skills_extraction.skills_naming_utils import (
    clean_cluster_description,
    get_clean_ngrams,
    get_skill_info,
)
from skills_taxonomy_v2.pipeline.skills_extraction.extract_skills_utils import (
    get_output_config_stamped,
)

logger = logging.getLogger(__name__)

def load_process_sentence_data(s3, reduced_embeddings_paths):
    sentences_data = pd.DataFrame()
    for reduced_embeddings_path in tqdm(reduced_embeddings_paths):
        sentences_data_i = load_s3_data(
            s3, BUCKET_NAME,
            reduced_embeddings_path
        )
        sentences_data = pd.concat([sentences_data, pd.DataFrame(sentences_data_i)])
    sentences_data.reset_index(drop=True, inplace=True)
    logger.info(f"{len(sentences_data)} sentences loaded")
    return sentences_data



def parse_arguments(parser):

    parser.add_argument(
        "--config_path",
        help="Path to config file",
        default="skills_taxonomy_v2/config/skills_extraction/2021.11.05.yaml",
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
    skill_sentences = load_s3_data(s3, BUCKET_NAME, params["skill_sentences_path"])

    reduced_embeddings_paths = get_s3_data_paths(
            s3,
            BUCKET_NAME,
            params["skills_embeds_path"],
            file_types=["*sentences_data_*.json"]
            )

    skills_embeds_df = load_process_sentence_data(s3, reduced_embeddings_paths)

    sent_cluster_embeds = load_s3_data(
        s3, BUCKET_NAME, params["mean_skills_embeds_path"]
    )
    skills = load_s3_data(s3, BUCKET_NAME, params["skills_path"])

    # wrangle data in the format needed
   
    skill_sentences_df = pd.DataFrame(skill_sentences)[
        ["job id", "sentence id", "Cluster number predicted"]
    ]

    merged_sents_embeds = pd.merge(
        skills_embeds_df, skill_sentences_df, on=["job id", "sentence id"], how='left'
    ) 

    merged_sents_embeds = merged_sents_embeds[
        merged_sents_embeds["Cluster number predicted"] != -2
    ]

    skills_df = pd.DataFrame(skills).T
    skills_df["Skill number"] = skills_df.index
    skills_df["Mean embedding"] = skills_df["Skill number"].apply(
        lambda x: sent_cluster_embeds[x]
    )

    skills_df["Skill number"] = skills_df["Skill number"].astype("int64")
    skills_df["Sentence embeddings"] = skills_df["Skill number"].apply(
        lambda x: list(
            merged_sents_embeds.loc[
                merged_sents_embeds["Cluster number predicted"] == x
            ]["embedding"]
        )
    )

    logger.info(f"Generating skill names for {len(skills_df)} skills...")

    # Save skill information
    skills_data_output_path = params["skills_path"].split(".json")[0] + "_named.json"
    logger.info(f"Saving skill names to {skills_data_output_path}")

    # generate skills names
    # and save iteratively
    skills_data = get_skill_info(
        skills_df,
        params["num_top_sent"],
        params["ngram"],
        params["min_count"],
        params["threshold"],
        s3,
        BUCKET_NAME,
        skills_data_output_path,
    )
    logger.info(f"{len(skills_data)} skills names saved")
