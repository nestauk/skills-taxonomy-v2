"""
Using the skills extracted from Text Kernel job adverts we want to find which skills map to ESCO skills.

Usage:
python -i skills_taxonomy_v2/pipeline/skills_extraction/esco_skills_mappings.py --config_path 'skills_taxonomy_v2/config/skills_extraction/2021.08.02.yaml'

"""

import json
from collections import Counter
from argparse import ArgumentParser
import logging
import yaml
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import boto3

from skills_taxonomy_v2.pipeline.sentence_classifier.sentence_classifier import (
    BertVectorizer,
)

from skills_taxonomy_v2.getters.s3_data import load_s3_data, save_to_s3
from skills_taxonomy_v2 import BUCKET_NAME
from skills_taxonomy_v2.pipeline.skills_extraction.extract_skills_utils import (
    get_output_config_stamped,
)

logger = logging.getLogger(__name__)


def load_tk_skills(s3, bucket_name, tk_skills_path):
    # Load TK skills. Don't include the not-skill class '-1'
    tk_skills = load_s3_data(s3, bucket_name, tk_skills_path)
    tk_skills = pd.DataFrame(tk_skills).T
    tk_skills.drop(["-1"], inplace=True)
    tk_skills.reset_index(drop=True, inplace=True)
    return tk_skills


def load_esco_skills(s3, bucket_name, esco_data_dir, csv_path_list):
    # Load ESCO skills, combine a few files into one since their data is split up
    esco_skills = pd.DataFrame()
    for csv_path in csv_path_list:
        skills = load_s3_data(s3, bucket_name, os.path.join(esco_data_dir, csv_path))
        esco_skills = pd.concat([esco_skills, skills])
    esco_skills.reset_index(drop=True, inplace=True)
    return esco_skills


def save_esco_ids(esco_skills):
    # Save out the ESCO id mapper, this is because the ESCO data has it's unique identifier as the
    # preferredlabel text, not a numerical index.
    # {0: 'manage musical staff', 1: ...}
    # This will be handy for a later point.

    esco_skill2ID = {v: k for k, v in esco_skills["preferredLabel"].to_dict().items()}
    esco_ID2skill = esco_skills["preferredLabel"].to_dict()

    save_to_s3(
        s3,
        BUCKET_NAME,
        esco_skill2ID,
        get_output_config_stamped(
            args.config_path, params["output_dir"], "esco_skill2ID.json"
        ),
    )
    save_to_s3(
        s3,
        BUCKET_NAME,
        esco_ID2skill,
        get_output_config_stamped(
            args.config_path, params["output_dir"], "esco_ID2skill.json"
        ),
    )


def find_esco_tk_mappings(
    esco_skills_texts, tk_skills_texts, map_similarity_score_threshold
):

    # Get most similar pairs of tk-esco skills.
    # vectorization + cosine similarity over a threshold

    bert_vectorizer = BertVectorizer(
        bert_model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
        multi_process=True,
    )
    bert_vectorizer.fit()

    embedded_esco = bert_vectorizer.transform(esco_skills_texts)
    embedded_tk = bert_vectorizer.transform(tk_skills_texts)

    similarities = cosine_similarity(embedded_tk, embedded_esco)

    # For each TK skill do you want to find the closest ESCO skill to it (over a threshold) and just map to that one? OR you can find all the ESCO skills over a threshold similar to it and map it to multiple?

    # Only map to one ESCO skill if the similarity is over a threshold
    esco2tk_mapper = {}
    tk2esco_mapper = {}
    for tk_id, esco_id in enumerate(np.argmax(similarities, axis=1)):
        similarity_score = similarities[tk_id, esco_id]
        if similarity_score > map_similarity_score_threshold:
            if esco_id in esco2tk_mapper:
                esco2tk_mapper[esco_id.item()].append(tk_id)
            else:
                esco2tk_mapper[esco_id.item()] = [tk_id]
            if tk_id in tk2esco_mapper:
                tk2esco_mapper[tk_id].append(esco_id.item())
            else:
                tk2esco_mapper[tk_id] = [esco_id.item()]

    return esco2tk_mapper, tk2esco_mapper


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

    FLOW_ID = "esco_mapper"

    flow_config = config["flows"][FLOW_ID]
    params = flow_config["params"]

    s3 = boto3.resource("s3")

    tk_skills = load_tk_skills(s3, BUCKET_NAME, params["tk_skills_path"])
    logger.info(f"{len(tk_skills)} skills loaded from bottom up approach")

    # Combine ESCO skills + language skills + ICT skills + transversal skills
    esco_skills = load_esco_skills(
        s3,
        BUCKET_NAME,
        params["esco_data_dir"],
        [
            params["esco_skills_file"],
            params["esco_lang_skills_file"],
            params["esco_ict_esco_skills_file"],
            params["esco_trans_esco_skills_file"],
        ],
    )
    save_esco_ids(esco_skills)
    logger.info(f"{len(esco_skills)} ESCO skills loaded")

    # Use original descriptions for comparisons
    esco_skills_texts = esco_skills["description"].tolist()
    tk_skills_texts = tk_skills["Description"].tolist()

    esco2tk_mapper, tk2esco_mapper = find_esco_tk_mappings(
        esco_skills_texts, tk_skills_texts, params["map_similarity_score_threshold"]
    )

    logger.info(
        f"{len(tk2esco_mapper)} out of {len(tk_skills_texts)} TK skills were linked with ESCO skills"
    )
    logger.info(
        f"{len([k for k,v in tk2esco_mapper.items() if len(v)>1])} of these were linked to multiple ESCO skills"
    )
    logger.info(
        f"{len(esco2tk_mapper)} out of {len(esco_skills_texts)} ESCO skills were linked with TK skills"
    )
    logger.info(
        f"{len([k for k,v in esco2tk_mapper.items() if len(v)>1])} of these were linked to multiple TK skills"
    )

    save_to_s3(
        s3,
        BUCKET_NAME,
        esco2tk_mapper,
        get_output_config_stamped(
            args.config_path, params["output_dir"], "esco2tk_mapper.json"
        ),
    )
    save_to_s3(
        s3,
        BUCKET_NAME,
        tk2esco_mapper,
        get_output_config_stamped(
            args.config_path, params["output_dir"], "tk2esco_mapper.json"
        ),
    )
