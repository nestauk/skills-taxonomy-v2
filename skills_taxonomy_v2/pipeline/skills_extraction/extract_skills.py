"""
Extract Skills from Job Advert Sentence Embeddings

In this script the embeddings for skills sentences are reduced into 2D space and clustered.
Those clusters of sentences are used as proxies for individual skills, a name and description
for each skill is found by:
- name: using the 5 most frequent and unique words for each cluster (using tf-idf vectors),
- description: the most similar original (unmasked) sentence(s) to the cluster centre.

Usage:
python -i skills_taxonomy_v2/pipeline/skills_extraction/extract_skills.py --config_path 'skills_taxonomy_v2/config/skills_extraction/2021.08.02.yaml'

"""

from argparse import ArgumentParser
import logging
import yaml

import pandas as pd
from tqdm import tqdm
import boto3

from skills_taxonomy_v2.getters.s3_data import get_s3_data_paths, save_to_s3
from skills_taxonomy_v2.pipeline.skills_extraction.extract_skills_utils import (
    replace_ngrams,
    get_top_tf_idf_words,
    load_sentences_embeddings,
    clean_cluster_descriptions,
    get_skill_info,
    get_output_config_stamped,
    reduce_embeddings,
    get_clusters,
)
from skills_taxonomy_v2 import BUCKET_NAME

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

    FLOW_ID = "extract_skills"

    flow_config = config["flows"][FLOW_ID]
    params = flow_config["params"]
    sentence_embeddings_dir = params["sentence_embeddings_dir"]
    prop_not_masked_threshold = params["prop_not_masked_threshold"]
    mask_seq = params["mask_seq"]
    umap_n_neighbors = params["umap_n_neighbors"]
    umap_min_dist = params["umap_min_dist"]
    umap_n_components = params["umap_n_components"]
    umap_random_state = params["umap_random_state"]
    dbscan_eps = params["dbscan_eps"]
    dbscan_min_samples = params["dbscan_min_samples"]
    desc_num_top_sent = params["desc_num_top_sent"]
    name_num_top_words = params["name_num_top_words"]
    output_dir = params["output_dir"]

    s3 = boto3.resource("s3")

    sentence_embeddings_dirs = get_s3_data_paths(
        s3, BUCKET_NAME, sentence_embeddings_dir, file_types=["*.json"]
    )

    sentences_data = load_sentences_embeddings(s3, sentence_embeddings_dirs)

    # It's easier to manipulate this dataset as a dataframe
    sentences_data = pd.DataFrame(sentences_data)

    # Reduce to 2d
    reduced_points_umap = reduce_embeddings(
        sentences_data["embedding"].tolist(),
        umap_n_neighbors,
        umap_min_dist,
        umap_random_state,
        umap_n_components=umap_n_components,
    )
    sentences_data["reduced_points x"] = list(reduced_points_umap[:, 0])
    sentences_data["reduced_points y"] = list(reduced_points_umap[:, 1])
    sentences_data.drop(["embedding"], axis=1, inplace=True)

    # Get clusters
    sentences_data["Cluster number"] = get_clusters(
        reduced_points_umap, dbscan_eps, dbscan_min_samples
    )

    # Get names and descriptions of each skill
    skills_data = get_skill_info(sentences_data, desc_num_top_sent, name_num_top_words)

    # Save
    # The skills data
    skills_data_output_path = get_output_config_stamped(
        args.config_path, output_dir, "skills_data.json"
    )
    save_to_s3(s3, BUCKET_NAME, skills_data, skills_data_output_path)
    # The sentences data inc the embedding reduction and which cluster/skill the sentence was in
    # You may not need to save this out, but will do for now:
    sentences_data_output_path = get_output_config_stamped(
        args.config_path, output_dir, "sentences_data.json"
    )
    save_to_s3(
        s3,
        BUCKET_NAME,
        sentences_data.to_dict(orient="list"),
        sentences_data_output_path,
    )
