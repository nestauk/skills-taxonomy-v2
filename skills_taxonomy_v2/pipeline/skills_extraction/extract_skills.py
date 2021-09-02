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
    sample_sentence_embeddings_dirs,
    ExtractSkills
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
    output_dir = params["output_dir"]

    s3 = boto3.resource("s3")

    sentence_embeddings_dirs = get_s3_data_paths(
        s3, BUCKET_NAME, sentence_embeddings_dir, file_types=["*.json"]
    )

    sentence_embeddings_dirs = sample_sentence_embeddings_dirs(
        sentence_embeddings_dirs, params["dir_sample_size"], sample_seed=params["dir_sample_seed"])

    sentences_data = load_sentences_embeddings(
        s3,
        sentence_embeddings_dirs,
        mask_seq=params["mask_seq"],
        prop_not_masked_threshold=params["prop_not_masked_threshold"],
        sample_seed=params["sent_emb_sample_seed"],
        sample_embeddings_size=params["sent_emb_sample_size"], 
        sentence_lengths=[params["sentence_lengths_lower"], params["sentence_lengths_upper"]])

    # It's easier to manipulate this dataset as a dataframe
    sentences_data = pd.DataFrame(sentences_data)

    extract_skills = ExtractSkills(
        umap_n_neighbors=params["umap_n_neighbors"],
        umap_min_dist=params["umap_min_dist"],
        umap_random_state=params["umap_random_state"],
        umap_n_components=params["umap_n_components"],
        dbscan_eps=params["dbscan_eps"],
        dbscan_min_samples=params["dbscan_min_samples"]
        )
    reduced_points_umap = extract_skills.reduce_embeddings(sentences_data["embedding"].tolist())
    clustering_number, cluster_centroids = extract_skills.get_clusters(reduced_points_umap)

    extract_skills_output_path = get_output_config_stamped(
        args.config_path, output_dir, ""
    )
    extract_skills.save_outputs(
        extract_skills_output_path,
        s3)

    # Add to sentences_data dataframe - each sentence will now have the
    # reduced embedding and the assigned cluster label
    sentences_data["reduced_points x"] = list(reduced_points_umap[:, 0])
    sentences_data["reduced_points y"] = list(reduced_points_umap[:, 1])
    sentences_data["reduced_points_umap"] = reduced_points_umap.tolist()
    sentences_data.drop(["embedding"], axis=1, inplace=True)
    sentences_data["Cluster number"] = clustering_number

    # Get names and descriptions of each skill
    skills_data = get_skill_info(sentences_data, params["desc_num_top_sent"], params["name_num_top_words"])

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
