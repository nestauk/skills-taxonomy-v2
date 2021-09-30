"""
Functions to reduce embeddings, and apply DSCAN clustering. Used in extract_skills.py.
"""

import logging
from collections import defaultdict
import os
import joblib
import tempfile
import random

import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import DBSCAN
import umap.umap_ as umap
import re

from skills_taxonomy_v2.getters.s3_data import load_s3_data, save_to_s3
from skills_taxonomy_v2 import BUCKET_NAME

logger = logging.getLogger(__name__)


def sample_sentence_embeddings_dirs(
    sentence_embeddings_dirs, sample_size, sample_seed=42
):
    """
    The sentence embeddings come in two parts for each file:
    1. {file_dir}/xx_embeddings.json
    2. {file_dir}/xx_original_sentences.json

    We need both parts of this, so when taking a sample we need to sample the "{file_dir}/xx_" part
    of the directory, and then make sure we have both the embeddings and the original_sentences
    in our output sample.
    """
    base_filedir = set(
        [
            s.split("_embeddings.json")[0].split("_original_sentences.json")[0]
            for s in sentence_embeddings_dirs
        ]
    )
    random.seed(sample_seed)
    base_filedir_sample = random.sample(base_filedir, sample_size)
    sentence_embeddings_dirs = [i + "_embeddings.json" for i in base_filedir_sample]
    sentence_embeddings_dirs += [
        i + "_original_sentences.json" for i in base_filedir_sample
    ]

    return sentence_embeddings_dirs


def load_sentences_embeddings(
    s3,
    sentence_embeddings_dirs,
    mask_seq="[MASK]",
    prop_not_masked_threshold=0.2,
    sample_seed=42,
    sample_embeddings_size=None,
    sentence_lengths=[0, 10000],
):
    """
    Load the sentence embeddings, the sentences with masking, and the original sentences

    If sample_embeddings_size is given then only output this sample size from each dir in sentence_embeddings_dirs
    i.e. output length will be sample_embeddings_size*len(sentence_embeddings_dirs) minus sentences which :
    - Don't include exact repeats of the masked sentence
    - Don't include sentences with too high a proportion of masked words
    - Don't include sentences which aren't between the sizes given in sentence_lengths
    """

    logger.info(f"Loading and processing sentences from files ...")

    original_sentences = {}
    for embedding_dir in sentence_embeddings_dirs:
        if "original_sentences.json" in embedding_dir:
            original_sentences.update(load_s3_data(s3, BUCKET_NAME, embedding_dir))

    sentences_data = []
    unique_sentences = set()
    for embedding_dir in tqdm(sentence_embeddings_dirs):
        if "embeddings.json" in embedding_dir:
            sentence_embeddings = load_s3_data(s3, BUCKET_NAME, embedding_dir)
            print(
                f"Loaded {len(sentence_embeddings)} sentences from file {embedding_dir}"
            )
            # Take a sample if needed
            if sample_embeddings_size:
                random.seed(sample_seed)
                sentence_embeddings = random.sample(
                    sentence_embeddings, sample_embeddings_size
                )
            # Only output data for this sentence if it matches various conditions
            count_keep = 0
            for job_id, sent_id, words, embedding in sentence_embeddings:
                if words not in unique_sentences:
                    words_without_mask = words.replace(mask_seq, "")
                    prop_not_masked = len(words_without_mask) / len(words)
                    if prop_not_masked > prop_not_masked_threshold:
                        original_sentence = original_sentences[str(sent_id)]
                        if len(original_sentence) in range(
                            sentence_lengths[0], sentence_lengths[1]
                        ):
                            unique_sentences.add(words)
                            sentences_data.append(
                                {
                                    "description": words.replace(mask_seq, ""),
                                    "original sentence": original_sentence,
                                    "job id": job_id,
                                    "sentence id": sent_id,
                                    "embedding": embedding,
                                }
                            )
                            count_keep += 1
            logger.info(
                f"{count_keep} sentences meet conditions out of {len(sentence_embeddings)}"
            )

    logger.info(f"Processed {len(sentences_data)} sentences")

    return sentences_data


def get_output_config_stamped(config_path, output_dir, filename_suffix):
    """
    You may want to have the name of the config file that was used to generate
    the output in the output file name
    e.g.
    config_path = 'skills_taxonomy_v2/config/skills_extraction/2021.08.02.yaml'
    output_dir = 'outputs/skills_extraction/data'
    filename_suffix = 'cluster_data.json'

    outputs:
        'outputs/skills_extraction/data/2021.08.02_cluster_data.json'
    """
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    return os.path.join(output_dir, "_".join([config_name, filename_suffix]))


class ExtractSkills(object):
    """
    """

    def __init__(
        self,
        umap_n_neighbors,
        umap_min_dist,
        umap_random_state,
        umap_n_components,
        dbscan_eps,
        dbscan_min_samples,
    ):
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.umap_random_state = umap_random_state
        self.umap_n_components = umap_n_components
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples

    def reduce_embeddings(
        self, embedding_list,
    ):

        # Reduce to 2d
        logger.info(f"Reducing {len(embedding_list)} sentence embeddings to 2D ...")
        self.reducer_class = umap.UMAP(
            n_neighbors=self.umap_n_neighbors,
            min_dist=self.umap_min_dist,
            random_state=self.umap_random_state,
            n_components=self.umap_n_components,
        )
        if len(embedding_list) >= 100000:
            # If there is a lot of data, then fit the reducer class to a sample of the data
            # since it takes up too much memory for saving otherwise.
            # Then transform all of the points.
            random.seed(42)
            embedding_list_sample = random.sample(embedding_list, 100000)
            self.reducer_class.fit(embedding_list_sample)
            reduced_points_umap = self.reducer_class.transform(embedding_list)
        else:
            reduced_points_umap = self.reducer_class.fit_transform(embedding_list)

        return reduced_points_umap

    def get_cluster_centroids(self, clustering_number, reduced_points_umap):
        """
        Find the average reduced points for each cluster
        """
        cluster_points = defaultdict(list)
        for i, clust_num in enumerate(clustering_number):
            cluster_points[clust_num].append(reduced_points_umap[i])
        # dicts are ordered by insertion order, and this is important for the next step,
        # so keep in numerical cluster number order
        cluster_names_ordered = list(cluster_points.keys())
        if -1 in cluster_points:
            cluster_names_ordered.remove(-1)
        cluster_names_ordered.sort()
        cluster_centroids = {
            clust_num: (np.mean(cluster_points[clust_num], axis=0)).tolist()
            for clust_num in cluster_names_ordered
        }
        return cluster_centroids

    def get_clusters(self, reduced_points_umap):

        # Get clusters using reduced data
        logger.info(f"Finding clusters in reduced data ...")
        clustering = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        clustering_number = clustering.fit_predict(reduced_points_umap).tolist()

        self.cluster_centroids = self.get_cluster_centroids(
            clustering_number, reduced_points_umap
        )

        logger.info(f"{len(set(clustering_number))} unique clusters")

        return clustering_number

    def save_outputs(self, output_dir, s3, bucket_name=None):
        """
        Save cluster centroids and reducer class
        """
        clust_cent_filename = output_dir + "cluster_centroids.json"
        reducer_class_filename = output_dir + "reducer_class.pkl"

        logger.info(
            f"Saving cluster centroids and the reducer class to {clust_cent_filename} and {reducer_class_filename}"
        )

        if not bucket_name:
            bucket_name = BUCKET_NAME
        save_to_s3(s3, bucket_name, self.cluster_centroids, clust_cent_filename)

        try:
            with tempfile.TemporaryFile() as fp:
                joblib.dump(self.reducer_class, fp, compress=("gzip", 5))
                fp.seek(0)
                s3.Bucket(bucket_name).put_object(
                    Key=reducer_class_filename, Body=fp.read()
                )
        except:
            # This is prone to happening due to memory issues
            logger.info(f"Reducer class not saved")

    def load_outputs(self, input_dir, s3, bucket_name=None):
        """
        Load cluster centroids and reducer class
        """

        clust_cent_filename = input_dir + "cluster_centroids.json"
        reducer_class_filename = input_dir + "reducer_class.pkl"

        logger.info(
            f"Loading cluster centroids and the reducer class from {clust_cent_filename} and {reducer_class_filename}"
        )

        if not bucket_name:
            bucket_name = BUCKET_NAME
        self.cluster_centroids = load_s3_data(s3, bucket_name, clust_cent_filename)

        with tempfile.TemporaryFile() as fp:
            s3.Bucket(bucket_name).download_fileobj(
                Fileobj=fp, Key=reducer_class_filename
            )
            fp.seek(0)
            self.reducer_class = joblib.load(fp)

    def predict(self, embedding_list):
        """
        Given a new array of embeddings, predict which cluster they will be in.
        Find closest cluster centroid to the reduced point
        """

        logger.info(
            f"Predicting cluster assignment for {len(embedding_list)} embeddings ..."
        )
        reduced_points_umap = self.reducer_class.transform(embedding_list)
        dists = euclidean_distances(
            reduced_points_umap, np.array(list(self.cluster_centroids.values()))
        )
        return dists.argmin(axis=1).tolist()
