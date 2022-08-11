"""
Cluster sentences into skills from reduced Job Advert Sentence Embeddings

1. Cluster a sample of the reduced sentence embeddings
2. Find centroids and merge small clusters together
3. Find cluster assignment for rest of sentences

Those clusters of sentences are used as proxies for individual skills.

Usage:
python -i skills_taxonomy_v2/pipeline/skills_extraction/cluster_embeddings.py --config_path 'skills_taxonomy_v2/config/skills_extraction/2021.11.05.yaml'

Experimentation was done in the notebook `Experiment - Cluster parameters.ipynb`.
"""

import yaml
from tqdm import tqdm
import logging
from argparse import ArgumentParser
from collections import Counter

import pandas as pd
import numpy as np
import boto3
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import euclidean_distances

from skills_taxonomy_v2.getters.s3_data import get_s3_data_paths, save_to_s3, load_s3_data
from skills_taxonomy_v2.pipeline.skills_extraction.extract_skills_utils import get_output_config_stamped
from skills_taxonomy_v2 import BUCKET_NAME

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

    sentences_data["reduced_points x"] = sentences_data["embedding"].apply(lambda x: x[0])
    sentences_data["reduced_points y"] = sentences_data["embedding"].apply(lambda x: x[1])
    sentences_data["original sentence length"] = sentences_data["original sentence"].apply(lambda x:len(x))

    return sentences_data

class ClusterEmbeddings():
    def __init__(
        self,
        dbscan_eps=0.01,
        dbscan_min_samples=4,
        max_length=100,
        train_cluster_n=300000,
        train_cluster_rand_seed=42,
        small_cluster_size_threshold=10,
        max_centroid_dist_before_merge=0.05
    ):

        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples  = dbscan_min_samples
        self.max_length = max_length
        self.train_cluster_n = train_cluster_n
        self.train_cluster_rand_seed = train_cluster_rand_seed
        self.small_cluster_size_threshold = small_cluster_size_threshold
        self.max_centroid_dist_before_merge = max_centroid_dist_before_merge

    def get_clusters(self, sentences_data):

        sentences_data_short = sentences_data[sentences_data["original sentence length"] <= self.max_length]
        logger.info(f"{len(sentences_data_short)} sentences <= {self.max_length} characters in length...")
        
        logger.info(f"Finding clusters in sample ...")
        sentences_data_short_sample = sentences_data_short.sample(n=self.train_cluster_n, random_state=self.train_cluster_rand_seed)
        clustering = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        clustering_number = clustering.fit_predict(sentences_data_short_sample["embedding"].tolist()).tolist()
        sentences_data_short_sample['cluster_number'] = clustering_number
        logger.info(f"{len(set(clustering_number))} clusters found in {len(sentences_data_short_sample)} sentences")

        self.sentences_data_short_sample = sentences_data_short_sample

        return clustering_number

    def get_closest_neighbour(self, cluster_centroids, clust_num):
        """
        Find the closest neighbour and the distance from cluster 'clust_num'
        """
        # clust_num's centroid
        clust_num_centroid = cluster_centroids[clust_num]
        # Remove current cluster from closest cluster calculation
        _ = cluster_centroids.pop(clust_num);
        other_cluster_embeddings = list(cluster_centroids.values())
        other_cluster_nums = list(cluster_centroids.keys())
        # Find the nearest neighbour to this cluster
        dists = euclidean_distances(
            [clust_num_centroid],
            other_cluster_embeddings
            )
        closest_clust_ix = dists.argmin(axis=1).tolist()[0]
        closest_clust = other_cluster_nums[closest_clust_ix]
        closest_clust_dist = dists[:, closest_clust_ix][0]

        return closest_clust, closest_clust_dist

    def merge_clusters(self):

        clustering_number = self.sentences_data_short_sample['cluster_number'].tolist()

        logger.info(f"Merging small clusters where suitable ...")
        cluster_centroids = self.sentences_data_short_sample.groupby('cluster_number')["embedding"].apply(
            lambda x: np.mean(x.tolist(), axis=0).tolist()).to_dict()
        if -1 in cluster_centroids:
            _ = cluster_centroids.pop(-1)
        
        all_cluster_size_dict = {k:v for k,v in Counter(clustering_number).items() if k!=-1}
        small_cluster_size_dict = {k:v for k,v in all_cluster_size_dict.items() if v < self.small_cluster_size_threshold}

        # Sort small cluster numbers from smallest to largest
        small_clusters = [k for k, v in sorted(small_cluster_size_dict.items(), key=lambda item: item[1])]

        updated_cluster_centroids = cluster_centroids.copy()
        still_small_clusters = set(small_clusters).copy()

        logger.info(f"Trying to merge {len(small_clusters)} small clusters ...")
        # This algorithm needs to be iterative since nearest neighbour calculation will 
        # be different as clusters get merged
        new_cluster_maps = {}
        for clust_num in tqdm(small_clusters):
            if clust_num in still_small_clusters:
                closest_clust, closest_clust_dist = self.get_closest_neighbour(cluster_centroids.copy(), clust_num)
                if closest_clust_dist < self.max_centroid_dist_before_merge:
                    # The cluster 'clust_num' will now be merged into the cluster 'closest_clust'
                    new_cluster_maps[clust_num] = closest_clust
                    # Delete this cluster from the options of cluster centroids
                    # we don't want to use it in comparisons with future small clusters
                    _ = updated_cluster_centroids.pop(clust_num);
                    # The cluster we merged with may no longer be a small cluster (if it was a small one in the first place),
                    # if so remove from list.
                    if closest_clust in still_small_clusters:
                        if all_cluster_size_dict[clust_num] + all_cluster_size_dict[closest_clust] >= self.small_cluster_size_threshold:
                            still_small_clusters.remove(closest_clust)
                    # If the 'closest_clust' cluster already maps to something else, then directly map
                    # e.g. 
                    # new_cluster_maps[594] = 1072
                    # new_cluster_maps[1072] = 3645
                    # should be updated to "new_cluster_maps[594] = 3645"
                    for old_key_map in [k for k, v in new_cluster_maps.items() if v==clust_num]:
                        new_cluster_maps[old_key_map] = closest_clust

        # Update cluster numbers if they have been flagged to be merged
        self.sentences_data_short_sample["Merged clusters"] = self.sentences_data_short_sample["cluster_number"].apply(
            lambda x: new_cluster_maps[x] if x in new_cluster_maps else x)
        
        logger.info(f"{len(set(clustering_number))} clusters merged into {self.sentences_data_short_sample['Merged clusters'].nunique()}")

    def predict_clusters(self, sentences_data):

        # Re-calculate cluster centroids
        merged_cluster_centroids = self.sentences_data_short_sample.groupby('Merged clusters')["embedding"].apply(
            lambda x: np.mean(x.tolist(), axis=0).tolist()).to_dict()
        if -1 in merged_cluster_centroids:
            _ = merged_cluster_centroids.pop(-1)
        # Careful of order of list when it came from a dict (can get messed up)
        merged_cluster_embeddings = list(merged_cluster_centroids.values())
        merged_cluster_nums = list(merged_cluster_centroids.keys())

        # Predict clusters on rest of sentences

        long_sentences = set(sentences_data[sentences_data["original sentence length"]>self.max_length].index.tolist())

        logger.info(f"{len(long_sentences)} sentences are over {self.max_length} characters - {round(len(long_sentences)*100/len(sentences_data), 2)}%")

        sentences_data_to_predict = sentences_data[sentences_data["original sentence length"]<=self.max_length]
        logger.info(f"Predicting clusters for the {len(sentences_data_to_predict)} sentences under {self.max_length} characters long")

        embeddings_to_predict = sentences_data_to_predict["embedding"].tolist()
        indices_to_predict = sentences_data_to_predict.index.tolist()

        chunk_size = 1000
        predicted_clusters_dict = {}
        for i in tqdm(range(0, len(embeddings_to_predict), chunk_size)):
            chunk_embeddings_to_predict = embeddings_to_predict[i:i+chunk_size]
            chunk_indices_to_predict = indices_to_predict[i:i+chunk_size]
            chunk_nearest_clusters = [merged_cluster_nums[n] for n in euclidean_distances(chunk_embeddings_to_predict, merged_cluster_embeddings).argmin(axis=1).tolist()]
            predicted_clusters_dict.update(dict(zip(chunk_indices_to_predict, chunk_nearest_clusters)))

        def get_pred(x):
            return predicted_clusters_dict.get(x, -2)

        sentences_data["Cluster number predicted"] = sentences_data.index.map(get_pred)

        return sentences_data
    

def parse_arguments(parser):

    parser.add_argument(
        "--config_path",
        help="Path to config file",
        default="skills_taxonomy_v2/config/skills_extraction/2022.01.14.yaml",
    )

    return parser.parse_args()

if __name__ == '__main__':
    parser = ArgumentParser()
    args = parse_arguments(parser)

    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    FLOW_ID = "cluster_embeddings"

    flow_config = config["flows"][FLOW_ID]
    params = flow_config["params"]

    s3 = boto3.resource("s3")

    reduced_embeddings_paths = get_s3_data_paths(
        s3,
        BUCKET_NAME,
        params["reduced_embeddings_dir"],
        file_types=["*sentences_data_*.json"]
        )

    logger.info(f"Loading reduced embeddings from {len(reduced_embeddings_paths)} files ...")
    sentences_data = load_process_sentence_data(s3, reduced_embeddings_paths)

    cluster_embeddings = ClusterEmbeddings(
        params["dbscan_eps"],
        params["dbscan_min_samples"],
        params["max_length"],
        params["train_cluster_n"],
        params["train_cluster_rand_seed"],
        params["small_cluster_size_threshold"],
        params["max_centroid_dist_before_merge"],
    )

    _ = cluster_embeddings.get_clusters(sentences_data)

    cluster_embeddings.merge_clusters()

    new_sentences_data = cluster_embeddings.predict_clusters(sentences_data)

    # How many merged cluster groups are the same as the predicted, for the sample fitted on.
    test_pred_similarity = pd.merge(
        cluster_embeddings.sentences_data_short_sample[['job id', 'sentence id', 'cluster_number', 'Merged clusters']],
        new_sentences_data[['job id', 'sentence id', 'Cluster number predicted']],
        how='left', on=['job id', 'sentence id'])
    test_pred_similarity_clustered = test_pred_similarity[test_pred_similarity['Merged clusters'] >= 0]
    num_pred_same = len(test_pred_similarity_clustered[test_pred_similarity_clustered['Merged clusters'] == test_pred_similarity_clustered['Cluster number predicted']])
    logger.info(f"{num_pred_same} out of {len(test_pred_similarity_clustered)} ({round(num_pred_same*100/len(test_pred_similarity_clustered))}%) of predictions were the same as original clusters for sample clustering was fitted to")

    # Create mapping of 'Cluster number predicted' to new index
    # At the moment this isn't 0:num skills, its 0,1,2,5,10,11,12..
    num_skills = new_sentences_data['Cluster number predicted'].nunique()
    skill_nums = [n for n in new_sentences_data['Cluster number predicted'].unique() if n!=-2]
    reindex_map = dict(zip(skill_nums, range(0, num_skills - 1)))
    reindex_map[-2] = -2
    new_sentences_data['Cluster number predicted'] = new_sentences_data['Cluster number predicted'].apply(lambda x: reindex_map[x])

    # Merge in the original cluster nums (for those in sample)
    clustered_sentences_data = pd.merge(
        new_sentences_data[['job id', 'sentence id', 'Cluster number predicted']],
        cluster_embeddings.sentences_data_short_sample[['job id', 'sentence id', 'cluster_number', 'Merged clusters']],
        how='left', on=['job id', 'sentence id'])

    # Save 
    extract_skills_output_path = get_output_config_stamped(
        args.config_path, params["output_dir"], "sentences_skills_data.json"
    )

    save_to_s3(
        s3,
        BUCKET_NAME,
        clustered_sentences_data.to_dict('records'),
        extract_skills_output_path,
    )

    # Save out just skills data
    # {skill_num: {"sentences": [], "centroid": []}}
    pred_cluster_centroids = new_sentences_data.groupby('Cluster number predicted')["embedding"].apply(
        lambda x: np.mean(x.tolist(), axis=0).tolist()).to_dict()
    _ = pred_cluster_centroids.pop(-2)
    skill_data = {}
    for skill_num, skill_centroid in tqdm(pred_cluster_centroids.items()):
        skill_sents = new_sentences_data[new_sentences_data['Cluster number predicted']==skill_num]['original sentence'].tolist()
        skill_data[skill_num] = {
            "Sentences": skill_sents,
            "Centroid": skill_centroid
        }

    skills_data_output_path = get_output_config_stamped(
            args.config_path, params["output_dir"], "skills_data.json"
        )

    save_to_s3(
        s3,
        BUCKET_NAME,
        skill_data,
        skills_data_output_path,
    )

    # Save light-weight version, the sentences_skills_data file is big, so create a version of lists rather than dictionaries

    extract_skills_output_path = get_output_config_stamped(
        args.config_path, params["output_dir"], "sentences_skills_data_lightweight.json"
    )

    save_to_s3(
        s3,
        BUCKET_NAME,
        clustered_sentences_data[['job id', 'sentence id',  'Cluster number predicted']].values.tolist(),
        extract_skills_output_path,
    )
