import json
import pickle
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from Levenshtein import distance

def get_many_clusters(skill_nums, average_emb_clust, n_clusters, numclust_its=10):

    # - numclust_its iterations of clustering,
    # - changing random prop_sampled% of data
    # - changing clustering intial conditions
    clustering_results = pd.DataFrame(index=skill_nums)
    for i in range(numclust_its):
        clustering = KMeans(n_clusters=n_clusters, random_state=i)
        cluster_num = clustering.fit_predict(
            [average_emb_clust[k] for k in skill_nums]).tolist()
        new_clustering_results = pd.DataFrame(cluster_num, index=skill_nums, columns=[f'Cluster set {i}'])
        clustering_results = pd.concat([clustering_results, new_clustering_results], axis=1)
        
    return clustering_results

def get_consensus_clusters_mappings(consensus_results_df, k):
    """
    consensus_results_df: a dataframe of each skill and the clusters it was assigned
    to with 10 iterations of clustering
    """
    
    consensus_sets = ["".join([str(cc) for cc in c]) for c in consensus_results_df.values.tolist()]

    consensus_sets_unique = set(consensus_sets)

    # e.g. how similar is '1234' to '1235'?
    all_dists_matrix = []
    for set_1 in consensus_sets_unique:
        temp_list = []
        for set_2 in consensus_sets_unique:
            lev_dict = distance(set_1, set_2)
            temp_list.append(lev_dict)
        all_dists_matrix.append(temp_list)

    # Cluster the consensus sets to group them together
    # e.g. '1234', '1235' and '1233' in group 1
    # '5478' and '5479' in group 2
    
    clustering_dists = KMeans(n_clusters = k, random_state=42)
    cluster_num = clustering_dists.fit_predict(all_dists_matrix).tolist() 
    consensus_set_mapper = dict(zip(list(consensus_sets_unique), cluster_num))
    
    return [consensus_set_mapper[c] for c in consensus_sets]

def get_top_tf_idf_words(vect, feature_names, top_n=2):
    """
    From https://stackoverflow.com/questions/34232190/scikit-learn-tfidfvectorizer-how-to-get-top-n-terms-with-highest-tf-idf-score
    """
    sorted_nzs = np.argsort(vect.data)[: -(top_n + 1) : -1]
    return feature_names[vect.indices[sorted_nzs]].tolist()

def get_level_names(sentence_embs, level_col_name, top_n):
    
    # Merge all the texts within each subsection of this level 
    hier_level_texts = []
    level_nums = []
    for level_num, level_data in sentence_embs.groupby(level_col_name):
        hier_level_texts.append(" ".join(level_data['description'].tolist()))
        level_nums.append(level_num)
        
    vectorizer = TfidfVectorizer()
    vect = vectorizer.fit_transform(hier_level_texts)
    
    feature_names = np.array(vectorizer.get_feature_names())
    
    level_names = {level_num: '-'.join(
        get_top_tf_idf_words(doc_vec, feature_names, top_n=top_n)
    ) for level_num, doc_vec in zip(level_nums, vect)}
    
    return level_names
      
def get_new_level(sentence_embs, previous_level_col, k_means_n, k_means_max_iter, check_low_siloutte=False, silhouette_threshold=0):

    # Mean sentence embedding for the previous level
    average_emb_dict = dict(
        sentence_embs.groupby(previous_level_col)['reduced_points_umap'].apply(lambda x: np.mean(x.tolist(), axis=0).tolist()))

    cluster_mapper = cluster_level_mapper(
        average_emb_dict,
        k_means_n=k_means_n,
        k_means_max_iter=k_means_max_iter,
        check_low_siloutte=check_low_siloutte,
        silhouette_threshold=silhouette_threshold
        )

    return cluster_mapper

def cluster_level_mapper(embeddings_dict, k_means_n, k_means_max_iter=5000, check_low_siloutte=False, silhouette_threshold=0):
    """
    Cluster the embeddings in embeddings_dict values to create a mapper dictionary 
    from the embeddings_dict keys to the cluster number.
    e.g. embeddings_dict = {0: [1.23,5.67], 1: [4.56,7.8],...}
    prev2next_map = {0:5, 1:34, ...}
    """

    clustering = KMeans(n_clusters=k_means_n, max_iter=k_means_max_iter, random_state=42)
    cluster_num = clustering.fit_predict(list(embeddings_dict.values())).tolist()

    if check_low_siloutte:
        # The Silhouette Coefficient is a measure of how well samples are clustered with samples 
        # that are similar to themselves.
        silhouette_samples_n = silhouette_samples(list(embeddings_dict.values()), cluster_num)
        # Give any not well clustered points a new cluster number 
        not_well_clust = list(np.argwhere(silhouette_samples_n < silhouette_threshold).flatten())
        new_cluster_num = k_means_n
        for ix in not_well_clust:
            cluster_num[ix] = new_cluster_num
            new_cluster_num += 1

    cluster_mapper = {k:v for k,v in zip(list(embeddings_dict.keys()), cluster_num)}

    return cluster_mapper

def get_new_level_consensus(sentence_embs, previous_level_col, k_means_n, numclust_its):

    # Mean sentence embedding for the previous level
    average_emb_dict = dict(
        sentence_embs.groupby(previous_level_col)['reduced_points_umap'].apply(lambda x: np.mean(x.tolist(), axis=0).tolist()))

    clustering_results = get_many_clusters(
        list(average_emb_dict.keys()),
        list(average_emb_dict.values()),
        n_clusters=k_means_n,
        numclust_its=numclust_its
        )

    consensus_set_mappings = get_consensus_clusters_mappings(clustering_results, k=k_means_n)
                
    cluster_mapper = dict(zip(
        list(average_emb_dict.keys()),
        consensus_set_mappings
    ))

    return cluster_mapper
    