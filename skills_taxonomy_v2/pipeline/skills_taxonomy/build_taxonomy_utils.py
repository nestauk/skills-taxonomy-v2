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
            [average_emb_clust[k] for k in skill_nums]
        ).tolist()
        new_clustering_results = pd.DataFrame(
            cluster_num, index=skill_nums, columns=[f"Cluster set {i}"]
        )
        clustering_results = pd.concat(
            [clustering_results, new_clustering_results], axis=1
        )

    return clustering_results


def get_consensus_clusters_mappings(consensus_results_df, k):
    """
    consensus_results_df: a dataframe of each skill and the clusters it was assigned
    to with 10 iterations of clustering
    """

    consensus_sets = [
        "".join([str(cc) for cc in c]) for c in consensus_results_df.values.tolist()
    ]

    # set(consensus_sets) is stochastic - so need to sort
    consensus_sets_unique = list(set(consensus_sets))
    consensus_sets_unique.sort()

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

    clustering_dists = KMeans(n_clusters=k, random_state=42)
    cluster_num = clustering_dists.fit_predict(all_dists_matrix).tolist()
    consensus_set_mapper = dict(zip(list(consensus_sets_unique), cluster_num))

    return [consensus_set_mapper[c] for c in consensus_sets]


def get_top_tf_idf_words(vect, feature_names, top_n=2):
    """
    From https://stackoverflow.com/questions/34232190/scikit-learn-tfidfvectorizer-how-to-get-top-n-terms-with-highest-tf-idf-score
    """
    sorted_nzs = np.argsort(vect.data)[: -(top_n + 1) : -1]
    return feature_names[vect.indices[sorted_nzs]].tolist()


def get_level_names(sentence_embs, level_col_name, top_n, text_col_name="description", ngram_range=(1,1), max_df=1.0):

    if sentence_embs[level_col_name].nunique() == 1:
        # This parameter only works for >1 document in tf-idf
        max_df = 1.0
    # Merge all the texts within each subsection of this level
    hier_level_texts = []
    level_nums = []
    for level_num, level_data in sentence_embs.groupby(level_col_name):
        hier_level_texts.append(" ".join(level_data[text_col_name].tolist()))
        level_nums.append(level_num)

    vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_df=max_df)
    vect = vectorizer.fit_transform(hier_level_texts)

    feature_names = np.array(vectorizer.get_feature_names())

    level_names = {
        level_num: "-".join(get_top_tf_idf_words(doc_vec, feature_names, top_n=top_n))
        for level_num, doc_vec in zip(level_nums, vect)
    }

    # If there are duplicates then add a number suffix
    duplicated_names = [
        lev_name
        for lev_name, name_count in Counter(level_names.values()).items()
        if name_count > 1
    ]
    if len(duplicated_names) != 0:
        new_level_names = level_names.copy()
        for duplicated_name in duplicated_names:
            i = 0
            for level_num, lev_name in level_names.items():
                if lev_name == duplicated_name:
                    new_level_names[level_num] = lev_name + "-" + str(i)
                    i += 1
        level_names = new_level_names

    return level_names


def get_new_level(
    sentence_embs,
    previous_level_col,
    k_means_n,
    k_means_max_iter,
    check_low_siloutte=False,
    silhouette_threshold=0,
    embedding_column_name="reduced_points_umap",
):

    # Mean sentence embedding for the previous level
    average_emb_dict = dict(
        sentence_embs.groupby(previous_level_col)[embedding_column_name].apply(
            lambda x: np.mean(x.tolist(), axis=0).tolist()
        )
    )

    cluster_mapper = cluster_level_mapper(
        average_emb_dict,
        k_means_n=k_means_n,
        k_means_max_iter=k_means_max_iter,
        check_low_siloutte=check_low_siloutte,
        silhouette_threshold=silhouette_threshold,
    )

    return cluster_mapper


def cluster_level_mapper(
    embeddings_dict,
    k_means_n,
    k_means_max_iter=5000,
    check_low_siloutte=False,
    silhouette_threshold=0,
):
    """
    Cluster the embeddings in embeddings_dict values to create a mapper dictionary
    from the embeddings_dict keys to the cluster number.
    e.g. embeddings_dict = {0: [1.23,5.67], 1: [4.56,7.8],...}
    prev2next_map = {0:5, 1:34, ...}
    """

    clustering = KMeans(
        n_clusters=k_means_n, max_iter=k_means_max_iter, random_state=42
    )
    cluster_num = clustering.fit_predict(list(embeddings_dict.values())).tolist()

    if check_low_siloutte:
        # The Silhouette Coefficient is a measure of how well samples are clustered with samples
        # that are similar to themselves.
        silhouette_samples_n = silhouette_samples(
            list(embeddings_dict.values()), cluster_num
        )
        # Give any not well clustered points a new cluster number
        not_well_clust = list(
            np.argwhere(silhouette_samples_n < silhouette_threshold).flatten()
        )
        new_cluster_num = k_means_n
        for ix in not_well_clust:
            cluster_num[ix] = new_cluster_num
            new_cluster_num += 1

    cluster_mapper = {k: v for k, v in zip(list(embeddings_dict.keys()), cluster_num)}

    return cluster_mapper


def get_new_level_consensus(
    sentence_embs,
    previous_level_col,
    k_means_n,
    numclust_its,
    embedding_column_name="reduced_points_umap",
):

    # Mean sentence embedding for the previous level
    average_emb_dict = dict(
        sentence_embs.groupby(previous_level_col)[embedding_column_name].apply(
            lambda x: np.mean(x.tolist(), axis=0).tolist()
        )
    )

    clustering_results = get_many_clusters(
        list(average_emb_dict.keys()),
        list(average_emb_dict.values()),
        n_clusters=k_means_n,
        numclust_its=numclust_its,
    )

    consensus_set_mappings = get_consensus_clusters_mappings(
        clustering_results, k=k_means_n
    )

    cluster_mapper = dict(zip(list(average_emb_dict.keys()), consensus_set_mappings))

    return cluster_mapper


def amend_level_b_mapper(
    level_b_cluster_mapper,
    levela_manual,
    misc_name="Misc",
    level_c_list_name="Level c list",
):
    """
    Some level B groups were too mixed and will be split up in order
    to fit them into manual level A groups. If their level C group
    would still be in the miscellaneous level A group, then no need
    to split up anyway.
    """

    level_b_cluster_mapper_manual = level_b_cluster_mapper.copy()

    # Give them brand new level B groups in the mapper:
    new_level_b_num = max(set(level_b_cluster_mapper_manual.values())) + 1
    for v in levela_manual.values():
        level_c_list = v.get("Level c list")
        if level_c_list and v["Name"] != "Misc":
            # If their original level B is the same, then group
            grouped_levbc = defaultdict(list)
            for lev_c in level_c_list:
                grouped_levbc[level_b_cluster_mapper[lev_c]].append(lev_c)
            for grouped_level_c_list in grouped_levbc.values():
                for lev_c in grouped_level_c_list:
                    level_b_cluster_mapper_manual[lev_c] = new_level_b_num
                new_level_b_num += 1

    return level_b_cluster_mapper_manual


def manual_cluster_level(levela_manual, level_b_cluster_mapper):

    # Some of the level B indices are no longer in use
    level_bs = set(level_b_cluster_mapper.values())
    not_used_level_bs = set(range(0, max(level_bs) + 1)).difference(level_bs)

    level_a_cluster_mapper = {}
    for level_a_num, level_a_manual_edits in levela_manual.items():
        for level_b_num in level_a_manual_edits["Level b list"]:
            if level_b_num not in not_used_level_bs:
                level_a_cluster_mapper[level_b_num] = int(level_a_num)
        if level_a_manual_edits.get("Level c list"):
            for lev_c in level_a_manual_edits["Level c list"]:
                level_b_num = level_b_cluster_mapper[lev_c]
                level_a_cluster_mapper[level_b_num] = int(level_a_num)

    return level_a_cluster_mapper
