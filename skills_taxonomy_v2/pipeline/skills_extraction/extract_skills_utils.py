"""
Functions to reduce embeddings, apply DSCAN clustering and name clusters using
top TF-IDF words. Used in extract_skills.py.
"""

import logging
from collections import Counter, defaultdict
import os
import joblib
import tempfile
import random

import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import umap.umap_ as umap
import re
import nltk
from nltk.util import ngrams  # function for making ngrams
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics.pairwise import euclidean_distances

from skills_taxonomy_v2.getters.s3_data import load_s3_data, save_to_s3
from skills_taxonomy_v2 import BUCKET_NAME

logger = logging.getLogger(__name__)


def replace_ngrams(sentence, ngram_words):
    for word_list in ngram_words:
        sentence = sentence.replace(" ".join(word_list), "-".join(word_list))
    return sentence


def get_top_tf_idf_words(clusters_vect, feature_names, top_n=2):
    """
    From https://stackoverflow.com/questions/34232190/scikit-learn-tfidfvectorizer-how-to-get-top-n-terms-with-highest-tf-idf-score
    """
    sorted_nzs = np.argsort(clusters_vect.data)[: -(top_n + 1) : -1]
    return feature_names[clusters_vect.indices[sorted_nzs]]


def sample_sentence_embeddings_dirs(sentence_embeddings_dirs, sample_size, sample_seed=42):
    """
    The sentence embeddings come in two parts for each file:
    1. {file_dir}/xx_embeddings.json
    2. {file_dir}/xx_original_sentences.json

    We need both parts of this, so when taking a sample we need to sample the "{file_dir}/xx_" part
    of the directory, and then make sure we have both the embeddings and the original_sentences
    in our output sample.
    """
    base_filedir = set([s.split('_embeddings.json')[0].split('_original_sentences.json')[0] for s in sentence_embeddings_dirs])
    random.seed(sample_seed)
    base_filedir_sample = random.sample(base_filedir, sample_size)
    sentence_embeddings_dirs = [i+'_embeddings.json' for i in base_filedir_sample]
    sentence_embeddings_dirs += [i+'_original_sentences.json' for i in base_filedir_sample]

    return sentence_embeddings_dirs

def load_sentences_embeddings(
    s3,
    sentence_embeddings_dirs,
    mask_seq="[MASK]",
    prop_not_masked_threshold=0.2,
    sample_seed=42,
    sample_embeddings_size=None, 
    sentence_lengths=[0, 10000]
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
            print(f"Loaded {len(sentence_embeddings)} sentences from file {embedding_dir}")
            # Take a sample if needed
            if sample_embeddings_size:
                random.seed(sample_seed)
                sentence_embeddings = random.sample(sentence_embeddings, sample_embeddings_size)
            # Only output data for this sentence if it matches various conditions
            count_keep = 0
            for job_id, sent_id, words, embedding in sentence_embeddings:
                if words not in unique_sentences:
                    words_without_mask = words.replace(mask_seq, "")
                    prop_not_masked = len(words_without_mask) / len(words)
                    if prop_not_masked > prop_not_masked_threshold:
                        original_sentence = original_sentences[str(sent_id)]
                        if len(original_sentence) in range(sentence_lengths[0], sentence_lengths[1]):
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
            logger.info(f"{count_keep} sentences meet conditions out of {len(sentence_embeddings)}")

    logger.info(f"Processed {len(sentences_data)} sentences")

    return sentences_data


def clean_cluster_descriptions(sentences_data):
    """
    For each cluster normalise the texts for getting descriptions from
    - lemmatize
    - lower case
    - remove duplicates
    - n-grams

    Input:
        sentences_data (DataFrame): The sentences in each cluster
            with "description" and "Cluster number" columns

    Output:
        cluster_descriptions (dict): Cluster number : list of cleaned
            sentences in this cluster
    """

    # Init the Wordnet Lemmatizer
    lemmatizer = WordNetLemmatizer()

    # How many times a n-gram has to occur in order for occurences of them
    # to be converted to a single dash separated word
    num_times_ngrams_thresh = 3

    sentences_data["description"] = sentences_data["description"].apply(
        lambda x: re.sub("\s+", " ", x)
    )

    cluster_descriptions = {}
    for cluster_num, cluster_group in sentences_data.groupby("Cluster number"):
        cluster_docs = cluster_group["description"].tolist()
        cluster_docs_cleaned = []
        for doc in cluster_docs:
            # Remove capitals, but not when it's an acronym
            acronyms = re.findall("[A-Z]{2,}", doc)
            # Lemmatize
            lemmatized_output = [
                lemmatizer.lemmatize(w)
                if w in acronyms
                else lemmatizer.lemmatize(w.lower())
                for w in doc.split(" ")
            ]
            cluster_docs_cleaned.append(" ".join(lemmatized_output).strip())
        # Remove duplicates
        cluster_docs_cleaned = list(set(cluster_docs_cleaned))

        # Find the ngrams for this cluster
        all_cluster_docs = " ".join(cluster_docs_cleaned).split(" ")

        esBigrams = ngrams(all_cluster_docs, 3)
        ngram_words = [
            words
            for words, count in Counter(esBigrams).most_common()
            if count >= num_times_ngrams_thresh
        ]

        cluster_docs_clean = [
            replace_ngrams(sentence, ngram_words) for sentence in cluster_docs_cleaned
        ]

        cluster_descriptions[cluster_num] = cluster_docs_clean
    return cluster_descriptions


def get_skill_info(sentences_data, desc_num_top_sent, name_num_top_words):
    """
    For each cluster/skill get a description and a name for it by:
    - Cleaning words (lemmatize...)
    - Name is top 5 tfidf words in the sentences for this cluster
    - Description is the closest original sentence with the embedding
    closest to the cluster centre

    Input:
        sentences_data (DataFrame): The sentences in each cluster
            with "description" and "Cluster number" columns
        desc_num_top_sent (int): How many original sentences closest to
            the cluster centroid to include as the skill description.
        name_num_top_words (int): How many top TF-IDF words to merge
            together as the name for clusters
    Output:
        skills_data (dict): Information about each skill -
            "Skill name": The top TF-IDF words merged together for each skill,
            "Description": The skill description - closest sentences to the
                cluster centre,
            "text": The text for each sentence that created this skill cluster


    """

    # Clean sentences
    cluster_descriptions = clean_cluster_descriptions(sentences_data)

    # Find the closest-to-the-cluster-center sentences in this cluster

    cluster_texts = [
        ". ".join(sentences) for sentences in cluster_descriptions.values()
    ]

    # Have a token pattern that keeps in words with dashes in between
    cluster_vectorizer = TfidfVectorizer(
        stop_words="english", token_pattern=r"(?u)\b\w[\w-]*\w\b|\b\w+\b"
    )
    clusters_vects = cluster_vectorizer.fit_transform(cluster_texts)

    cluster_main_sentence = {}
    for cluster_num, cluster_data in sentences_data.groupby("Cluster number"):
        # There may be the same sentence repeated
        cluster_data.drop_duplicates(["sentence id"], inplace=True)
        cluster_text = cluster_data["original sentence"].tolist()
        cluster_coords = cluster_data[["reduced_points x", "reduced_points y"]].values

        # Get similarities to centre
        similarities = cosine_similarity(
            np.mean(cluster_coords, axis=0).reshape(1, -1), cluster_coords
        )
        # Join the closest sentences to the centroid
        cluster_main_sentence[cluster_num] = ". ".join(
            [
                cluster_text[i]
                for i in similarities.argsort()[0][::-1].tolist()[0:desc_num_top_sent]
            ]
        )

    # Top n words for each cluster + other info
    feature_names = np.array(cluster_vectorizer.get_feature_names())

    # Get top words for each cluster
    skills_data = {}
    for (cluster_num, text), clusters_vect in zip(
        cluster_descriptions.items(), clusters_vects
    ):
        skills_data[cluster_num] = {
            "Skill name": " ".join(
                list(
                    get_top_tf_idf_words(
                        clusters_vect, feature_names, name_num_top_words
                    )
                )
            ),
            "Description": cluster_main_sentence[cluster_num],
            "text": text,
        }

    return skills_data


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
        dbscan_min_samples
        ):
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.umap_random_state = umap_random_state
        self.umap_n_components = umap_n_components
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples

    def reduce_embeddings(
        self,
        embedding_list,
        ):

        # Reduce to 2d
        logger.info(f"Reducing {len(embedding_list)} sentence embeddings to 2D ...")
        self.reducer_class = umap.UMAP(
            n_neighbors=self.umap_n_neighbors,
            min_dist=self.umap_min_dist,
            random_state=self.umap_random_state,
            n_components=self.umap_n_components,
        )
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
            clust_num: (
                np.mean(cluster_points[clust_num], axis=0)
                ).tolist() for clust_num in cluster_names_ordered
            }
        return cluster_centroids


    def get_clusters(
        self, reduced_points_umap
        ):

        # Get clusters using reduced data
        logger.info(f"Finding clusters in reduced data ...")
        clustering = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        clustering_number = clustering.fit_predict(reduced_points_umap).tolist()

        self.cluster_centroids = self.get_cluster_centroids(clustering_number, reduced_points_umap)

        logger.info(f"{len(set(clustering_number))} unique clusters")

        return clustering_number, self.cluster_centroids

    def save_outputs(
        self,
        output_dir,
        s3,
        bucket_name=None
        ):
        """
        Save cluster centroids and reducer class
        """
        clust_cent_filename = output_dir + 'cluster_centroids.json'
        reducer_class_filename = output_dir + 'reducer_class.pkl'

        logger.info(f"Saving cluster centroids and the reducer class to {clust_cent_filename} and {reducer_class_filename}")

        if not bucket_name:
            bucket_name = BUCKET_NAME
        save_to_s3(s3, bucket_name, self.cluster_centroids, clust_cent_filename)

        with tempfile.TemporaryFile() as fp:
            joblib.dump(self.reducer_class, fp)
            fp.seek(0)
            s3.Bucket(bucket_name).put_object(Key=reducer_class_filename , Body=fp.read())

    def load_outputs(
        self,
        input_dir,
        s3,
        bucket_name=None):
        """
        Load cluster centroids and reducer class
        """

        clust_cent_filename = input_dir + 'cluster_centroids.json'
        reducer_class_filename = input_dir + 'reducer_class.pkl'

        logger.info(f"Loading cluster centroids and the reducer class from {clust_cent_filename} and {reducer_class_filename}")

        if not bucket_name:
            bucket_name = BUCKET_NAME
        self.cluster_centroids = load_s3_data(s3, bucket_name, clust_cent_filename)

        with tempfile.TemporaryFile() as fp:
            s3.Bucket(bucket_name).download_fileobj(Fileobj=fp, Key=reducer_class_filename)
            fp.seek(0)
            self.reducer_class = joblib.load(fp)

    def predict(
        self,
        embedding_list
        ):
        """
        Given a new array of embeddings, predict which cluster they will be in.
        Find closest cluster centroid to the reduced point
        """

        logger.info(f"Predicting cluster assignment for {len(embedding_list)} embeddings ...")
        reduced_points_umap = self.reducer_class.transform(embedding_list)
        dists = euclidean_distances(reduced_points_umap, np.array(list(self.cluster_centroids.values())))
        return dists.argmin(axis=1).tolist()

