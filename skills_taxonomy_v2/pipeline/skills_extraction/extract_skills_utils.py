"""
Functions to reduce embeddings, apply DSCAN clustering and name clusters using
nearest ngram embeddings. Used in extract_skills.py.
"""

import logging
from collections import Counter
import os
import itertools

import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import umap.umap_ as umap
import re
import nltk
from nltk.util import ngrams 
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.models.phrases import Phrases, Phraser, ENGLISH_CONNECTOR_WORDS
from sklearn.metrics.pairwise import cosine_similarity

from skills_taxonomy_v2.getters.s3_data import load_s3_data
from skills_taxonomy_v2 import BUCKET_NAME
from skills_taxonomy_v2.pipeline.sentence_classifier.sentence_classifier import (
    BertVectorizer,
)

logger = logging.getLogger(__name__)


def replace_ngrams(sentence, ngram_words):
    for word_list in ngram_words:
        sentence = sentence.replace(" ".join(word_list), "-".join(word_list))
    return sentence

def load_sentences_embeddings(
    s3, sentence_embeddings_dirs, mask_seq="[MASK]", prop_not_masked_threshold=0.2
):
    """
    Load both the sentences with masking and the original sentences
    """

    # Load all the embeddings and the masked sentences used to create them
    # Don't include exact repeats of the masked sentence
    # Don't include sentences with too high a proportion of masked words

    logger.info(f"Loading and processing sentences from files ...")

    original_sentences = {}
    for embedding_dir in sentence_embeddings_dirs:
        if "original_sentences.json" in embedding_dir:
            original_sentences.update(load_s3_data(s3, BUCKET_NAME, embedding_dir))

    sentences_data = []
    unique_sentences = set()
    counter_disgard = 0
    for embedding_dir in tqdm(sentence_embeddings_dirs):
        if "embeddings.json" in embedding_dir:
            sentence_embeddings = load_s3_data(s3, BUCKET_NAME, embedding_dir)
            for job_id, sent_id, words, embedding in sentence_embeddings:
                if words not in unique_sentences:
                    words_without_mask = words.replace(mask_seq, "")
                    prop_not_masked = len(words_without_mask) / len(words)
                    if prop_not_masked > prop_not_masked_threshold:
                        unique_sentences.add(words)
                        sentences_data.append(
                            {
                                "description": words.replace(mask_seq, ""),
                                "original sentence": original_sentences[str(sent_id)],
                                "job id": job_id,
                                "sentence id": sent_id,
                                "embedding": embedding,
                            }
                        )
                else:
                    counter_disgard += 1

    logger.info(f"Processed {len(sentences_data)} sentences")
    logger.info(f"{counter_disgard} sentences were duplicates so removed")

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


def get_skill_info(sentences_data, ngram, bert_vectorizer, min_count, threshold):
    """
    For each cluster/skill get a description and a name for it by:
    - Cleaning words (lemmatize...)
    - generating ngrams using gensim's Phraser model based on collocation detection 
    - Name is closest embedded ngram to cluster centre

    Input:
        sentences_data (DataFrame): The sentences in each cluster
            with "description" and "Cluster number" columns
        ngram (int): length ngram generated by Phrases
        min_count (int): the number of words and bigrams with total collected count to be ignored
        threshold (int): a score threshold for forming phrases (higher means fewer phrases)
    
    Output:
        skills_data (dict): Information about each skill -
            "Skill name": closest ngram to cluster centre,
            "Description": The skill description - closest sentences to the
                cluster centre,
            "text": The text for each sentence that created this skill cluster

    """

    # Clean sentences
    cluster_descriptions = clean_cluster_descriptions(sentences_data)

    # get cluster texts 
    cluster_texts = [
    " ".join(sentences) for sentences in cluster_descriptions.values()
    ]

    #tokenise skills  
    tokenised_skills = [word_tokenize(skill) for skill in cluster_texts]

    #generate ngrams 
    t = 1
    while t < ngram:
        phrases = Phrases(
                tokenised_skills, min_count=min_count, threshold=threshold, scoring="npmi",
            connector_words = ENGLISH_CONNECTOR_WORDS
            )
        ngram_phraser = Phraser(phrases)
        tokenised_skills = ngram_phraser[tokenised_skills]
        t += 1

    #clean up ngrams 
    clean_ngrams = [[skill.replace('_', ' ').replace('-', ' ') for skill in skills] for skills in  list(tokenised_skills)]
    clean_ngrams = list(set([skill for skill in list(itertools.chain(*clean_ngrams)) if len(skill.split(' ')) > 1]))

    #embed ngrams
    ngram_embeddings = bert_vectorizer.transform(clean_ngrams)

    #calculate similarities 
    ngram_labels = []
    descriptions = []
    for cluster_num, cluster_data in sentences_data.groupby("Cluster number"):
        # There may be the same sentence repeated
        cluster_data.drop_duplicates(["sentence id"], inplace=True)
        cluster_text = cluster_data["original sentence"].tolist()
        cluster_coords = cluster_data[["reduced_points x", "reduced_points y"]].values
        cluster_embeds = [np.array(sent).astype('float32') for sent in cluster_data['embedding'].values.tolist()]


        # Get sent similarities to centre
        sent_similarities = cosine_similarity(
            np.mean(cluster_coords, axis=0).reshape(1, -1), cluster_coords
        )
        # Join the closest sentences to the centroid
        descriptions.append(" ".join(
            [
                cluster_text[i]
                for i in sent_similarities.argsort()[0][::-1].tolist()[0:2]
            ]
        ))

        #calculate similarities between ngrams per cluster and cluster mean 
        ngram_similarities = cosine_similarity(np.mean(cluster_embeds, axis=0).reshape(1, -1), ngram_embeddings)
        #join the closest ngram to the centroid  
        ngram_labels.append(' '.join([clean_ngrams[i] for i in ngram_similarities.argsort()[0][::-1].tolist()[0:1]]))

    skills_data = {}
    for ((cluster_num, text), label, description) in zip(cluster_descriptions.items(), ngram_labels, descriptions):
        skills_data[cluster_num] = {'Skills name': label, 
                                    'description': description,
                                    'text': text}
    
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


def reduce_embeddings(
    embedding_list,
    umap_n_neighbors,
    umap_min_dist,
    umap_random_state,
    umap_n_components=2,
):

    # Reduce to 2d
    logger.info(f"Reducing {len(embedding_list)} sentence embeddings to 2D ...")
    reducer_class = umap.UMAP(
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        random_state=umap_random_state,
        n_components=umap_n_components,
    )
    reduced_points_umap = reducer_class.fit_transform(embedding_list)

    return reduced_points_umap, reducer_class


def get_clusters(reduced_points_umap, dbscan_eps, dbscan_min_samples):

    # Get clusters using reduced data
    logger.info(f"Finding clusters in reduced data ...")
    clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    clustering_number = clustering.fit_predict(reduced_points_umap).tolist()
    logger.info(f"{len(set(clustering_number))} unique clusters")

    return clustering_number, clustering
