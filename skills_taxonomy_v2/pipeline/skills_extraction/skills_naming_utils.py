"""
Functions to name skills. Used in skills_naming.py.
"""

import logging
from collections import Counter
import re
import itertools
import spacy
from argparse import ArgumentParser
import yaml
import pandas as pd
import boto3

import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.models.phrases import Phrases, Phraser, ENGLISH_CONNECTOR_WORDS
from sklearn.metrics.pairwise import cosine_similarity

from skills_taxonomy_v2.pipeline.sentence_classifier.sentence_classifier import (
    BertVectorizer,
)

from pattern.text.en import singularize
from collections import OrderedDict
import random
from collections import defaultdict

logger = logging.getLogger(__name__)

nltk.download("wordnet")


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


def clean_cluster_descriptions(sentences_data):
    """
    For each cluster normalise the texts for getting descriptions from
    - lemmatize
    - lower case
    - remove work-related stopwords
    - remove duplicates
    - singularise descriptions
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
    work_stopwords = [
        "essential",
        "requirement",
        "degree",
        "responsibility",
        "duties",
        "responsibilities",
        "experienced",
        "previous",
        "andor",
        "minimum",
        "years",
    ]

    sentences_data["description"] = sentences_data["description"].apply(
        lambda x: re.sub("\s+", " ", x)
    )

    cluster_descriptions = {}
    for cluster_num, cluster_group in sentences_data.groupby("Cluster number"):
        cluster_docs = cluster_group["description"].tolist()
        cluster_docs_cleaned = []
        for doc in cluster_docs:
            # Remove capitals, but not when it's an acronym
            no_work_stopwords = [w for w in doc.split(" ") if w not in work_stopwords]

            acronyms = re.findall("[A-Z]{2,}", doc)
            # Lemmatize
            lemmatized_output = [
                lemmatizer.lemmatize(w)
                if w in acronyms
                else lemmatizer.lemmatize(w.lower())
                for w in doc.split(" ")
            ]
            # singularise
            singularised_output = [singularize(w) for w in doc.split(" ")]

            # remove work stopwords
            cluster_docs_cleaned.append(" ".join(no_work_stopwords).strip())

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


def get_clean_ngrams(sentence_skills, ngram, min_count, threshold):
    """
    Using the sentences data where each sentence has been clustered into skills,
    find a list of all cleaned n-grams
    """

    # Init the Wordnet Lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Clean sentences
    cluster_descriptions = clean_cluster_descriptions(sentence_skills)

    # get cluster texts
    cluster_texts = [" ".join(sentences) for sentences in cluster_descriptions.values()]

    # tokenise skills
    tokenised_skills = [word_tokenize(skill) for skill in cluster_texts]

    # generate ngrams
    t = 1
    while t < ngram:
        phrases = Phrases(
            tokenised_skills,
            min_count=min_count,
            threshold=threshold,
            scoring="npmi",
            connector_words=ENGLISH_CONNECTOR_WORDS,
        )
        ngram_phraser = Phraser(phrases)
        tokenised_skills = ngram_phraser[tokenised_skills]
        t += 1

    # clean up ngrams
    clean_ngrams = [
        [skill.replace("_", " ").replace("-", " ") for skill in skills]
        for skills in list(tokenised_skills)
    ]
    clean_ngrams = list(
        set(
            [
                skill
                for skill in list(itertools.chain(*clean_ngrams))
                if len(skill.split(" ")) > 1
            ]
        )
    )

    # get rid of duplicate terms in ngrams
    clean_ngrams = [
        " ".join(OrderedDict((w, w) for w in ngrm.split()).keys())
        for ngrm in clean_ngrams
    ]

    # lemmatise ngrams
    clean_ngrams = [
        " ".join([lemmatizer.lemmatize(n) for n in ngram.split(" ")])
        for ngram in clean_ngrams
    ]

    # only return ngrams that are more than 1 word long
    return [
        clean for clean in clean_ngrams if len(clean.split(" ")) > 1
    ], cluster_descriptions


def get_skill_info(
    sentence_skills, sentence_embs, num_top_sent=2, ngram=3, min_count=1, threshold=0.15
):
    """
    Output: skills_data (dict), for each skill number:
        'Skills name' : the closest ngram to the centroid or, if no ngrams generated, shortest description
        'Name method': How the ngram was generated (i.e. chunking, Spacy Phrases)
        'Skills name embed': embedding of closest ngram to the centroid or shortest description embedding
        'Examples': Join the num_top_sent closest original sentences to the centroid
        'Texts': All the cleaned sentences for this cluster
    """
    bert_vectorizer = BertVectorizer(
        bert_model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
        multi_process=True,
    )
    bert_vectorizer.fit()

    skills_data = {}
    for cluster_num, cluster_data in tqdm(sentence_skills.groupby("Cluster number")):
        # There may be the same sentence repeated
        cluster_data.drop_duplicates(["sentence id"], inplace=True)
        cluster_text = cluster_data["original sentence"].tolist()
        cluster_coords = cluster_data[["reduced_points x", "reduced_points y"]].values
        cluster_embeds = [
            np.array(sentence_embs[str(sent_id)]).astype("float32")
            for sent_id in cluster_data["sentence id"].values.tolist()
            if str(sent_id) in sentence_embs
        ]

        # Get sent similarities to centre
        sent_similarities = cosine_similarity(
            np.mean(cluster_coords, axis=0).reshape(1, -1), cluster_coords
        )

        candidate_ngrams, cluster_descriptions = get_clean_ngrams(
            cluster_data, ngram, min_count, threshold
        )

        if (
            len(candidate_ngrams) > 1
        ):  # if there are more than 1 candidate ngrams, skill cluster is labelled as the closest ngram embedding to the cluster mean embedding
            candidate_ngrams_embeds = bert_vectorizer.transform(candidate_ngrams)
            # calculate similarities between ngrams per cluster and cluster mean
            ngram_similarities = cosine_similarity(
                np.mean(cluster_embeds, axis=0).reshape(1, -1), candidate_ngrams_embeds
            )
            closest_ngram = candidate_ngrams[
                int(ngram_similarities.argsort()[0][::-1].tolist()[0:1][0])
            ]

            skills_data[cluster_num] = {
                "Skills name": closest_ngram,
                "Name method": "phrases_embedding",
                "Examples": " ".join(
                    [
                        cluster_text[i]
                        for i in sent_similarities.argsort()[0][::-1].tolist()[
                            0:num_top_sent
                        ]
                    ]
                ),
                "Texts": cluster_descriptions[cluster_num],
            }

        else:  # if no candidate ngrams are generated, skill name is smallest skill description
            skills_data[cluster_num] = {
                "Skills name": min(cluster_descriptions[cluster_num], key=len),
                "Name method": "minimum description",
                "Examples": " ".join(
                    [
                        cluster_text[i]
                        for i in sent_similarities.argsort()[0][::-1].tolist()[
                            0:num_top_sent
                        ]
                    ]
                ),
                "Texts": cluster_descriptions[cluster_num],
            }

    return skills_data


def rename_duplicate_named_skills(named_skills):
    """
    If the skill name is a duplicate, add a count i.e. 'project management 1' to skill name.

    Output: skills_data (dict), for each skill number:
        'Skills name' : closest ngram to the centroid per skill cluster. Count is added to name, if duplicate.
        'Name method': how the ngram was generated (i.e. verb chunking, spacy Phrases)
        'Examples': Join the num_top_sent closest original sentences to the centroid
        'Texts': All the cleaned sentences for this cluster
    """

    skill_name_counts = Counter(
        [skill_data["Skills name"] for skill, skill_data in named_skills.items()]
    )

    for skill_name, skill_name_count in skill_name_counts.items():
        dup_name_counter = 0
        if skill_name_count > 1:
            for skill, skill_data in named_skills.items():
                if skill_data["Skills name"] == skill_name:
                    dup_name_counter += 1
                    skill_data["Skills name"] = skill_name + " " + str(dup_name_counter)

    return named_skills
