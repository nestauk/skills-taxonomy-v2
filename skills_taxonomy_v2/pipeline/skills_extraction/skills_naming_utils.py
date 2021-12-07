"""
Functions to name skills. Used in skills_naming.py.
"""
import logging
from collections import Counter
import re
import itertools
import spacy
import pandas as pd
from tqdm import tqdm
import boto3

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.util import ngrams  # function for making ngrams
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.models.phrases import Phrases, Phraser, ENGLISH_CONNECTOR_WORDS
from sklearn.metrics.pairwise import cosine_similarity

from skills_taxonomy_v2.pipeline.sentence_classifier.sentence_classifier import (
    BertVectorizer,
)
from skills_taxonomy_v2.getters.s3_data import load_s3_data, save_to_s3
from skills_taxonomy_v2 import BUCKET_NAME

from pattern.text.en import singularize
from collections import OrderedDict

from nltk.corpus import stopwords

nltk.download("stopwords")

logger = logging.getLogger(__name__)

nltk.download("wordnet")


def get_new_skills_embeds(new_skills_embeds_path, bucket_name):
    """
    takes as input string path to new skills embeddings directory and bucket name.
    outputs:
        dictionary of new skills embeddings.
    """

    s3 = boto3.resource("s3")

    sentence_embeds = {}
    all_sentence_embeds = []

    for i in tqdm(range(0, 8)):  # 8 files

        new_sentences_dict = load_s3_data(
            s3, bucket_name, new_skills_embeds_path + f"sentences_data_{i}.json"
        )
        all_sentence_embeds.append(new_sentences_dict)

    # https://stackoverflow.com/questions/57340332/how-do-you-combine-lists-of-multiple-dictionaries-in-python
    for k in all_sentence_embeds[0].keys():
        sentence_embeds[k] = sum(
            [skills_dict[k] for skills_dict in all_sentence_embeds], []
        )

    return sentence_embeds


def replace_ngrams(sentence, ngram_words):
    for word_list in ngram_words:
        sentence = sentence.replace(" ".join(word_list), "-".join(word_list))
    return sentence


def clean_cluster_description(sentences, cluster_number):
    """
    For each cluster normalise the texts for getting descriptions from
    - lemmatize
    - lower case
    - remove work-related stopwords
    - remove duplicates
    - singularise descriptions
    - n-grams

    Input:
        sentences (list): The sentences in each cluster
        cluster_number(str(int)): cluster number the sentences belong to. 

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
        "required" "degree",
        "responsibility",
        "duties",
        "responsibilities",
        "experienced",
        "previous",
        "andor",
        "minimum",
        "year",
        "skill",
        "ideal",
        "candidate",
        "desirable",
        "willing",
        "prepared",
        "knowledge",
        "experience",
        "skills",
        "ideally",
        "responsible",
        "require",
        "environment",
        "role",
        "work",
        "job",
        "description",
        "ymy",
    ]

    all_stopwords = stopwords.words("english") + work_stopwords

    cluster_descriptions = {}
    cluster_docs_cleaned = []

    for sentence in sentences:
        acronyms = re.findall("[A-Z]{2,}", sentence)
        # Lemmatize
        lemmatized_output = [
            lemmatizer.lemmatize(w)
            if w in acronyms
            else lemmatizer.lemmatize(w.lower())
            for w in sentence.split(" ")
        ]
        # singularise
        singularised_output = [singularize(w) for w in sentence.split(" ")]
        no_stopwords = [
            word for word in singularised_output if word not in all_stopwords
        ]
        no_numbers = [word for word in no_stopwords if not word.isdigit()]
        cluster_docs_cleaned.append(" ".join(no_numbers))

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

        cluster_descriptions[cluster_number] = cluster_docs_clean

    return cluster_descriptions


def get_clean_ngrams(sents, cluster_number, ngram, min_count, threshold):
    """
    Using the sentences data where each sentence has been clustered into skills,
    find a list of all cleaned n-grams
    """

    # Init the Wordnet Lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Clean sentences
    cluster_descriptions = clean_cluster_description(sents, cluster_number)

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
    return (
        [clean for clean in clean_ngrams if len(clean.split(" ")) > 1],
        cluster_descriptions,
    )


def get_skill_info(skills_df, num_top_sent, ngram, min_count, threshold):
    """
    Inputs:
        'skills_df' (dataframe)
    
    Output: skills_data (dict), for each skill number:
        'Skills name' : the closest ngram to the centroid or, if no ngrams generated, shortest description
        'Examples': Join the num_top_sent closest original sentences to the centroid
        'Texts': All the cleaned sentences for this cluster
    """

    bert_vectorizer = BertVectorizer(
        bert_model_name="sentence-transformers/all-MiniLM-L6-v2", multi_process=True,
    )
    bert_vectorizer.fit()

    skill_data = {}
    for skills_num, skills_data in skills_df.iterrows():
        sents = skills_data["Sentences"]
        reduced_centroid_embeds = np.array(skills_data["Centroid"]).astype("float32")
        centroid_embeds = np.array(skills_data["Mean embedding"]).astype("float32")
        sent_embeds = [
            np.array(sent_embed).astype("float32")
            for sent_embed in skills_data["Sentence embeddings"]
        ]

        sent_similarities = cosine_similarity(
            reduced_centroid_embeds.reshape(1, -1), sent_embeds
        )

        candidate_ngrams, cluster_descriptions = get_clean_ngrams(
            sents, skills_num, ngram, min_count, threshold
        )
        if len(candidate_ngrams) > 1:
            candidate_ngrams_embeds = bert_vectorizer.transform(candidate_ngrams)
            ngram_similarities = cosine_similarity(
                centroid_embeds.reshape(1, -1), candidate_ngrams_embeds
            )
            closest_ngram = candidate_ngrams[
                int(ngram_similarities.argsort()[0][::-1].tolist()[0:1][0])
            ]
            skill_data[skills_num] = {
                "Skills name": closest_ngram,
                "Examples": " ".join(
                    [
                        sents[i]
                        for i in sent_similarities.argsort()[0][::-1].tolist()[
                            0:num_top_sent
                        ]
                    ]
                ),
                "Texts": cluster_descriptions[skills_num],
            }
        else:
            print(
                "no candidate ngrams"
            )  # if no candidate ngrams are generated, skill name is smallest skill description
            skill_data[skills_num] = {
                "Skills name": min(cluster_descriptions[skills_num], key=len),
                "Examples": " ".join(
                    [
                        sents[i]
                        for i in sent_similarities.argsort()[0][::-1].tolist()[
                            0:num_top_sent
                        ]
                    ]
                ),
                "Texts": cluster_descriptions[skills_num],
            }

    return skill_data
