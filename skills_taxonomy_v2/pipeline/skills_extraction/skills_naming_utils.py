"""
Functions to name skills. Used in skills_naming.py.
"""
import logging
from collections import Counter
import re
import itertools
import pandas as pd
import boto3

import numpy as np
import nltk
from nltk.util import ngrams  # function for making ngrams
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.models.phrases import Phrases, Phraser, ENGLISH_CONNECTOR_WORDS
from sklearn.metrics.pairwise import cosine_similarity

from skills_taxonomy_v2.pipeline.sentence_classifier.sentence_classifier import (
    BertVectorizer,
)
from skills_taxonomy_v2.getters.s3_data import (
    save_to_s3
)
from skills_taxonomy_v2 import BUCKET_NAME

from collections import OrderedDict

from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

logger = logging.getLogger(__name__)

# Init the Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()

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

def replace_ngrams(sentence, ngram_words):
    for word_list in ngram_words:
        sentence = sentence.replace(" ".join(word_list), "-".join(word_list))
    return sentence


def clean_cluster_description(sentences):
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

    # How many times a n-gram has to occur in order for occurences of them
    # to be converted to a single dash separated word
    num_times_ngrams_thresh = 3

    # Don't append duplicates, just use set
    cluster_docs_cleaned = set()
    for sentence in sentences:
        # acronyms = re.findall("[A-Z]{2,}", sentence)
        # # Lemmatize - lemmatized_output not used
        # lemmatized_output = [
        #     lemmatizer.lemmatize(w)
        #     if w in acronyms
        #     else lemmatizer.lemmatize(w.lower())
        #     for w in sentence.split(" ")
        # ]
        sentence = sentence.lower()
        singularised_output = [lemmatizer.lemmatize(word) for word in sentence.split(" ") if not word.isdigit()]
        # singularise
        # singularised_output = [singularize(w) for w in sentence.split(" ")]
        no_numbers = [
            word for word in singularised_output if word not in all_stopwords
        ]
        cluster_docs_cleaned.add(" ".join(no_numbers))

    cluster_docs_cleaned = list(cluster_docs_cleaned)

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

    return cluster_docs_clean


def get_clean_ngrams(sents, ngram, min_count, threshold):
    """
    Using the sentences data where each sentence has been clustered into skills,
    find a list of all cleaned n-grams
    """

    # Clean sentences
    cluster_descriptions = clean_cluster_description(sents)

    # tokenise skills
    tokenised_skills = [word_tokenize(skill) for skill in cluster_descriptions]

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


def get_skill_info(skills_df, num_top_sent, ngram, min_count, threshold, s3, BUCKET_NAME, skills_data_output_path):
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

    logger.info(f"Processing skill names ...")
    named_skill_data = {}
    for skills_df_i, skills_data in skills_df.iterrows():
        logger.info(f"Skill {skills_df_i} of {len(skills_df)}")
        try:
            skills_num = skills_data["Skill number"]
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
                sents, ngram, min_count, threshold
            )
            if len(candidate_ngrams) > 1:
                candidate_ngrams_embeds = bert_vectorizer.transform(candidate_ngrams)
                ngram_similarities = cosine_similarity(
                    centroid_embeds.reshape(1, -1), candidate_ngrams_embeds
                )
                closest_ngram = candidate_ngrams[
                    int(ngram_similarities.argsort()[0][::-1].tolist()[0:1][0])
                ]
                named_skill_data[skills_num] = {
                    "Skills name": closest_ngram,
                    "Examples": " ".join(
                        [
                            sents[i]
                            for i in sent_similarities.argsort()[0][::-1].tolist()[
                                0:num_top_sent
                            ]
                        ]
                    ),
                    "Texts": cluster_descriptions[0:100],
                }
            else:
                print(
                    "no candidate ngrams"
                )  # if no candidate ngrams are generated, skill name is smallest skill description
                named_skill_data[skills_num] = {
                    "Skills name": min(cluster_descriptions, key=len),
                    "Examples": " ".join(
                        [
                            sents[i]
                            for i in sent_similarities.argsort()[0][::-1].tolist()[
                                0:num_top_sent
                            ]
                        ]
                    ),
                    "Texts": cluster_descriptions[0:100],
                }
            if int(skills_df_i) % 100 ==0:
                # Re-save updated dict every 100 skills (to be on the safe side in terms of losing data)
                save_to_s3(s3, BUCKET_NAME, named_skill_data, skills_data_output_path)
        except:
            logger.info(f"Problem finding name for skill index {skills_df_i}, skill number {skills_num}")


    # Save final dictionary
    save_to_s3(s3, BUCKET_NAME, named_skill_data, skills_data_output_path)

    return named_skill_data
