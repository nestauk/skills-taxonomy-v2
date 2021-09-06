"""
Functions to name skills. Used in skills_naming.py.
"""

import logging
from collections import Counter
import re
import itertools

import numpy as np
from tqdm import tqdm
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

logger = logging.getLogger(__name__)

nltk.download('wordnet')

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


def get_clean_ngrams(sentence_skills, ngram, min_count, threshold):
    """
    Using the sentences data where each sentence has been clustered into skills,
    find a list of all cleaned n-grams 
    """

    # Clean sentences
    cluster_descriptions = clean_cluster_descriptions(sentence_skills)

    # get cluster texts 
    cluster_texts = [
    " ".join(sentences) for sentences in cluster_descriptions.values()
    ]

    # tokenise skills  
    tokenised_skills = [word_tokenize(skill) for skill in cluster_texts]

    # generate ngrams 
    t = 1
    while t < ngram:
        phrases = Phrases(
                tokenised_skills, min_count=min_count, threshold=threshold, scoring="npmi",
            connector_words = ENGLISH_CONNECTOR_WORDS
            )
        ngram_phraser = Phraser(phrases)
        tokenised_skills = ngram_phraser[tokenised_skills]
        t += 1

    # clean up ngrams 
    clean_ngrams = [[skill.replace('_', ' ').replace('-', ' ') for skill in skills] for skills in  list(tokenised_skills)]
    clean_ngrams = list(set([skill for skill in list(itertools.chain(*clean_ngrams)) if len(skill.split(' ')) > 1]))

    return clean_ngrams, cluster_descriptions

def get_skill_info(clean_ngrams, sentence_skills, sentence_embs, cluster_descriptions, num_top_sent=2):
    """
    Output: skills_data (dict), for each skill number:
        'Skills name' : join the closest ngram to the centroid
        'Examples': Join the num_top_sent closest original sentences to the centroid
        'Texts': All the cleaned sentences for this cluster
    """

    bert_vectorizer = BertVectorizer(
            bert_model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
            multi_process=True,
    )
    bert_vectorizer.fit()

    # embed ngrams
    ngram_embeddings = bert_vectorizer.transform(clean_ngrams)

    #calculate similarities 
    skills_data = {}
    for cluster_num, cluster_data in tqdm(sentence_skills.groupby("Cluster number")):
        # There may be the same sentence repeated
        cluster_data.drop_duplicates(["sentence id"], inplace=True)
        cluster_text = cluster_data["original sentence"].tolist()
        cluster_coords = cluster_data[["reduced_points x", "reduced_points y"]].values
        cluster_embeds = [np.array(
            sentence_embs[sent_id]
        ).astype('float32') for sent_id in cluster_data['sentence id'].values.tolist() if sent_id in sentence_embs]
        
        # Get sent similarities to centre
        sent_similarities = cosine_similarity(
            np.mean(cluster_coords, axis=0).reshape(1, -1), cluster_coords
        )

        #calculate similarities between ngrams per cluster and cluster mean 
        ngram_similarities = cosine_similarity(np.mean(cluster_embeds, axis=0).reshape(1, -1), ngram_embeddings)

        skills_data[cluster_num] = {
            'Skills name': ' '.join(
                [clean_ngrams[i] for i in ngram_similarities.argsort()[0][::-1].tolist()[0:1]]),
            'Examples': " ".join(
                [
                    cluster_text[i]
                    for i in sent_similarities.argsort()[0][::-1].tolist()[0:num_top_sent]
                ]
            ),
            'Texts': cluster_descriptions[cluster_num]
        }

    return skills_data
