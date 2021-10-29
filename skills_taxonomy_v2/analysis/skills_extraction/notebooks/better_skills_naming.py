# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: skills-taxonomy-v2
#     language: python
#     name: skills-taxonomy-v2
# ---

# %%
import logging
from collections import Counter
import re
import itertools
import spacy
import pytextrank
from argparse import ArgumentParser
import yaml
import pandas as pd
from tqdm import tqdm
import boto3
import pytextrank

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
from skills_taxonomy_v2.getters.s3_data import load_s3_data, save_to_s3
from skills_taxonomy_v2 import BUCKET_NAME

from pattern.text.en import singularize
from collections import OrderedDict
import random
from collections import defaultdict
import time

logger = logging.getLogger(__name__)

nltk.download("wordnet")


# %% [markdown]
# # 0. Set up
# ## 0.1 load functions

# %%
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


def clean_cluster_descriptions(sentences_data):  # remove job based stop words!
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

    # only return ngrams that are more than 1 word long
    return [
        clean for clean in clean_ngrams if len(clean.split(" ")) > 1
    ], cluster_descriptions


def get_skill_info(
    sentence_skills, sentence_embs, num_top_sent=2, ngram=4, min_count=1, threshold=0.15
):
    """
    Output: skills_data (dict), for each skill number:
        'Skills name' : join the closest ngram to the centroid
        'Examples': Join the num_top_sent closest original sentences to the centroid
        'Skills name embed': embedding of closest ngram to the centroid or shortest description embedding
        'Texts': All the cleaned sentences for this cluster
    """

    start_time = time.time()

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

        # generate candidate ngrams per sentence cluster and embed
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
                "Skills name embed": candidate_ngrams_embeds[
                    int(ngram_similarities.argsort()[0][::-1].tolist()[0:1][0])
                ],
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

        else:
            print(
                "no candidate ngrams"
            )  # if no candidate ngrams are generated, skill name is smallest skill description
            skills_data[cluster_num] = {
                "Skills name": min(cluster_descriptions[cluster_num], key=len),
                "Name method": "minimum description",
                "Skills name embed": bert_vectorizer.transform(
                    [min(cluster_descriptions[cluster_num], key=len)]
                )[0],
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

    print("--- %s seconds ---" % (time.time() - start_time))

    return skills_data


# %% [markdown]
# ## 0.2. load data

# %%
s3 = boto3.resource("s3")

# Load data
sentence_skills_path = (
    "outputs/skills_extraction/extracted_skills/2021.08.31_sentences_data.json"
)
embedding_sample_path = "outputs/skills_extraction/extracted_skills/2021.08.31_sentence_id_2_embedding_dict.json.gz"
sentence_skills = load_s3_data(s3, BUCKET_NAME, sentence_skills_path)
sentence_skills = pd.DataFrame(sentence_skills)
sentence_skills = sentence_skills[sentence_skills["Cluster number"] != -1]
sentence_embs = load_s3_data(s3, BUCKET_NAME, embedding_sample_path)

# %% [markdown]
# ## 0.3. get sample based on random clusters

# %%
# random sample
k = 100
random_clusters = random.sample(
    list(set(sentence_skills["Cluster number"].tolist())), k
)
sentence_skills_sample = sentence_skills[
    sentence_skills["Cluster number"].isin(random_clusters)
]

# %% [markdown]
# ## 0.4. run updated get_skill_info on sample of k clusters
# * updated text cleaning to a) get rid of job specific language, b) singularise terms, c) get rid of duplicate term phrases i.e. 'day day'
# * updated get_skill_info to generate phrases _per_ skill cluster and to assign minimum text description if less than two phrases per cluster are generated
# * lowered Phrases algorithm threshold parameters to generate more phrases per cluster
# * return dictionary incl. closest ngram embed for cluster merging and candidate_ngram list per cluster for updated skill name per merged skill cluster

# %%
named_skills = get_skill_info(sentence_skills_sample, sentence_embs)

# %% [markdown]
# # 1. Skill clusters EDA

# %%
# how large are skill clusters?
cluster_counts = (
    sentence_skills_sample.groupby("Cluster number")
    .count()
    .sort_values("description", ascending=False)
)
cluster_counts[cluster_counts["description"] > 10]
cluster_counts[cluster_counts["description"] <= 5]
cluster_counts["description"].plot.hist(
    bins=15, range=[5, 50]
)  # vast majority of clusters are quite small!

# %% [markdown]
# # 2. Experiment w/ summarisation methods

# %% [markdown]
# ## 2.0 PyTextRank

# %%
all_texts = " ".join(sentence_skills[:10]["description"].tolist())

en_nlp = spacy.load("en_core_web_sm")
en_nlp.add_pipe("textrank")
doc = en_nlp(all_texts)
candidate_phrases = list(
    set(
        [
            phrase.text.strip()
            for phrase in doc._.phrases
            if 1 < len(phrase.text.split(" ")) < 4
        ]
    )
)

# %% [markdown]
# ## 2.1 Noun Chunks

# %%
nlp = spacy.load("en_core_web_sm")
doc = nlp(all_texts)

noun_chunks = []
for chunk in doc.noun_chunks:
    noun_chunks.append(chunk.text)

candidate_chunks = [
    noun.strip() for noun in noun_chunks if 2 <= len(noun.strip().split(" ")) <= 4
]
print(candidate_chunks)


# %% [markdown]
# ## 3.0 PyTextRank w embeddings, verb chunking experiments

# %%
def get_clean_ngrams_pytextrank(sentence_skills, min_ngram=2, max_ngram=3):
    # Clean sentences
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("textrank")

    cluster_descriptions = clean_cluster_descriptions(sentence_skills)
    # get cluster texts
    cluster_texts = [" ".join(sentences) for sentences in cluster_descriptions.values()]

    candidate_chunks = []
    for cluster_text in cluster_texts:
        doc = nlp(cluster_text)
        # get rid of double terms
        chunks = [phrase.text for phrase in doc._.phrases]
        clean_chunks = [
            " ".join(OrderedDict((w, w) for w in ngrm.split(" "))) for ngrm in chunks
        ]
        candidate_chunks.append(
            [
                chunk
                for chunk in clean_chunks
                if min_ngram <= len(chunk.split(" ")) <= max_ngram
            ]
        )

    return list(itertools.chain(*candidate_chunks)), cluster_descriptions


# %%
def get_clean_ngrams_chunks(sentence_skills):

    nlp = spacy.load("en_core_web_sm")
    patterns = [
        "VERB NOUN",
        "NOUN VERB",
        "VERB ADJ NOUN",
        "VERB NOUN NOUN",
    ]  # experiment with this!
    re_patterns = [" ".join(["(\w+)_!" + pos for pos in p.split()]) for p in patterns]

    # Clean sentences
    cluster_descriptions = clean_cluster_descriptions(sentence_skills)
    # get cluster texts
    cluster_texts = [" ".join(sentences) for sentences in cluster_descriptions.values()]

    candidate_chunks = []
    for cluster_text in cluster_texts:
        doc = nlp(cluster_text)
        text_pos = " ".join([token.text + "_!" + token.pos_ for token in doc])
        candidate_chunk = [
            [" ".join(result) for result in re.findall(pattern, text_pos)]
            for i, pattern in enumerate(re_patterns)
        ]
        candidate_chunks.append(candidate_chunk)

    return (
        list(
            set(itertools.chain(*list(itertools.chain.from_iterable(candidate_chunks))))
        ),
        cluster_descriptions,
    )


# %%
# get skill info w/ pytextrank embeddings

start_time = time.time()

bert_vectorizer = BertVectorizer(
    bert_model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
    multi_process=True,
)
bert_vectorizer.fit()

skills_data = {}
for cluster_num, cluster_data in tqdm(sentence_skills_sample.groupby("Cluster number")):
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

    # ----GENERATE CANDIDATE NGRAMS BASED ON PYTEXTRANK----#
    candidate_ngrams, cluster_descriptions = get_clean_ngrams_pytextrank(
        cluster_data, min_ngram=2, max_ngram=3
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
            "Name method": "pytextrank_embedding",
            "Skills name embed": candidate_ngrams_embeds[
                int(ngram_similarities.argsort()[0][::-1].tolist()[0:1][0])
            ],
            "Examples": " ".join(
                [
                    cluster_text[i]
                    for i in sent_similarities.argsort()[0][::-1].tolist()[0:2]
                ]
            ),
            "Texts": cluster_descriptions[cluster_num],
        }

    else:
        print(
            "no candidate ngrams"
        )  # if no candidate ngrams are generated, skill name is smallest skill description
        skills_data[cluster_num] = {
            "Skills name": min(cluster_descriptions[cluster_num], key=len),
            "Name method": "minimum description",
            "Skills name embed": bert_vectorizer.transform(
                [min(cluster_descriptions[cluster_num], key=len)]
            )[0],
            "Examples": " ".join(
                [
                    cluster_text[i]
                    for i in sent_similarities.argsort()[0][::-1].tolist()[0:2]
                ]
            ),
            "Texts": cluster_descriptions[cluster_num],
        }

print("--- %s seconds ---" % (time.time() - start_time))

# %%
# get skill info w/ verb chunking

start_time = time.time()

skills_data = {}
for cluster_num, cluster_data in tqdm(sentence_skills_sample.groupby("Cluster number")):
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

    # ----GENERATE CANDIDATE NGRAMS BASED ON VERB CHUNKING----#
    candidate_ngrams, cluster_descriptions = get_clean_ngrams_chunks(cluster_data)

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
            "Name method": "verb_chunks_embedding",
            "Skills name embed": candidate_ngrams_embeds[
                int(ngram_similarities.argsort()[0][::-1].tolist()[0:1][0])
            ],
            "Examples": " ".join(
                [
                    cluster_text[i]
                    for i in sent_similarities.argsort()[0][::-1].tolist()[0:2]
                ]
            ),
            "Texts": cluster_descriptions[cluster_num],
        }

    else:
        print(
            "no candidate ngrams"
        )  # if no candidate ngrams are generated, skill name is smallest skill description
        skills_data[cluster_num] = {
            "Skills name": min(cluster_descriptions[cluster_num], key=len),
            "Name method": "minimum description",
            "Skills name embed": bert_vectorizer.transform(
                [min(cluster_descriptions[cluster_num], key=len)]
            )[0],
            "Examples": " ".join(
                [
                    cluster_text[i]
                    for i in sent_similarities.argsort()[0][::-1].tolist()[0:2]
                ]
            ),
            "Texts": cluster_descriptions[cluster_num],
        }

print("--- %s seconds ---" % (time.time() - start_time))


# %% [markdown]
# # 3. Merge based on named skills proximity AND centroid proximity

# %%
def merge_skill_clusters(
    named_skills,
    sentence_skills,
    skill_name_sim_threshold=0.9,
    centroid_threshold=0.9,
):

    skill_name_sims = cosine_similarity(
        np.array(
            [
                skill_data["Skills name embed"]
                for skill_clust, skill_data in named_skills.items()
            ]
        )
    )

    duplicate_skills = []
    for sims in skill_name_sims:
        sims_indexes = np.where(sims > skill_name_sim_threshold)[0]
        if len(sims_indexes) > 1:
            print([sims[inds] for inds in sims_indexes])
            sim_skill_ids = [list(named_skills.keys())[inds] for inds in sims_indexes]
            duplicate_skills.append(sim_skill_ids)

    # remove duplicate similar skill ids
    duplicate_skills = list(map(list, set(map(frozenset, duplicate_skills))))
    # remove subsets
    # https://stackoverflow.com/questions/1318935/python-list-filtering-remove-subsets-from-list-of-lists
    duplicate_skills = [
        x
        for x in duplicate_skills
        if not any(set(x) <= set(y) for y in duplicate_skills if x is not y)
    ]
    # for skill names that are similar in semantic space AND skill cluster centroids are close, merge skills
    merged_skills = dict()
    for skill_ids in duplicate_skills:
        clust_centroids = [
            np.mean(
                [
                    np.array(embeds).astype("float32")
                    for embeds in sentence_skills[
                        sentence_skills["Cluster number"] == skill_id
                    ]["reduced_points_umap"].values
                ],
                axis=0,
            )
            for skill_id in skill_ids
        ]
        centroid_sims = cosine_similarity(clust_centroids)
        centroid_sims = centroid_sims[~np.eye(len(centroid_sims), dtype=bool)].reshape(
            len(centroid_sims), -1
        )  # remove diagonal values of cosine sims to itself
        # if ALL cluster centroids very close together, merge skills
        if (
            all([all(list(sim > centroid_threshold)) == True for sim in centroid_sims])
            == True
        ):
            merged_skills["_".join([str(skill_id) for skill_id in skill_ids])] = {
                "Skills name": named_skills[skill_ids[0]]['Skills name'],
                "Texts": list(
                    itertools.chain(
                        *[named_skills[skill_id]["Texts"] for skill_id in skill_ids]
                    )
                ),
            }

    if merged_skills:
        flat_dup_skills = list(itertools.chain(*duplicate_skills))
        named_skills_no_dups = {
            key: named_skills[key] for key in named_skills if key not in flat_dup_skills
        }

        return dict(named_skills_no_dups, **merged_skills)

    else:
        return named_skills


# %%
merged_skills = merge_skill_clusters(
    named_skills,
    sentence_skills_sample,
    skill_name_sim_threshold=0.8,
    centroid_threshold=0.9,
)

# %%
