# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Word2Vec approach
#
# - Clean sentences to nice list of words (no stop words, no super common pairs of words, e.g. apply-now)
# - Get mean word2vec for each sentence
# - Cluster skills using this
#
# There might be a more appropriate pre-trained word2vec model for this?

# %%
# cd ../../../..

# %%
import json
from collections import Counter
from itertools import chain, combinations
import time
import random
import re
import yaml
import pickle

import matplotlib.pyplot as plt
from tqdm import tqdm
import spacy
import pandas as pd
import networkx as nx
from cdlib import algorithms, ensemble, evaluation, viz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from gensim import models
from gensim.corpora import Dictionary
import umap.umap_ as umap

# %%
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

# %%
import bokeh.plotting as bpl
from bokeh.plotting import ColumnDataSource, figure, output_file, show
from bokeh.models import (
    BoxZoomTool,
    WheelZoomTool,
    HoverTool,
    SaveTool,
    Label,
    CategoricalColorMapper,
)
from bokeh.io import output_file, reset_output, save, export_png, show
from bokeh.resources import CDN
from bokeh.embed import file_html

bpl.output_notebook()

# %%
from skills_taxonomy_v2.pipeline.skills_extraction.cleaning_sentences import (
    deduplicate_sentences,
    sentences2cleantokens,
    build_ngrams,
    get_common_tuples,
    remove_common_tuples,
)

# %%
config_name = "2021.07.09.small"
# with open(
#     f"outputs/sentence_classifier/data/skill_sentences/{config_name}_jobsnew1_embeddings.pkl", "rb") as file:
#     sentences_vec = pickle.load(file)

with open(
    f"outputs/sentence_classifier/data/skill_sentences/{config_name}_jobsnew1_predictions.pkl",
    "rb",
) as file:
    sentences_pred = pickle.load(file)

with open(
    f"outputs/sentence_classifier/data/skill_sentences/{config_name}_jobsnew1_sentences.pkl",
    "rb",
) as file:
    sentences = pickle.load(file)

# %% [markdown]
# ## Clean

# %%
token_len_threshold = (
    20  # To adjust. I had a look and > this number seems to all be not words (urls etc)
)
lemma_n = 2
top_n_common_remove = 20

# %%
# Filter out the non-skill sentences
sentences = [sentences[i] for i, p in enumerate(sentences_pred.astype(bool)) if p == 1]

# Deduplicate sentences
sentences = deduplicate_sentences(sentences)

# Apply cleaning to the sentences
sentence_words, lemma_sentence_words = sentences2cleantokens(
    sentences, token_len_threshold=token_len_threshold
)

# Get n-grams
sentence_words = build_ngrams(sentence_words, n=lemma_n)
lemma_sentence_words = build_ngrams(lemma_sentence_words, n=lemma_n)

# Remove common co-occurring words
common_word_tuples_set = get_common_tuples(sentence_words, top_n=top_n_common_remove)
sentence_words_clean = remove_common_tuples(sentence_words, common_word_tuples_set)

lemma_common_word_tuples_set = get_common_tuples(
    lemma_sentence_words, top_n=top_n_common_remove
)
lemma_sentence_words_clean = remove_common_tuples(
    lemma_sentence_words, lemma_common_word_tuples_set
)

# %% [markdown]
# ## Embeddings

# %%
# [(k,v.get('file_size'), v['description']) for k, v in api.info()['models'].items()]

# %%
# May be able to find a more appropriate pre-trained model
from gensim.models.word2vec import Word2Vec
import gensim.downloader as api

# %%
wv = api.load("word2vec-google-news-300")

# %%
sentence_mean_vec = []
lemma_sentences_clean_filt = []
bad_words = []

unique_sentences_merged = set()

for sentence_words in lemma_sentence_words_clean:
    sentence_vecs = []
    sentence_words_filt = []
    for word in sentence_words:
        try:
            word_vec = wv[word]
            sentence_vecs.append(word_vec)
            sentence_words_filt.append(word)
        except:
            bad_words.append(word)
    if len(sentence_vecs) != 0:
        sentence_id = "-".join(sorted(sentence_words_filt))
        if sentence_id not in unique_sentences_merged:
            mean_vec = np.mean(np.array(sentence_vecs), axis=0)
            sentence_mean_vec.append(mean_vec)
            lemma_sentences_clean_filt.append(sentence_words_filt)
            unique_sentences_merged.add(sentence_id)

# %%
len(sentence_mean_vec)

# %%
len(lemma_sentences_clean_filt)

# %% [markdown]
# ## Reduce to 2D

# %%
reducer_class = umap.UMAP(n_neighbors=50, min_dist=0.2, random_state=42)
reduced_points_umap = reducer_class.fit_transform(sentence_mean_vec)

reduced_points = reduced_points_umap
reduced_x = reduced_points[:, 0]
reduced_y = reduced_points[:, 1]

# %%
skills_data = pd.DataFrame(
    {
        "reduced_points x": reduced_x,
        "reduced_points y": reduced_y,
        "description": lemma_sentences_clean_filt,
    }
)

# %%
ds_dict = dict(x=reduced_x, y=reduced_y, texts=skills_data["description"].tolist())
hover = HoverTool(
    tooltips=[
        ("node", "@texts"),
    ]
)
source = ColumnDataSource(ds_dict)
p = figure(
    plot_width=500,
    plot_height=500,
    tools=[hover, WheelZoomTool(), BoxZoomTool(), SaveTool()],
    title=f"Skills",
    toolbar_location="below",
)
p.circle(
    x="x",
    y="y",
    radius=0.05,
    alpha=0.5,
    source=source,
)
show(p)


# %% [markdown]
# ## Cluster

# %%
def get_top_tf_idf_words(clusters_vect, feature_names, top_n=2):
    """
    From https://stackoverflow.com/questions/34232190/scikit-learn-tfidfvectorizer-how-to-get-top-n-terms-with-highest-tf-idf-score
    """
    sorted_nzs = np.argsort(clusters_vect.data)[: -(top_n + 1) : -1]
    return feature_names[clusters_vect.indices[sorted_nzs]]


# %%
def cluster_data(
    skills_data_orig, num_clusters, cluster_id=None, random_state=0, num_top_words=5
):

    skills_data = skills_data_orig.copy()
    clustering = KMeans(
        n_clusters=num_clusters, max_iter=1000, random_state=random_state
    )
    try:
        clustering.fit(
            list(zip(skills_data["reduced_points x"], skills_data["reduced_points y"]))
        )
        clustering_number = clustering.labels_
        cluster_centers = clustering.cluster_centers_
    except ValueError:
        # There might not be enough data to cluster
        clustering_number = [0] * len(skills_data)
        cluster_centers = [np.array([0, 0])] * len(skills_data)

    skills_data["Cluster number"] = clustering_number

    # Get information for each cluster
    cluster_sizes = skills_data["Cluster number"].value_counts().to_dict()

    skills_data["description"] = skills_data["description"].apply(" ".join)
    # TFIDF vectors for all words in each cluster
    cluster_texts = (
        skills_data.groupby(["Cluster number"])["description"]
        .apply(" ".join)
        .reset_index()
    )
    cluster_texts = cluster_texts.set_index("Cluster number").T.to_dict("records")[0]

    cluster_vectorizer = TfidfVectorizer(stop_words="english")
    clusters_vects = cluster_vectorizer.fit_transform(cluster_texts.values())

    # Top n words for each cluster + other info
    feature_names = np.array(cluster_vectorizer.get_feature_names())
    cluster_info = {
        cluster_num: {
            "Defining words": "-".join(
                list(get_top_tf_idf_words(clusters_vect, feature_names, num_top_words))
            ),
            "Number skills": int(cluster_sizes[cluster_num]),
            "Cluster center": list(map(float, cluster_centers[cluster_num])),
        }
        for cluster_num, clusters_vect in zip(cluster_texts.keys(), clusters_vects)
    }

    return skills_data, cluster_info


# %%
skills_data_cluster, cluster_info = cluster_data(
    skills_data, 200, cluster_id=None, random_state=0, num_top_words=5
)

# %%
from bokeh.palettes import Plasma, magma, cividis, inferno, plasma, viridis

# %%
colors_by_labels = skills_data_cluster["Cluster number"].astype(str).tolist()
color_palette = viridis

ds_dict = dict(
    x=reduced_x,
    y=reduced_y,
    texts=skills_data_cluster["description"].tolist(),
    label=colors_by_labels,
)
hover = HoverTool(
    tooltips=[
        ("node", "@texts"),
    ]
)
source = ColumnDataSource(ds_dict)
unique_colors = list(set(colors_by_labels))
num_unique_colors = len(unique_colors)

color_palette_cols = color_palette(len(unique_colors))
color_mapper = CategoricalColorMapper(factors=unique_colors, palette=color_palette_cols)

p = figure(
    plot_width=500,
    plot_height=500,
    tools=[hover, WheelZoomTool(), BoxZoomTool(), SaveTool()],
    title=f"Skills",
    toolbar_location="below",
)
p.circle(
    x="x",
    y="y",
    radius=0.05,
    alpha=0.5,
    source=source,
    color={"field": "label", "transform": color_mapper},
)
show(p)


# %% [markdown]
# ## Skills
# Skills are the clusters - could remove ones with not many/too many elements

# %%
plt.hist([c["Number skills"] for c in cluster_info.values()])

# %%
[c["Defining words"] for c in cluster_info.values() if c["Number skills"] > 20]

# %%
pd.DataFrame(cluster_info).T.sort_values("Number skills", ascending=False)

# %%
[c["Defining words"] for c in cluster_info.values()]

# %%
