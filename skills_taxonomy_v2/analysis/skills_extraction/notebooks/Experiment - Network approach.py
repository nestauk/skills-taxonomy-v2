# -*- coding: utf-8 -*-
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
# ## Find skill communities using a network approach
# After processing job adverts, taking out no skills sentences, and separating and cleaning words.

# %%
# cd ../../../..

# %%
from skills_taxonomy_v2.pipeline.skills_extraction.cleaning_sentences import (
    deduplicate_sentences,
    sentences2cleantokens,
    build_ngrams,
    get_common_tuples,
    remove_common_tuples,
)
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
import numpy as np
from gensim import models
from gensim.corpora import Dictionary

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

# %% [markdown]
# ## Load list of sentences/predictions from a sample of job adverts

# %%
config_name = "2021.07.09.small"

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
# To adjust. I had a look and > this number seems to all be not words (urls etc)
token_len_threshold = 20
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


# %%
print(len(sentences))


# %% [markdown]
# ## Build network from co-occurences

# %%
def get_cooccurences_network(sentence_words):
    # Sort list of words in alphabetical order
    # so you get tuples in the same order e.g. ('A_word', 'B_word') not ('B_word', 'A_word')
    pairs = list(chain(*[list(combinations(sorted(x), 2)) for x in sentence_words]))
    pairs = [x for x in pairs if len(x) > 0]

    edge_list = pd.DataFrame(pairs, columns=["source", "target"])
    edge_list["weight"] = 1
    edge_list_weighted = (
        edge_list.groupby(["source", "target"])["weight"].sum().reset_index(drop=False)
    )
    # Build network
    net = nx.from_pandas_edgelist(edge_list_weighted, edge_attr=True)

    # # Filter out low frequency words
    # thres = 0.1
    # freq = pd.Series(list(chain(*sentences_broken))).value_counts() / len(sentences_broken)
    # to_keep = freq.loc[freq < thres].index.tolist()
    # net_filt = net.subgraph(to_keep)

    return net


# %%
net = get_cooccurences_network(lemma_sentence_words_clean)

# %% [markdown]
# ## Which nodes are highly connected?
#
# Prune the network
# - nodes with lots of connections
# - edges with high weights
#
# Could the resulting nodes be the skills? (rather than go through community detection too)

# %%
num_edges = [degree for node, degree in dict(net.degree()).items()]
plt.hist(num_edges, bins=100)

# %%
all_weights = [w["weight"] for s, e, w in edges]
plt.hist(all_weights, bins=100)


# %%
def get_word_stats(word):
    word_weights = [
        w["weight"] for s, e, w in list(net.edges(data=True)) if s == word or e == word
    ]
    return dict(net.degree())[word], np.mean(word_weights), np.median(word_weights)


# %%
node_stats = {}
for node in dict(net.nodes(data=True)).keys():
    node_stats[node] = get_word_stats(node)

# %%
x = [n[0] for n in node_stats.values()]  # Number of connections
# 1: mean, 2: median weight of all connections
y = [n[1] for n in node_stats.values()]
hover_text = list(node_stats.keys())
ds_dict = dict(x=x, y=y, texts=hover_text)
hover = HoverTool(tooltips=[("node", "@texts"),])
source = ColumnDataSource(ds_dict)
p = figure(
    plot_width=500,
    plot_height=500,
    tools=[hover, WheelZoomTool(), BoxZoomTool(), SaveTool()],
    title=f"Words",
    x_axis_label="Number of connections to other words",
    y_axis_label="Mean weight of all connections",
    toolbar_location="below",
    x_range=(0, 300),
    y_range=(0, 6),
)
p.circle(
    x="x", y="y", radius=1, alpha=0.7, source=source,
)
show(p)

# %%
# Remove highly connected nodes
highly_connected = [node for node, degree in dict(net.degree()).items() if degree > 100]
net.remove_nodes_from(highly_connected)

# %%
edges = list(net.edges(data=True))
high_weights = ((s, e) for s, e, w in edges if w["weight"] > 2)
net.remove_edges_from(high_weights)

# %% [markdown]
# ## Look at word frequency

# %%

# %%
words_frequency = pd.Series(
    list(chain(*lemma_sentence_words_clean))
).value_counts() / len(lemma_sentence_words_clean)

# %%
plt.hist(words_frequency, 100, density=True, facecolor="g", alpha=0.75)

# %% [markdown]
# ## Remove low TFIDF terms (common)

# %%
job_d_words_merged = [" ".join(c) for c in lemma_sentence_words_clean]


# %%
# https://kavita-ganesan.com/extracting-keywords-from-text-tfidf/#.YOhlIBNueWA


def get_word_values(tfidf_vects, feature_names):
    coo_matrix = tfidf_vects.tocoo()
    tuples = zip(coo_matrix.col, coo_matrix.data)

    score_vals = []
    feature_vals = []

    for idx, score in tuples:
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of feature,score
    # results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results


def filter_extreme_results(results_dict, filter_top=0.1, filter_bottom=0.1):
    sorted_words = [
        k for k, v in sorted(results_dict.items(), key=lambda item: item[1])
    ]
    return sorted_words[
        round(len(sorted_words) * filter_bottom) : round(
            len(sorted_words) * (1 - filter_top)
        )
    ]


# %%
filter_top = 0.1
filter_bottom = 0.1

tfidf_vectorizer = TfidfVectorizer(stop_words="english")
tfidf_vects = tfidf_vectorizer.fit_transform(job_d_words_merged)

feature_names = tfidf_vectorizer.get_feature_names()

results_dict = get_word_values(tfidf_vects, feature_names)
keep_words = filter_extreme_results(
    results_dict, filter_top=filter_top, filter_bottom=filter_bottom
)


# %%
net_filt = net.subgraph(keep_words)

# %% [markdown]
# ## Community detection algorithms
# Trying out different ones

# %%
# def best_cd_algo(net):
#     method_pars = {
#         algorithms.louvain: [
#             ensemble.Parameter(name="resolution", start=0.7, end=1, step=0.1)
#         ],
#         algorithms.leiden:[ensemble.BoolParameter(name='weights',value='weight')],
#         algorithms.agdl: [ensemble.Parameter(name="number_communities", start=20, end=50, step=10),
#         ensemble.Parameter(name="kc", start=4, end=10, step=2)],
#         algorithms.chinesewhispers: [],
#         algorithms.label_propagation: [],
#         algorithms.markov_clustering: [],
#     }

#     qual=evaluation.erdos_renyi_modularity
#     aggregate=max

#     algos = list(method_pars.keys())
#     pars = list(method_pars.values())

#     results = ensemble.pool_grid_filter(net, algos, pars, qual, aggregate=aggregate)

#     results_container = []
#     comm_assignments_container = {}
#     for comm, score in results:
#         try:
#             print(comm.method_name)
#             out = [
#                 comm.method_name,
#                 len(comm.communities),
#                 comm.method_parameters,
#                 score.score,
#                 comm,
#             ]
#             results_container.append(out)
#             comm_assignments_container[comm.method_name] = comm.communities
#         except:
#             print(f"error with algorithm {comm.method_name}")
#             pass

#     results_df = pd.DataFrame(
#         results_container,
#         columns=["method", "comm_n", "parametres", "score", "instance"],
#     )

#     return results_df, comm_assignments_container

# %%
# results_df, comm_assignments_container = best_cd_algo(net)

# %% [markdown]
# ## Or just one

# %%
coms = algorithms.louvain(net_filt, weight="weight", resolution=0.7, randomize=None)

# %%
communities = coms.communities

# %%
len(communities)


# %% [markdown]
# ## Name communities by top TFIDF terms

# %%
def get_top_tf_idf_words(clusters_vect, feature_names, top_n=2):
    """
    From https://stackoverflow.com/questions/34232190/scikit-learn-tfidfvectorizer-how-to-get-top-n-terms-with-highest-tf-idf-score
    """
    sorted_nzs = np.argsort(clusters_vect.data)[: -(top_n + 1) : -1]
    return feature_names[clusters_vect.indices[sorted_nzs]]


# %%
communities_merged = [" ".join(c) for c in communities]

# %%
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
tfidf_vects = tfidf_vectorizer.fit_transform(communities_merged)

num_top_words = 5
# Top n words for each cluster + other info
feature_names = np.array(tfidf_vectorizer.get_feature_names())
cluster_info = {
    i: "-".join(list(get_top_tf_idf_words(document, feature_names, num_top_words)))
    for i, document in enumerate(tfidf_vects)
}

# %%
cluster_info

# %% [markdown]
# # Visualise

# %% [markdown]
# ## 1. Sentences network

# %%
# Plot and colour nodes by community
# top_k – int, Show the top K influential communities. If set to zero or negative value indicates all.
# min_size – int, Exclude communities below the specified minimum size.
pos = nx.spring_layout(net_filt)
viz.plot_network_clusters(
    net_filt, coms, pos, figsize=(15, 15), node_size=50, plot_labels=False
)
