# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Word embeddings from transformers
#
# New approach: mask out sentence of unclean words, then find sentence embedding.
#
# Cluster sentence embeddings of all skill sentences with certain words cleaned out.

# %%
# cd ../../../..

# %%
from skills_taxonomy_v2.getters.s3_data import get_s3_data_paths, load_s3_data

# %%
from skills_taxonomy_v2.pipeline.skills_extraction.cleaning_sentences import (
    get_common_tuples, build_ngrams
)

# %%
import json
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import umap.umap_ as umap
import boto3

import bokeh.plotting as bpl
from bokeh.plotting import ColumnDataSource, figure, output_file, show
from bokeh.models import (
    BoxZoomTool,
    WheelZoomTool,
    HoverTool,
    SaveTool,
    Label,
    CategoricalColorMapper,
    ColorBar,
    ColumnDataSource
)
from bokeh.io import output_file, reset_output, save, export_png, show
from bokeh.resources import CDN
from bokeh.embed import file_html
from bokeh.palettes import Plasma, magma, cividis, inferno, plasma, viridis, Spectral6
from bokeh.transform import linear_cmap

bpl.output_notebook()

# %% [markdown]
# ## Load data from S3

# %%
bucket_name = "skills-taxonomy-v2"
s3 = boto3.resource("s3")
word_embeddings_dir = 'outputs/skills_extraction/word_embeddings/data/'

# %%
word_embeddings_dirs = get_s3_data_paths(s3, bucket_name, word_embeddings_dir, file_types=["*.json"])
len(word_embeddings_dirs)

# %%
# The files are quite big!
# Load them all in one, but don't include exact repeats
sentence_mean_embeddings = []
sentence_words = []
unique_sentences = set()
counter_disgard = 0
for embedding_dir in word_embeddings_dirs:
    word_embeddings = load_s3_data(s3, bucket_name, embedding_dir)
    print(f"Length of {embedding_dir} is {len(word_embeddings)}")
    for words, embedding in word_embeddings:
        if words not in unique_sentences:
            unique_sentences.add(words)
            sentence_mean_embeddings.append(embedding)
            cleaned_words = words.replace('[MASK]', '')
            sentence_words.append(words)#.replace('[MASK]', ''))
#         joined_sentence = '_'.join(words)
#         if joined_sentence not in unique_sentences:
#             unique_sentences.add(joined_sentence)
#             sentence_mean_embeddings.append(embedding)
#             sentence_words.append(words)
        else:
            counter_disgard += 1

# %%
counter_disgard

# %%
sentence_words[0]

# %%
# Before we had 3253 sentences, now we have about 47875
len(sentence_words)

# %%
len(sentence_mean_embeddings[0])

# %% [markdown]
# ## Clean and get mean of remaining clean words
# - filter out commonly occuring not-skill words
# - take average of remaining embeddings
# - remove repeats
#
# You should probably do this in get_word_embeddings too

# %%
from collections import Counter
flat_sentence_words = [item for sublist in sentence_words for item in sublist.split(' ')]
Counter(flat_sentence_words).most_common(20)

# %% [markdown]
# ## Remove sentences with too much masking
# If the sentence is long and most of the words are masked it isn't ideal - this scenario skews the clusters since the average embedding has little influence from the actual not-masked words.

# %%
sentence_words[10]

# %%
prop_not_masked = []
for words in sentence_words:
    words_without_mask = words.replace('[MASK]', '')
    prop_not_masked.append(len(words_without_mask)/len(words))
    

# %%
plt.hist(prop_not_masked, 100, density=True, facecolor='g', alpha=0.75);

# %%
prop_threshold = 0.2
keep_index = [i for i, prop in enumerate(prop_not_masked) if prop>prop_threshold]
print(f"Keeping {len(keep_index)} sentences from {len(prop_not_masked)}")
sentence_mean_embeddings_filt = [sentence_mean_embeddings[i] for i in keep_index]
sentence_words_filt = [sentence_words[i] for i in keep_index]

# %% [markdown]
# ## Reduce to 2D

# %%
reducer_class = umap.UMAP(n_neighbors=50, min_dist=0.2, random_state=42)
reduced_points_umap = reducer_class.fit_transform(sentence_mean_embeddings_filt)

reduced_points = reduced_points_umap
reduced_x = reduced_points[:, 0]
reduced_y = reduced_points[:, 1]

# %%
skills_data = pd.DataFrame(
    {
        "reduced_points x": reduced_x,
        "reduced_points y": reduced_y,
        "description": [s.replace('[MASK]', '') for s in sentence_words_filt],
        "number of words": [len(s) for s in sentence_words_filt]
    }
)

# %%
num_words = skills_data["number of words"].tolist()

# %%
colour_by_list = num_words

# %%
ds_dict = dict(x=reduced_x,
               y=reduced_y,
               texts=skills_data["description"].tolist(),
               cols=colour_by_list)
mapper = linear_cmap(field_name='cols', palette=Spectral6 ,low=min(colour_by_list) ,high=max(colour_by_list))
hover = HoverTool(
    tooltips=[
        ("node", "@texts")#,("colour by", "@cols"),
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
    radius=0.02,
    alpha=0.5,
    source=source,
    color=mapper
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

    skills_data["description"] = skills_data["description"]#.apply(" ".join)
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
    skills_data, 50, cluster_id=None, random_state=0, num_top_words=5
)

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

# %%
[c['Defining words'] for c in cluster_info.values()]

# %%
