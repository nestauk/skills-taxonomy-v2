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
    get_common_tuples,
    build_ngrams,
)

# %%
import json
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
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
    ColumnDataSource,
    LinearColorMapper,
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
word_embeddings_dir = "outputs/skills_extraction/word_embeddings/data/"

# %%
word_embeddings_dirs = get_s3_data_paths(
    s3, bucket_name, word_embeddings_dir, file_types=["*.json"]
)
word_embeddings_dirs = [
    file_dir for file_dir in word_embeddings_dirs if "embeddings.json" in file_dir
]
len(word_embeddings_dirs)

# %%
# The files are quite big!
# Load them all in one, but don't include exact repeats
sentence_mean_embeddings = []
sentence_words = []
sentence_job_ids = []
sentence_ids = []

unique_sentences = set()
counter_disgard = 0
for embedding_dir in word_embeddings_dirs:
    word_embeddings = load_s3_data(s3, bucket_name, embedding_dir)
    print(f"Length of {embedding_dir} is {len(word_embeddings)}")
    for job_id, sent_id, words, embedding in word_embeddings:
        if words not in unique_sentences:
            unique_sentences.add(words)
            sentence_mean_embeddings.append(embedding)
            cleaned_words = words.replace("[MASK]", "")
            sentence_words.append(words)
            sentence_job_ids.append(job_id)
            sentence_ids.append(sent_id)
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

flat_sentence_words = [
    item for sublist in sentence_words for item in sublist.split(" ")
]
Counter(flat_sentence_words).most_common(20)

# %% [markdown]
# ## Remove sentences with too much masking
# If the sentence is long and most of the words are masked it isn't ideal - this scenario skews the clusters since the average embedding has little influence from the actual not-masked words.
#
# So we set a threshold which the proportion of not masked words has to be over:
# ```
# sentence length without masked words / sentence length with masked words > 0.2
# ```

# %%
sentence_words[10]

# %%
prop_not_masked = []
for words in sentence_words:
    words_without_mask = words.replace("[MASK]", "")
    prop_not_masked.append(len(words_without_mask) / len(words))


# %%
plt.hist(prop_not_masked, 100, density=True, facecolor="g", alpha=0.75)

# %%
prop_threshold = 0.2
keep_index = [i for i, prop in enumerate(prop_not_masked) if prop > prop_threshold]
print(f"Keeping {len(keep_index)} sentences from {len(prop_not_masked)}")

sentence_mean_embeddings_filt = [sentence_mean_embeddings[i] for i in keep_index]
sentence_words_filt = [sentence_words[i] for i in keep_index]
sentence_job_ids_filt = [sentence_job_ids[i] for i in keep_index]
sentence_ids_filt = [sentence_ids[i] for i in keep_index]

# %% [markdown]
# ## Reduce to 2D
#
# The `min_dist` parameter controls how tightly UMAP is allowed to pack points together. It, quite literally, provides the minimum distance apart that points are allowed to be in the low dimensional representation. This means that low values of min_dist will result in clumpier embeddings. This can be useful if you are interested in clustering, or in finer topological structure. Larger values of min_dist will prevent UMAP from packing points together and will focus on the preservation of the broad topological structure instead.
#
# `n_neighbors` This parameter controls how UMAP balances local versus global structure in the data. It does this by constraining the size of the local neighborhood UMAP will look at when attempting to learn the manifold structure of the data. This means that low values of n_neighbors will force UMAP to concentrate on very local structure (potentially to the detriment of the big picture), while large values will push UMAP to look at larger neighborhoods of each point when estimating the manifold structure of the data, losing fine detail structure for the sake of getting the broader of the data.
#
#

# %%
# min_dist was 0.0001 and it complained
# /Users/elizabethgallagher/miniconda3/envs/skills-taxonomy-v2/lib/python3.8/site-packages/umap/spectral.py:255: UserWarning: WARNING: spectral initialisation failed! The eigenvector solver
# failed. This is likely due to too small an eigengap. Consider
# adding some noise or jitter to your data.

# %%
reducer_class = umap.UMAP(n_neighbors=3, min_dist=0.001, random_state=42)
reduced_points_umap = reducer_class.fit_transform(sentence_mean_embeddings_filt)

reduced_points = reduced_points_umap
reduced_x = reduced_points[:, 0]
reduced_y = reduced_points[:, 1]

# %%
# with open("outputs/skills_extraction/reduced_data.json", "w") as file:
#     json.dump(reduced_points_umap.tolist(), file)
# with open("outputs/skills_extraction/sentence_words_filt.json", "w") as file:
#     json.dump(sentence_words_filt, file)

# %%
skills_data = pd.DataFrame(
    {
        "reduced_points x": reduced_x,
        "reduced_points y": reduced_y,
        "description": [s.replace("[MASK]", "") for s in sentence_words_filt],
        "number of words": [len(s) for s in sentence_words_filt],
        "job id": sentence_job_ids_filt,
        "sentence id": sentence_ids_filt,
    }
)

# %%
num_words = skills_data["number of words"].tolist()
colour_by_list = num_words

ds_dict = dict(
    x=reduced_x,
    y=reduced_y,
    texts=skills_data["description"].tolist(),
    cols=colour_by_list,
)
mapper = linear_cmap(
    field_name="cols",
    palette=Spectral6,
    low=min(colour_by_list),
    high=max(colour_by_list),
)
hover = HoverTool(tooltips=[("node", "@texts")])  # ,("colour by", "@cols"),
source = ColumnDataSource(ds_dict)
p = figure(
    plot_width=500,
    plot_height=500,
    tools=[hover, WheelZoomTool(), BoxZoomTool(), SaveTool()],
    title=f"Skills",
    toolbar_location="below",
)
p.circle(x="x", y="y", radius=0.05, alpha=0.5, source=source, color=mapper)
show(p)


# %% [markdown]
# ## Cluster to get skills
#
# ### DSCAN
# `eps`, default=0.5 The maximum distance between two samples for one to be considered as in the neighborhood of the other. This is not a maximum bound on the distances of points within a cluster. This is the most important DBSCAN parameter to choose appropriately for your data set and distance function.
#
# `min_samples`, default=5 The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.

# %%
def cluster_data(
    skills_data_orig,
    num_clusters,
    cluster_id=None,
    random_state=0,
    num_top_words=5,
    clust_type="kmeans",
):

    skills_data = skills_data_orig.copy()
    if clust_type == "kmeans":
        clustering = KMeans(
            n_clusters=num_clusters, max_iter=1000, random_state=random_state
        )
        try:
            clustering.fit(
                list(
                    zip(
                        skills_data["reduced_points x"], skills_data["reduced_points y"]
                    )
                )
            )
            clustering_number = clustering.labels_
            cluster_centers = clustering.cluster_centers_
        except ValueError:
            # There might not be enough data to cluster
            clustering_number = [0] * len(skills_data)
            cluster_centers = [np.array([0, 0])] * len(skills_data)
    elif clust_type == "dbscan":
        clustering = DBSCAN(eps=0.05, min_samples=3)
        clustering_number = clustering.fit_predict(reduced_points_umap).tolist()
        print(f"{len(set(clustering_number))} unique clusters")
    else:
        print("Use another clust_type")

    skills_data["Cluster number"] = clustering_number

    # Get information for each cluster
    cluster_sizes = skills_data["Cluster number"].value_counts().to_dict()

    skills_data["description"] = skills_data["description"]  # .apply(" ".join)
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
            "Number skills": int(cluster_sizes[cluster_num]),
        }
        for cluster_num, clusters_vect in zip(cluster_texts.keys(), clusters_vects)
    }

    return skills_data, cluster_info


# %%
skills_data_cluster, cluster_info = cluster_data(
    skills_data,
    1000,
    cluster_id=None,
    random_state=0,
    num_top_words=5,
    clust_type="dbscan",
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

# color_palette_cols = color_palette(len(unique_colors))
# color_mapper = CategoricalColorMapper(factors=unique_colors, palette=color_palette_cols)

color_mapper = LinearColorMapper(palette="Turbo256", low=0, high=len(unique_colors) + 1)

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
# Number of sentences in each cluster (not inc not clustered 'cluster' -1)
plt.hist(
    [c["Number skills"] for k, c in cluster_info.items() if k != -1],
    100,
    facecolor="g",
    alpha=0.75,
)

# %%
skills_data_cluster.to_csv("outputs/skills_extraction/data/clustered_data.csv")

# %%
# With the none clusters - these are transversial, so recluster these separately?


# %% [markdown]
# The final taxonomy can be seen in the diagram below and has a tree-like structure with three layers. The first layer contains 6 broad clusters of skills; these split into 35 groups, and then split once more to give 143 clusters of specific skills. Each of the approximately 10,500 skills lives within one of these 143 skill groups. The same methodology could be used to create further layers.

# %%
