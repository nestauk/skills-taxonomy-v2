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
# ## Find a good sample size for data reduction
# Using a sample of the embeddings, find a good number of data points where the overlap of closest neighbours plateaus.

# %%
import yaml
import random
from tqdm import tqdm
import json

import pandas as pd
import numpy as np
import boto3
from sklearn import metrics
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.neighbors import NearestNeighbors

from skills_taxonomy_v2.getters.s3_data import get_s3_data_paths, save_to_s3, load_s3_data
from skills_taxonomy_v2.pipeline.skills_extraction.extract_skills_utils import (
    load_sentences_embeddings,ExtractSkills
    )
from skills_taxonomy_v2 import BUCKET_NAME

# %%

s3 = boto3.resource("s3")

# %% [markdown]
# ## Load sample of embeddings
# - under 250 characters
# - no repeats
# - up to 2000 from each file

# %%
embeddings_sample_0 = load_s3_data(s3, BUCKET_NAME, "outputs/skills_extraction/word_embeddings/data/2021.11.05_sample_0.json")
embeddings_sample_1 = load_s3_data(s3, BUCKET_NAME, "outputs/skills_extraction/word_embeddings/data/2021.11.05_sample_1.json")
embeddings_sample_2 = load_s3_data(s3, BUCKET_NAME, "outputs/skills_extraction/word_embeddings/data/2021.11.05_sample_2.json")

# %%
embeddings_sample = embeddings_sample_0 + embeddings_sample_1 + embeddings_sample_2

# %%
len(embeddings_sample)

# %%
random.seed(42)
random.shuffle(embeddings_sample)

size_hold_out = 10000

hold_out_data = embeddings_sample[0:size_hold_out]
rest_data = embeddings_sample[size_hold_out:]

print(len(hold_out_data))
print(len(rest_data))

# %%
umap_n_neighbors= 10
umap_min_dist= 0.0
umap_random_state=42
umap_n_components=2

# %%
from collections import defaultdict

# %%
list(range(0,3))

# %%
list(range(3,6))

# %%
# nneighs_holdout = {}
# incremental_overlap = defaultdict(list)

for rep in range(3, 6):
    print(rep)
    
    for i, samp_size in tqdm(enumerate(np.linspace(100, 500000, num=10))):
        samp_size = int(samp_size)

        reducer_class = umap.UMAP(
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            random_state=umap_random_state,
            n_components=umap_n_components,
        )
        random.seed(rep)
        reducer_class.fit(random.sample(rest_data, samp_size))

        hold_out_data_reduced = reducer_class.transform(hold_out_data)

        neigh = NearestNeighbors(n_neighbors=2)
        neigh.fit(hold_out_data_reduced)
        nearest = neigh.kneighbors(hold_out_data_reduced, 2, return_distance=False)

        # Which are the close neighbours?
        close_neighbours = set([tuple(i) for i in nearest])
        nneighs_holdout[samp_size] = close_neighbours

        # Get overlap from previous nearesrt neighbours
        if i!=0:
            incremental_overlap[samp_size].append(
                len(nneighs_holdout[samp_size].intersection(nneighs_holdout[prev_ss])))
        prev_ss = samp_size


# %%
incremental_overlap

# %%
with open('incremental_overlap_12nov.txt', 'w') as file:
    file.write('\n'.join([str(s) for s in incremental_overlap.items()]))

# %%
import ast

with open('incremental_overlap_12nov.txt', 'r') as file:
    incremental_overlap = file.read()
incremental_overlap = [ast.literal_eval(s) for s in incremental_overlap.split('\n')]
incremental_overlap = {k: v for (k,v) in incremental_overlap}

# %%
mean_incremental_overlap = [(k, np.mean(v)) for k,v in incremental_overlap.items()]

# %%
mean_incremental_overlap


# %%
plt.plot([i[0] for i in mean_incremental_overlap], [i[1] for i in mean_incremental_overlap], color="black")
plt.title("Intersection size of the nearest neighbours of hold-out embeddings \nas reducer class is fit on more embeddings")
plt.xlabel("Number of embeddings the reducer class is fitted on")
plt.ylabel("Size of intersection");
plt.savefig("../figures/reduction_emb_sample_size.pdf")


# %% [markdown]
# ## Looking for  different number of components to reduce to
#

# %%
nneighs_holdout_k = {}
incremental_overlap_k = defaultdict(list)

rep = 0

samp_size = 300000
for i, umap_n_components in tqdm(enumerate(range(2, 6))):

    reducer_class = umap.UMAP(
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        random_state=umap_random_state,
        n_components=umap_n_components,
    )
    random.seed(rep)
    reducer_class.fit(random.sample(rest_data, samp_size))

    hold_out_data_reduced = reducer_class.transform(hold_out_data)

    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(hold_out_data_reduced)
    nearest = neigh.kneighbors(hold_out_data_reduced, 2, return_distance=False)

    # Which are the close neighbours?
    close_neighbours = set([tuple(i) for i in nearest])
    nneighs_holdout_k[umap_n_components] = close_neighbours

    # Get overlap from previous nearesrt neighbours
    if i!=0:
        incremental_overlap_k[umap_n_components].append(
            len(nneighs_holdout_k[umap_n_components].intersection(nneighs_holdout_k[prev_ss])))
    prev_ss = umap_n_components


# %%
mean_incremental_overlap_k = [(k, np.mean(v)) for k,v in incremental_overlap_k.items()]

# %%
plt.plot([i[0] for i in mean_incremental_overlap_k], [i[1] for i in mean_incremental_overlap_k])


# %%

