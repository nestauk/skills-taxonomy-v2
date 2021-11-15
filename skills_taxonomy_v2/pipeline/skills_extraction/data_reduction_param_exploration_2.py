import yaml
import random
from tqdm import tqdm
import json

import pandas as pd
import numpy as np
import boto3
import umap.umap_ as umap

from skills_taxonomy_v2.getters.s3_data import get_s3_data_paths, save_to_s3, load_s3_data
from skills_taxonomy_v2 import BUCKET_NAME

s3 = boto3.resource("s3")

print("Loading sample of embeddings...")
embeddings_sample = load_s3_data(s3, BUCKET_NAME, "outputs/skills_extraction/word_embeddings/data/2021.11.05_sample_300k.json")
print(f"Loaded {len(embeddings_sample)} embeddings")

# These to stay the same
umap_random_state= 42
umap_n_components = 2

print(f"Reducing sample of {len(embeddings_sample)} embeddings with different parameters and saving ...")
umap_min_dist = 0.0
for umap_n_neighbors in tqdm([3,4,5,6,7,8,9,10,15,20,30]):
    reducer_class = umap.UMAP(
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            random_state=umap_random_state,
            n_components=umap_n_components,
        )
    reducer_class.fit(embeddings_sample)
    reduced_points_umap_sample = reducer_class.transform(embeddings_sample)
    dict_obj = {
        'umap_min_dist': umap_min_dist,
        'umap_n_neighbors': umap_n_neighbors,
        'umap_random_state': umap_random_state,
        'umap_n_components': umap_n_components,
        'reduced_points_umap_samp': reduced_points_umap_sample.tolist()
        }
    save_to_s3(
        s3,
        BUCKET_NAME,
        dict_obj,
        f"outputs/skills_extraction/word_embeddings/data/umap_param_experiments/nneig_{umap_n_neighbors}.json",
    )
