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
# ## Analysing the output of the sentence classifications.
# The outputs of running:
# ```
# python skills_taxonomy_v2/pipeline/sentence_classifier/predict_sentence_class.py --config_path 'skills_taxonomy_v2/config/predict_skill_sentences/2021.07.19.yaml'
# ```
#
# 1. How many skill sentences now? 5,823,903
# 2. Distribution of length of them?

# %%
import json

import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
# cd ../../../..

# %%
from skills_taxonomy_v2.pipeline.sentence_classifier.predict_sentence_class import get_s3_data_paths, load_neccessary


# %%
def load_s3_json_data(file_name, s3, bucket_name):
    obj = s3.Object(bucket_name, file_name)
    file = obj.get()['Body'].read().decode()
    data = json.loads(file)
    return data


# %%
bucket_name = "skills-taxonomy-v2"
s3 = boto3.resource("s3")
bucket = s3.Bucket(bucket_name)

# %%
output_dir = 'outputs/sentence_classifier/data/skill_sentences/textkernel-files/'
data_paths = get_s3_data_paths(bucket, output_dir, pattern="*.json*")
len(data_paths)

# %%
data_path_metrics = {}
all_len_sentences = []
for data_path in data_paths:
    data = load_s3_json_data(data_path, s3, bucket_name)
    
    total_number_sentences = 0
    len_sentences = []
    for job_ad_sentences in data.values():
        total_number_sentences += len(job_ad_sentences)
        len_sentences += [len(sent) for sent in job_ad_sentences]
    all_len_sentences += len_sentences
    data_path_metrics[data_path] = {
        'num_job_ad_skills': len(data),
        'num_job_ad_no_skills': 100000 - len(data),
        'total_number_sentences': total_number_sentences,
        'average_sentence_len': np.mean(len_sentences)
    }


# %%
len(all_len_sentences)

# %%
data[list(data.keys())[1]]

# %%
pd.DataFrame(data_path_metrics).T.round()

# %%
plt.hist(all_len_sentences, 10, density=True, facecolor='g', alpha=0.75);


# %%
