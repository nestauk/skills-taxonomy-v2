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
# cd ../../../..

# %%
import json

import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
from skills_taxonomy_v2.pipeline.sentence_classifier.predict_sentence_class import *

# %%
input_dir = "inputs/data/"
model_config_name = "2021.08.16"
data_dir = "textkernel-files/"

# %%
sent_classifier, _ = load_model(model_config_name)
nlp = spacy.load("en_core_web_sm")

# %%
root = os.path.join(input_dir, data_dir)

s3 = boto3.resource("s3")
bucket = s3.Bucket(BUCKET_NAME)
data_paths = get_s3_data_paths(bucket, root)

# %% [markdown]
# ### Load one file of data

# %%
data_path = data_paths[0]
data = load_s3_data(data_path, s3)

# %%
data_5000 = data[0:5000]

# %%
with Pool(4) as pool:  # 4 cpus
    partial_split_sentence = partial(split_sentence, nlp=nlp, min_length=30)
    split_sentence_pool_output = pool.map(partial_split_sentence, data_5000)

# %%
# Process output into one list of sentences for all documents
sentences = []
job_ids = []
for i, (job_id, s) in enumerate(split_sentence_pool_output):
    if s:
        sentences += s
        job_ids += [job_id] * len(s)

# %%
print(f"There were {len(sentences)} sentences in {len(data_5000)} job adverts")
print(f"This is about {len(sentences)/len(data_5000)} sentences in each job advert")

# %%
print(
    f"So in a sample of 1 mil job adverts we'd expect {(len(sentences)/len(data_5000))*1000000} sentences"
)
print(f"And since we found 4 million skill sentences in this sample")
print(
    f"it means {round(4000000*100/((len(sentences)/len(data_5000))*1000000),1)}% of sentences are skill sentences"
)

# %% [markdown]
# ### Predict skill sentences for a small number of job adverts

# %%
data_path = data_paths[0]
data = load_s3_data(data_path, s3)

# %%
# Only use a small sample
data = data[0:100]

# %%
with Pool(4) as pool:  # 4 cpus
    partial_split_sentence = partial(split_sentence, nlp=nlp, min_length=30)
    split_sentence_pool_output = pool.map(partial_split_sentence, data)

# %%
# Process output into one list of sentences for all documents
sentences = []
job_ids = []
for i, (job_id, s) in enumerate(split_sentence_pool_output):
    if s:
        sentences += s
        job_ids += [job_id] * len(s)

# %%
sentences_vec = sent_classifier.transform(sentences)
pool_sentences_vec = [(vec_ix, [vec]) for vec_ix, vec in enumerate(sentences_vec)]

# Manually chunk up the data to predict multiple in a pool
# This is because predict can't deal with massive vectors
pool_sentences_vecs = []
pool_sentences_vec = []
for vec_ix, vec in enumerate(sentences_vec):
    pool_sentences_vec.append((vec_ix, vec))
    if len(pool_sentences_vec) > 1000:
        pool_sentences_vecs.append(pool_sentences_vec)
        pool_sentences_vec = []
if len(pool_sentences_vec) != 0:
    # Add the final chunk if not empty
    pool_sentences_vecs.append(pool_sentences_vec)


# %%
with Pool(4) as pool:  # 4 cpus
    partial_predict_sentences = partial(
        predict_sentences, sent_classifier=sent_classifier
    )
    predict_sentences_pool_output = pool.map(
        partial_predict_sentences, pool_sentences_vecs
    )

# %%
skill_sentences_dict = defaultdict(list)
for chunk_output in predict_sentences_pool_output:
    for (sent_ix, pred) in chunk_output:
        if pred == 1:
            job_id = job_ids[sent_ix]
            sentence = sentences[sent_ix]
            skill_sentences_dict[job_id].append(sentence)

# %%
num_skill_sent_job = [len(s) for s in skill_sentences_dict.values()]

# %%
print(f"From a sample of {len(data)} job adverts")
print(f"There were {len(sentences)} sentences found")
print(f"{len(skill_sentences_dict)} job adverts had skill sentences in")
print(f"There were {sum(num_skill_sent_job)} skill sentences found")
print(
    f"Each job had a mean number of {round(np.mean(num_skill_sent_job),1)} skill sentences in"
)
print(f"{sum(num_skill_sent_job)*100/len(sentences)}% of sentences are skill sentences")


# %% [markdown]
# ## Older code

# %%
def load_s3_json_data(file_name, s3, bucket_name):
    obj = s3.Object(bucket_name, file_name)
    file = obj.get()["Body"].read().decode()
    data = json.loads(file)
    return data


# %%
bucket_name = "skills-taxonomy-v2"
s3 = boto3.resource("s3")
bucket = s3.Bucket(bucket_name)

# %%
output_dir = "outputs/sentence_classifier/data/skill_sentences/textkernel-files/"
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
        "num_job_ad_skills": len(data),
        "num_job_ad_no_skills": 100000 - len(data),
        "total_number_sentences": total_number_sentences,
        "average_sentence_len": np.mean(len_sentences),
    }


# %%
len(all_len_sentences)

# %%
data[list(data.keys())[1]]

# %%
pd.DataFrame(data_path_metrics).T.round()

# %%
plt.hist(all_len_sentences, 10, density=True, facecolor="g", alpha=0.75)


# %%
