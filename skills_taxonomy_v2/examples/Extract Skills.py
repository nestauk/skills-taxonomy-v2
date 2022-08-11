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

# %%
# cd ../..

# %%
import json

import pandas as pd
import matplotlib.pyplot as plt

# %%
from skills_taxonomy_v2.examples.extract_skills import (
    load_prerequisites, split_skills, predict_skill_sents, reduce_embeddings, get_skill_name, cluster_embeddings
)
from skills_taxonomy_v2.pipeline.skills_extraction.get_sentence_embeddings import get_embeddings

# %%
sent_classifier_model_dir = 'outputs/sentence_classifier/models/2021_08_16.pkl'
job_adverts_file = 'skills_taxonomy_v2/examples/job_advert_examples.txt'

# Parameters - these will need tweaking depending on your input data
reduction_n_neighbors = 6
reduction_min_dist = 0.0
clustering_eps = 1
clustering_min_samples = 1

# %%
nlp, bert_vectorizer, sent_classifier = load_prerequisites(sent_classifier_model_dir)

# Load your job advert texts; a list of dicts with the keys "full_text" and "job_id"
with open(job_adverts_file) as f:
    job_adverts = json.load(f)

# Run the pipeline to extract skills
all_job_ids, all_sentences = split_skills(job_adverts)
skill_sentences_dict = predict_skill_sents(sent_classifier, all_job_ids, all_sentences)
sentence_embeddings, original_sentences = get_embeddings(skill_sentences_dict, nlp, bert_vectorizer)
sentences_data_df = reduce_embeddings(sentence_embeddings, original_sentences, reduction_n_neighbors, reduction_min_dist)
sentences_clustered = cluster_embeddings(sentences_data_df, clustering_eps, clustering_min_samples)

# %% [markdown]
# ## Look at skills extracted

# %%
skill_name_dict = sentences_clustered.groupby('cluster_number').apply(lambda x: get_skill_name(x)).to_dict()
job_skills_dict = sentences_clustered.groupby('job id')['cluster_number'].unique().to_dict()


for job_advert in job_adverts:
    job_advert["Skills"] = [skill_name_dict[skill_num] for skill_num in job_skills_dict[job_advert['job_id']]]

print(f"There are {len(skill_name_dict)} skills extracted using this data")
for i in range(3):
    print(f'The job advert: \n{job_adverts[i]["full_text"]} \nHas skills: \n{job_adverts[i]["Skills"]}\n')



# %% [markdown]
# ## Plot sentences coloured by which skill they are assigned to

# %%
fig, ax = plt.subplots(figsize=(8,8))
# plot
sentences_clustered.plot.scatter(
    x = 'reduced_points x',
    y = 'reduced_points y',
    c = 'cluster_number',
    cmap = "rainbow",
    colorbar=False,
    ax=ax, s=50, alpha=0.6);
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.axis('off')
# annotate points in axis
for idx, row in sentences_clustered.iterrows():
    ax.annotate(row['original sentence'], (row['reduced_points x'], row['reduced_points y']) )

# %%
