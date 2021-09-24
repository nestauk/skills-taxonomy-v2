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
# Investigate the effect of sample size of skill sentences and how many words are in the vocab.
# This is to justify taking a sample of the data to find skills from.
#
# Ideally want to show that sampling 100,000 sentences (sample size for fitting the reducing embeddings class) would have had a stable vocab by then. At least by 300,000 (clustered on this).

# %%
# cd ../../../..

# %%
import pandas as pd
import boto3
import matplotlib.pyplot as plt
from tqdm import tqdm

from skills_taxonomy_v2.getters.s3_data import get_s3_data_paths, save_to_s3, load_s3_data
from skills_taxonomy_v2.pipeline.skills_extraction.extract_skills_utils import (
    load_sentences_embeddings,
    get_output_config_stamped,
    sample_sentence_embeddings_dirs,
)
from skills_taxonomy_v2 import BUCKET_NAME

# %%
s3 = boto3.resource("s3")

# %%
# sentences_data = load_s3_data(
#     s3,
#     BUCKET_NAME,
#     "outputs/skills_extraction/extracted_skills_sample_50k/2021.08.31_sentences_data.json")

# sentences_data = pd.DataFrame(sentences_data)

# # See how the vocab size changes as you add more sentences.
# # Using the words that went into creating the embeddings.
# sentences_data['description clean'] = sentences_data['description'].apply(
#     lambda x: " ".join(x.split()).split(' '))

# vocab_words = set()
# vocab_size_iteratively = []
# for i, desc_list in tqdm(enumerate(sentences_data['description clean'].tolist()[0:10])):
#     vocab_words = vocab_words.union(set(desc_list))
#     vocab_size_iteratively.append((i, len(vocab_words)))
    
# x = [v[0] for v in vocab_size_iteratively]
# y = [v[1] for v in vocab_size_iteratively]

# plt.plot(x,y);
# plt.xlabel('Number of sentences')
# plt.ylabel('Number of unique words in vocab')

# %% [markdown]
# ## I used to save it out, so can retrieve directly

# %%
num_sentences_and_vocab_size = load_s3_data(
    s3,
    BUCKET_NAME,
    "outputs/skills_extraction/extracted_skills_sample_50k/2021.08.31_num_sentences_and_vocab_size.json")

# %%
x = [v[0] for v in num_sentences_and_vocab_size]
y = [v[1] for v in num_sentences_and_vocab_size]

# %%
nesta_orange = [255/255,90/255,0/255]
plt.plot(x,y, color='black');
plt.axvline(322071, color=nesta_orange, ls='--')
plt.xlabel('Number of sentences')
plt.ylabel('Number of unique words in vocab')
plt.savefig('outputs/skills_extraction/figures/num_sent_vocab_size.pdf',bbox_inches='tight')


# %% [markdown]
# ## Number of job adverts and unique number of skills

# %%
file_name_date = '2021.08.31'
sentence_data = load_s3_data(s3, BUCKET_NAME, f'outputs/skills_extraction/extracted_skills/{file_name_date}_sentences_data.json')


# %%
sentence_data = pd.DataFrame(sentence_data)
sentence_data = sentence_data[sentence_data['Cluster number']!=-1]

# %%
len(sentence_data)

# %%
unique_skills = {}
for k in tqdm(range(0, len(sentence_data))):
    unique_skills[k] = sentence_data.iloc[0:k]['Cluster number'].nunique()

# %%
x = list(unique_skills.keys())
y = list(unique_skills.values())

nesta_orange = [255/255,90/255,0/255]
plt.plot(x,y, color='black');
plt.xlabel('Number of sentences')
plt.ylabel('Number of unique skills')
plt.savefig('outputs/skills_extraction/figures/num_sent_num_skills.pdf',bbox_inches='tight')


# %% [markdown]
# ## Together

# %%
x_vocab = [v[0] for v in num_sentences_and_vocab_size]
y_vocab = [v[1] for v in num_sentences_and_vocab_size]

x_skills = list(unique_skills.keys())
y_skills = list(unique_skills.values())

# %%
fig, axs = plt.subplots(1,2, figsize=(10,3))

axs[0].plot(x_vocab, y_vocab, color='black');
axs[0].axvline(322071, color=nesta_orange, ls='--')
axs[0].set_xlabel('Number of sentences')
axs[0].set_ylabel('Number of unique words in vocab')

axs[1].plot(x_skills, y_skills, color='black');
axs[1].set_xlabel('Number of sentences')
axs[1].set_ylabel('Number of unique skills')

plt.tight_layout()
plt.savefig('outputs/skills_extraction/figures/num_sent_num_skills_vocab_size.pdf',bbox_inches='tight')


# %%
