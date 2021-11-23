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
# ## Analysis of multi-skill sentence occurences
# After taking a sample of sentences (in sentence_data_sample.py) and manually tagging whether the sentences contained multiple skills or not, and whether the sentence was well split, here we can see which sentence length bounds are correlated with multi-skill sentences.
#
# ### Findings
# We took a stratified sample of 240 sentences from different length bounds. We manually tagged sentences as being well split, and whether they contain one skill or multiple skills. This way we can see which sentence length threshold gives a good proportion which are well split and contain only one skill. 
#
# **By chosing a threshold of 250 characters**, we find that sentences shorter than this tend to be well split and as well as possible only contain one skill. To improve whether they contain one skill or not would need more information than simply sentence length - but at least 70% of sentences contains one skill when the sentence length is less than 250 characters.
#
# From `data_reduction_param_exploration.py` (all 512 files of data):
# - Total sentences with embeddings: 16,993,065
# - Number sentences in sample prefiltered: 1,029,659
# - Number sentences in sample after filter (<250 chars and no repeats): 742,771
# - Proportion of sentences filtered through 0.72
#
# ### Findings elsewhere
# In a sample of 10 data files with 337245 sentences, we see 80% of these are less than 250 characters. Thus, we aren't getting rid of too much data and also can reasonably assume each sentence contains one skill and is well split.
#

# %%
import pandas as pd
import matplotlib.pyplot as plt
import boto3

from skills_taxonomy_v2.getters.s3_data import load_s3_data
from skills_taxonomy_v2 import BUCKET_NAME

# %%
s3 = boto3.resource("s3")

# %%
tagged_sentences_data = pd.read_csv("../../../../outputs/skills_extraction/tagged_data/sentences_data_sample_8_nov_2021_tagged.csv")

# %%
len(tagged_sentences_data)

# %%
pd.notnull(tagged_sentences_data["Well split and formatted into single sentence?"][27])


# %%
def get_binary():
    return lambda x: None if pd.isnull(x) else int(x)

tagged_sentences_data['Well split binary'] = tagged_sentences_data["Well split and formatted into single sentence?"].apply(
    get_binary())
tagged_sentences_data['One skill binary'] = tagged_sentences_data["One skill mentioned? (or at least very similar skills)"].apply(
    get_binary())
print(len(tagged_sentences_data))
tagged_sentences_data.head(3)

# %%
tagged_sentences_data[["Well split and formatted into single sentence?", "One skill mentioned? (or at least very similar skills)"]].value_counts()

# %%
tagged_sentences_data.plot.scatter(x="length original", y = 'Well split binary')

# %%
tagged_sentences_data.plot.scatter(x="length original", y = 'One skill binary')

# %%
tagged_sentences_data.groupby("One skill mentioned? (or at least very similar skills)")['length original'].plot(
    kind='hist', bins=15, alpha =0.5, legend=True)

# %%
tagged_sentences_data.groupby('Well split and formatted into single sentence?')['length original'].plot(
    kind='hist', bins=15, alpha =0.5, legend=True)

# %%
tagged_sentences_data['length original'].min()

# %%
one_skill_accuracy = []
well_split_accuracy = []
threshs = []
num_data = []
for sent_thresh in range(50,1000):
    
    filt = tagged_sentences_data[tagged_sentences_data['length original']<sent_thresh]
    if len(filt) != 0:
        threshs.append(sent_thresh)
        one_skill_accuracy.append(filt["One skill binary"].sum()/filt["One skill binary"].notnull().sum())
        well_split_accuracy.append(filt["Well split binary"].sum()/filt["Well split binary"].notnull().sum())
        num_data.append(len(filt))
    

# %%
fig, ax1 = plt.subplots(figsize=(8,4))

ax1.plot(threshs, one_skill_accuracy, label="Proportion with one skill")
ax1.plot(threshs, well_split_accuracy, label="Proportion well split")
ax1.legend(loc="center right")
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.plot(threshs, num_data, c="black", label="Number of data points")
ax2.legend(loc="upper right")

fig.tight_layout()  # otherwise the right y-label is slightly clipped


plt.axvline(x=250, c="red")
plt.show()

# %%
tagged_sentences_data.boxplot(by=[
    "One skill mentioned? (or at least very similar skills)",
    "Well split and formatted into single sentence?"
], column="length original")

# %% [markdown]
# ### Number filtered
# - Going through all 516 files of sentences and their embeddings
# - Take a random sample of 2000 sentences from each
# - Not including sentence with <250 characters long
# - Not including repeated sentences

# %%
n_in_sample_each_file = load_s3_data(
    s3,
    BUCKET_NAME,
    "outputs/skills_extraction/word_embeddings/data/2021.11.05_n_in_sample_each_file.json"
)
n_all_each_file = load_s3_data(
    s3,
    BUCKET_NAME,
    "outputs/skills_extraction/word_embeddings/data/2021.11.05_n_all_each_file.json"
)

# %%
len(n_in_sample_each_file)

# %%
len(n_all_each_file)

# %%
file_name = 'outputs/skills_extraction/word_embeddings/data/2021.11.05/historical/2019/2019-11-14/jobs.0_2021.08.16_embeddings.json'
print(n_all_each_file[file_name])
print(n_in_sample_each_file[file_name])

# %%
num_prefilter_sampled = [min(v, 2000) for k,v in n_all_each_file.items()]
print(f"Total sentences with embeddings: {sum(n_all_each_file.values())}")
print(f"Number sentences in sample prefiltered: {sum(num_prefilter_sampled)}")
print(f"Number sentences in sample after filter (<250 chars and no repeats): {sum(n_in_sample_each_file.values())}")
print(f"Proportion of sentences filtered through {round(sum(n_in_sample_each_file.values())/sum(num_prefilter_sampled),2)}")

# %%
