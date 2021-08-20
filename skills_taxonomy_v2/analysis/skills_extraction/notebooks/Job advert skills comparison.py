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
# ## Compare with ESCO per job advert
# For a sample of job adverts we have:
# - Extracted ESCO skills using Karlis' algorithm
# - Extracted TK skills
#
# Use new merged skills to label these.
#
# - How many unique ESCO skills in this are there?
# - and what is the overlap?
# - Do we want to use a probability threshold for Karlis' algorithm?

# %%
# cd ../../../..

# %%
from ast import literal_eval
from collections import defaultdict
import json

from tqdm import tqdm
import boto3
import pandas as pd

from skills_taxonomy_v2.getters.s3_data import get_s3_data_paths, load_s3_data

# %%
with open("outputs/skills_extraction/data/esco_ID2skill.json", "r") as file:
    esco_ID2skill = json.load(file)
esco_skill2ID = {v: k for k, v in esco_ID2skill.items()}

# %%
joined_skills = pd.read_csv("outputs/skills_extraction/data/merged_skills.csv")

# %%
len(joined_skills)

# %%
joined_skills.loc[0]

# %% [markdown]
# ## Get maps from TK/ESCO ids to canonical skills

# %%
tk2skill = {}
esco2skill = {}
for skill_id, skill_details in joined_skills.iterrows():
    tk_ids = skill_details["TK id"]  # These are sometimes a list!
    esco_id = skill_details["ESCO id"]
    if not pd.isna(tk_ids):
        if type(tk_ids) == str:
            tk_ids = literal_eval(tk_ids)
            if not type(tk_ids) == list:
                tk_ids = [tk_ids]
        for tk_id in tk_ids:
            tk2skill[tk_id] = skill_id
    if not pd.isna(esco_id):
        esco2skill[int(esco_id)] = skill_id

# %% [markdown]
# ## Get lists of skills in each job advert looked at
# Create a dictionary called 'job2skill' which is for every job id the list of skills within them - using the canonical skill IDs which link to skills described in `joined_skills`.

# %% [markdown]
# ### Job advert - skills from cluster approach

# %%
clustered_data = pd.read_csv("outputs/skills_extraction/data/clustered_data.csv")
clustered_data.head(2)

# %%
# Add the cluster skills to a dictionary, use the canonical skills id
job2skill = dict(
    clustered_data.groupby("job id")["Cluster number"].apply(
        lambda x: [tk2skill[tk_id] for tk_id in x if not tk_id == -1]
    )
)
job2skill["0010de901b164efaa05c4cf57c5d3a77"]

# %% [markdown]
# ### Job advert - ESCO skills

# %%
bucket_name = "skills-taxonomy-v2"
s3 = boto3.resource("s3")
esco_skills_output_dir = "outputs/esco_skills/data"

# %%
esco_skills_dir = get_s3_data_paths(
    s3, bucket_name, esco_skills_output_dir, file_types=["*.json"]
)
len(esco_skills_dir)

# %%
# Takes a while!
# Get extracted ESCO skills from the same job adverts
# we took a sample of the TK don't forget

# There may be new ESCO skills not in my dict, add them if they occur
extra_esco = []
for esco_dir in tqdm(esco_skills_dir):
    esco_extracted_skills = load_s3_data(s3, bucket_name, esco_dir)
    for job_id, skills in esco_extracted_skills.items():
        if job_id in job2skill.keys():
            if skills:
                for skill in skills:
                    esco_skill_id = esco_skill2ID.get(skill["preferred_label"])
                    if esco_skill_id:
                        skill_id = esco2skill[int(esco_skill_id)]
                        if job_id in job2skill:
                            job2skill[job_id].append(skill_id)
                        else:
                            job2skill[job_id] = [skill_id]
                    else:
                        extra_esco.append(skill["preferred_label"])

# %%
len(job2skill)

# %%
len(extra_esco)

# %%
# These skills found using Karlis' algorithm aren't included - must be from an old version of
# the ESCO skills?
set(extra_esco)

# %% [markdown]
# ## Save

# %%
with open("outputs/skills_taxonomy/data/job_skills.json", "w") as file:
    json.dump(job2skill, file)

# %% [markdown]
# ## See which skills are in jobs
# - How many
# - Most common

# %%
job_ix = 30
one_job_skills = [
    joined_skills.iloc[skill_id]["Name"]
    for skill_id in list(job2skill.values())[job_ix]
]
print(one_job_skills)

# %%
import matplotlib.pyplot as plt

# %%
num_skills = [len(skills) for skills in job2skill.values()]
plt.hist(num_skills, 50, facecolor="g", alpha=0.75)
plt.title(f"Number of skills found in each of {len(job2skill)} job adverts")

# %%
all_skills = [
    joined_skills.iloc[skill_id]["Name"]
    for skills in job2skill.values()
    for skill_id in skills
]

# %%
len(all_skills)

# %%
from collections import Counter

# %%
Counter(all_skills).most_common(10)

# %%
Counter(all_skills).most_common()[-10:]

# %%
