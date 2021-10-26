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
# ## Examining the sample of textkernel data used from the extension work from 25th October 2021.
#
# - Not all the job ids have dates given in the date metadata (it may not even be a key, or may have a value of None).
# i.e. there are 62892486 job adverts, but only 50566709 keys in the dates metadata
#
#

# %%
from skills_taxonomy_v2.getters.s3_data import (
    load_s3_data,
    get_s3_data_paths,
    save_to_s3,
)

from collections import Counter, defaultdict
import random
import os

from datetime import datetime
from tqdm import tqdm
import pandas as pd
import boto3
import matplotlib.pyplot as plt

bucket_name = "skills-taxonomy-v2"
s3 = boto3.resource("s3")

# %% [markdown]
# ### All the TK data with dates

# %%
tk_dates = {}
for file_name in tqdm(range(0, 13)):
    file_date_dict = load_s3_data(
        s3, bucket_name, f"outputs/tk_data_analysis/metadata_date/{file_name}.json"
    )
    tk_dates.update({k: f[0] for k, f in file_date_dict.items()})

print(len(tk_dates))

# %%
job_ads_date_count = defaultdict(int)

for k, v in tqdm(tk_dates.items()):
    if v:
        date = v[0:7]
        job_ads_date_count[date] += 1
    else:
        job_ads_date_count["No date given"] += 1

# %%
sum(job_ads_date_count.values()) == total_n_job_adverts

# %% [markdown]
# ### Dates for the sample

# %%
sample_dict = load_s3_data(
    s3, bucket_name, "outputs/tk_sample_data/sample_file_locations.json"
)

# %%
sample_dict["historical/2020/2020-03-11/jobs_0.0.jsonl.gz"][0:10]

# %%
sum([len(v) for v in sample_dict.values()])

# %%
job_ads_date_count_sample = defaultdict(int)
for job_id_list in tqdm(sample_dict.values()):
    for job_id in job_id_list:
        v = tk_dates.get(job_id)
        if v:
            date = v[0:7]
            job_ads_date_count_sample[date] += 1
        else:
            job_ads_date_count_sample["No date given"] += 1

# %%
sum(job_ads_date_count_sample.values())


# %% [markdown]
# ### Plot proportions together

# %%
def find_num_dates(count_dict):
    num_dates = {
        int(k.split("-")[0]) + int(k.split("-")[1]) / 12: v
        for k, v in count_dict.items()
        if k != "No date given"
    }
    num_dates[2014] = count_dict["No date given"]
    return num_dates


# %%
num_dates = find_num_dates(job_ads_date_count)
num_dates_sample = find_num_dates(job_ads_date_count_sample)

# %%
plt.figure(figsize=(10, 4))
plt.bar(
    num_dates.keys(),
    [v / sum(num_dates.values()) for v in num_dates.values()],
    width=0.1,
    alpha=0.5,
    label="All data",
)
plt.bar(
    num_dates_sample.keys(),
    [v / sum(num_dates_sample.values()) for v in num_dates_sample.values()],
    width=0.1,
    color="red",
    alpha=0.5,
    label="Sample of data",
)
plt.legend()
plt.xlabel("Date of job advert (2014 = no date given)")
plt.ylabel("Proportion")

# %%
