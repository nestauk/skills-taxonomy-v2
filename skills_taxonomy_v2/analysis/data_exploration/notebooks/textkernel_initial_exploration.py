# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ### Import libraries

# %%
import pandas as pd
import json
import os
import gzip
import shutil


# %% [markdown]
# ### Functions

# %%
# check if file is gz


def is_gz_file(file):

    return file.endswith(".gz")


# check if directory exists


def path_exists(path):

    return os.path.exists(path)


# extract gz files and save in new directory


def extract_gz_files(input_dir, output_dir):

    for fname in os.listdir(input_dir):
        if is_gz_file(fname):
            filepath = os.path.join(input_dir, fname)
            destpath = os.path.join(output_dir, fname[:-3])
            with gzip.open(filepath, "rb") as f_in:
                with open(destpath, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)


# %% [markdown]
# ### Loop through files and unzip
#
# loop through files in data/2021-04-01

# %%
DATA_PATH = "../../../../../skills_taxonomy_v2/"

# %%
# extract gz files from 2021-04-01

output_dir = os.path.join(f"{DATA_PATH}inputs/data/", "unzip_2021-04-01")
if path_exists(output_dir) != True:
    os.mkdir(output_dir)

input_dir = f"{DATA_PATH}inputs/data/2021-04-01"


# %%
extract_gz_files(input_dir, output_dir)

# %% [markdown]
# ### Read extrated jsonl files

# %%
jobs = []
with open(os.path.join(output_dir, "jobs_new.1.jsonl")) as f:
    for line in f:
        jobs.append(json.loads(line))

# %%
jobs_df = pd.DataFrame(jobs)

# %%
jobs_df.columns

# %%
jobs_df.head()

# %%
jobs_df = jobs_df[["job_id", "full_text"]]
jobs_df.rename(columns={"job_id": "id"}, inplace=True)
jobs_df.rename(columns={"full_text": "description"}, inplace=True)

# %%
jobs_df_sample = jobs_df.head(1000)

# %%
csv_output_dir = os.path.join(f"{DATA_PATH}inputs/data/", "csv_files")
if path_exists(csv_output_dir) != True:
    os.mkdir(csv_output_dir)

jobs_df_sample.to_csv(os.path.join(csv_output_dir, "jobs_new.1_sample.csv"), sep=",")

# %%
jobs_df_sample

# %%
