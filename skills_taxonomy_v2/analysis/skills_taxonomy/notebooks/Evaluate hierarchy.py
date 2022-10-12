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
# ## Evaluate hierarchy based on jobs
# - How many skill groups per job advert
# - Popular skill groups for job titles
#

# %%
# cd ../../../..

# %%
from skills_taxonomy_v2.getters.s3_data import (
    get_s3_data_paths,
    load_s3_data,
)
import json
from itertools import chain, combinations
from tqdm import tqdm
import random
from collections import Counter

import matplotlib.pyplot as plt
import boto3
import pandas as pd
import numpy as np

# %%
from ipywidgets import interact
import bokeh.plotting as bpl
from bokeh.io import (
    output_file,
    show,
    push_notebook,
    output_notebook,
    reset_output,
    save,
)

from bokeh.plotting import (
    figure,
    from_networkx,
    ColumnDataSource,
    output_file,
    show,
    gridplot,
)

from bokeh.models import (
    BoxZoomTool,
    WheelZoomTool,
    HoverTool,
    SaveTool,
    Circle,
    MultiLine,
    Plot,
    Range1d,
    ResetTool,
    Label,
    CategoricalColorMapper,
    ColorBar,
    ColumnDataSource,
    LinearColorMapper,
)

from bokeh.palettes import Turbo256, Spectral, Spectral4, viridis, inferno, Spectral6

from bokeh.resources import CDN
from bokeh.embed import file_html
from bokeh.transform import linear_cmap


bpl.output_notebook()

# %% [markdown]
# ## Load skills sentences and hierarchy information

# %%
bucket_name = "skills-taxonomy-v2"
s3 = boto3.resource("s3")

# %%
hier_date = '2022.01.21'
skills_date = '2022.01.14'

# %%
hier_structure_file = f"outputs/skills_taxonomy/{hier_date}_hierarchy_structure.json"
hier_structure = load_s3_data(s3, bucket_name, hier_structure_file)

# %%
skill_hierarchy_file = f"outputs/skills_taxonomy/{hier_date}_skills_hierarchy_named.json"
skill_hierarchy = load_s3_data(s3, bucket_name, skill_hierarchy_file)

# %%
sentence_data = load_s3_data(
    s3,
    bucket_name,
    f"outputs/skills_extraction/extracted_skills/{skills_date}_sentences_skills_data_lightweight.json",
)

# %%
sentence_data_df = pd.DataFrame(sentence_data, columns=['job id', 'sentence id',  'Cluster number predicted'])
del sentence_data
sentence_data_df.head(2)

# %%
sentence_data_df = sentence_data_df[sentence_data_df["Cluster number predicted"] >= 0]

# %%
mean_num_skills = sentence_data_df.groupby('job id')['Cluster number predicted'].nunique().mean()
print(f"The mean number of skills in each job advert is {mean_num_skills}")


# %% [markdown]
# ### Add duplicate sentence information

# %%
dupe_words_id = load_s3_data(
    s3,
    bucket_name,
    f"outputs/skills_extraction/word_embeddings/data/2022.01.14_unique_words_id_list.json",
)

# %%
# The job ids in the skill sentences which have duplicates
dupe_job_ids = set(sentence_data_df['job id'].tolist()).intersection(set(dupe_words_id.keys()))
# What are the word ids for these?
skill_job_ids_with_dupes_list = [(job_id, sent_id, word_id) for job_id, s_w_list in dupe_words_id.items() for (word_id, sent_id) in s_w_list if job_id in dupe_job_ids]
skill_job_ids_with_dupes_df = pd.DataFrame(skill_job_ids_with_dupes_list, columns = ['job id', 'sentence id', 'words id'])
del skill_job_ids_with_dupes_list
# Get the words id for the existing deduplicated sentence data
sentence_data_ehcd = sentence_data_df.merge(skill_job_ids_with_dupes_df, how='left', on=['job id', 'sentence id'])
del skill_job_ids_with_dupes_df
skill_sent_word_ids = set(sentence_data_ehcd['words id'].unique())
len(skill_sent_word_ids)


# %%
# Get all the job id+sent id for the duplicates with these word ids
dupe_sentence_data = []
for job_id, s_w_list in tqdm(dupe_words_id.items()):
    for (word_id, sent_id) in s_w_list:
        if word_id in skill_sent_word_ids:
            cluster_num = sentence_data_ehcd[sentence_data_ehcd['words id']==word_id].iloc[0]['Cluster number predicted']
            dupe_sentence_data.append([job_id, sent_id, cluster_num])
dupe_sentence_data_df = pd.DataFrame(dupe_sentence_data, columns = ['job id', 'sentence id', 'Cluster number predicted'])           
del dupe_sentence_data
del sentence_data_ehcd


# %%
# Add new duplicates to sentence data
sentence_data_df = pd.concat([sentence_data_df, dupe_sentence_data_df])
sentence_data_df.drop_duplicates(inplace=True)
sentence_data_df.reset_index(inplace=True)

# %%
len(sentence_data_df)

# %%
del dupe_sentence_data_df

# %%
sentence_data_df['job id'].nunique()

# %% [markdown]
# ### Join the sentence data with hierarchy information

# %%
sentence_data_df.head(2)

# %%

sentence_data_df["Hierarchy level A"] = (
    sentence_data_df["Cluster number predicted"]
    .astype(str)
    .apply(lambda x: skill_hierarchy[x]["Hierarchy level A"])
)
sentence_data_df["Hierarchy level A name"] = (
    sentence_data_df["Cluster number predicted"]
    .astype(str)
    .apply(lambda x: skill_hierarchy[x]["Hierarchy level A name"])
)
sentence_data_df["Hierarchy level B"] = (
    sentence_data_df["Cluster number predicted"]
    .astype(str)
    .apply(lambda x: skill_hierarchy[x]["Hierarchy level B"])
)
sentence_data_df["Hierarchy level C"] = (
    sentence_data_df["Cluster number predicted"]
    .astype(str)
    .apply(lambda x: skill_hierarchy[x]["Hierarchy level C"])
)
sentence_data_df["Hierarchy ID"] = (
    sentence_data_df["Cluster number predicted"]
    .astype(str)
    .apply(lambda x: skill_hierarchy[x]["Hierarchy ID"])
)

# %%
sentence_data_df.head(2)

# %%
mean_num_skills = sentence_data_df.groupby('job id')['Cluster number predicted'].nunique().mean()
mean_leva = sentence_data_df.groupby('job id')['Hierarchy level A'].nunique().mean()
mean_levb = sentence_data_df.groupby('job id')['Hierarchy level B'].nunique().mean()
mean_levc = sentence_data_df.groupby('job id')['Hierarchy level C'].nunique().mean()

print(f"The mean number of skills in each job advert is {mean_num_skills}")
print(f"The mean number of level A skill groups in each job advert is {mean_leva}")
print(f"The mean number of level B skill groups in each job advert is {mean_levb}")
print(f"The mean number of level C skill groups in each job advert is {mean_levc}")


# %%
sentence_data_df.groupby('job id')['Cluster number predicted'].nunique().plot.hist(bins=100)

# %%
sentence_data_df.groupby('job id')['Hierarchy level A'].nunique().plot.hist(bins=11)

# %%
sentence_data_df.groupby('job id')['Hierarchy level B'].nunique().plot.hist(bins=10)

# %%
sentence_data_df.groupby('job id')['Hierarchy level C'].nunique().plot.hist(bins=50)

# %% [markdown]
# ## Evaluate
# - Look at skills for individual jobs - each job's skills should be roughly in the same branch of the hierarchy
# - Warning: sometimes there is only one skill in a job id, so this would sounds better than it is

# %%
nesta_orange = [255 / 255, 90 / 255, 0 / 255]

# %%
job_id_levels = sentence_data_df.groupby("job id").agg(
    {
        "sentence id": "count",
        "Hierarchy level A": "nunique",
        "Hierarchy level B": "nunique",
        "Hierarchy level C": "nunique",
    }
)

# %%
job_id_levels["Hierarchy level A normalised"] = (
    job_id_levels["Hierarchy level A"] / job_id_levels["sentence id"]
)
job_id_levels["Hierarchy level B normalised"] = (
    job_id_levels["Hierarchy level B"] / job_id_levels["sentence id"]
)
job_id_levels["Hierarchy level C normalised"] = (
    job_id_levels["Hierarchy level C"] / job_id_levels["sentence id"]
)

# %%
mean_nums_with_filt = []
for min_num_sent in range(5, 40):
    filt_data = job_id_levels[job_id_levels["sentence id"] >= min_num_sent]
    mean_nums_with_filt.append(
        {
            "Minimum number of sentences in job advert": min_num_sent,
            "Level A skills": filt_data["Hierarchy level A"].mean(),
            "Level B skills": filt_data["Hierarchy level B"].mean(),
            "Level C skills": filt_data["Hierarchy level C"].mean(),
        }
    )
mean_nums_with_filt = pd.DataFrame(mean_nums_with_filt)

# %%
num_sent_with_filt = []
for min_num_sent in range(5, 40):
    filt_data = job_id_levels[job_id_levels["sentence id"] >= min_num_sent]
    num_sent_with_filt.append(
        {
            "Minimum number of sentences in job advert": min_num_sent,
            "Number job adverts": len(filt_data),
        }
    )
num_sent_with_filt = pd.DataFrame(num_sent_with_filt)

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

mean_nums_with_filt.plot(
    x="Minimum number of sentences in job advert",
    title="The mean numbers of unique skill groups in job adverts\ncalculated for different minimum numbers of\nsentences for each job advert",
    ylabel="Mean number of unique skill groups",
    color=[nesta_orange, "black", "grey"],
    ax=axes[0],
)
# axes[0].axhline(6, color='orange', linestyle='--')


num_sent_with_filt.plot(
    x="Minimum number of sentences in job advert",
    title="",
    ylabel="Number of job adverts",
    color=nesta_orange,
    ax=axes[1],
)

plt.tight_layout()

plt.savefig(
    f"outputs/skills_taxonomy/figures/{hier_date}/evaluate_mean_nums.pdf",
    bbox_inches="tight",
)


# %% [markdown]
# ## Normalised for when number sentences>=10

# %%
min_num_sent = 10

fig, axes = plt.subplots(1, 3, figsize=(10, 3))

job_id_levels[job_id_levels["sentence id"] >= min_num_sent][
    "Hierarchy level A normalised"
].plot.hist(
    ax=axes[0],
    color=nesta_orange,
    title="Number of unique level A skill\ngroups per job id (normalised)",
)
job_id_levels[job_id_levels["sentence id"] >= min_num_sent][
    "Hierarchy level B normalised"
].plot.hist(
    ax=axes[1],
    color=nesta_orange,
    title="Number of unique level B skill\ngroups per job id (normalised)",
)
job_id_levels[job_id_levels["sentence id"] >= min_num_sent][
    "Hierarchy level C normalised"
].plot.hist(
    ax=axes[2],
    color=nesta_orange,
    title="Number of unique level C skill\ngroups per job id (normalised)",
)

plt.tight_layout()


plt.savefig(
    f"outputs/skills_taxonomy/figures/{hier_date}/evaluate_nums_normalised.pdf",
    bbox_inches="tight",
)


# %% [markdown]
# ## Raw number for when number sentences>=10

# %%
min_num_sent = 10

fig, axes = plt.subplots(1, 3, figsize=(10, 3))

job_id_levels[job_id_levels["sentence id"] >= min_num_sent][
    "Hierarchy level A"
].plot.hist(
    ax=axes[0],
    color=nesta_orange,
    title="Number of unique level A skill\ngroups per job id",
    bins=6,
)
job_id_levels[job_id_levels["sentence id"] >= min_num_sent][
    "Hierarchy level B"
].plot.hist(
    ax=axes[1],
    color=nesta_orange,
    title="Number of unique level B skill\ngroups per job id",
)
job_id_levels[job_id_levels["sentence id"] >= min_num_sent][
    "Hierarchy level C"
].plot.hist(
    ax=axes[2],
    color=nesta_orange,
    title="Number of unique level C skill\ngroups per job id",
)

plt.tight_layout()

plt.savefig(
    f"outputs/skills_taxonomy/figures/{hier_date}/evaluate_nums_raw.pdf",
    bbox_inches="tight",
)


# %% [markdown]
# ## Raw number for when number sentences==10

# %%
len(job_id_levels[job_id_levels["sentence id"] == 10])

# %%
min_num_sent = 6

fig, axes = plt.subplots(1, 3, figsize=(10, 3))

job_id_levels[job_id_levels["sentence id"] == min_num_sent][
    "Hierarchy level A"
].plot.hist(
    ax=axes[0],
    color=nesta_orange,
    title="Number of unique level A skill\ngroups per job id",
    bins=min_num_sent,
)
job_id_levels[job_id_levels["sentence id"] == min_num_sent][
    "Hierarchy level B"
].plot.hist(
    ax=axes[1],
    color=nesta_orange,
    title="Number of unique level B skill\ngroups per job id",
    bins=min_num_sent,
)
job_id_levels[job_id_levels["sentence id"] == min_num_sent][
    "Hierarchy level C"
].plot.hist(
    ax=axes[2],
    color=nesta_orange,
    title="Number of unique level C skill\ngroups per job id",
    bins=min_num_sent,
)

plt.tight_layout()

plt.savefig(
    f"outputs/skills_taxonomy/figures/{hier_date}/evaluate_nums_6.pdf",
    bbox_inches="tight",
)


# %%
job_id_levels["sentence id"].plot.hist(bins=100)

# %% [markdown]
# ## Most common job titles for skills
# October 21 results:
# - There are 65219 unique job titles
# - There are 23 unique organization industries
# - 82.899% skill sentences have job title data
# - 82.969% skill sentences have organization industry data
#
# January 22 results:
#
# - There are 746504 unique job titles
# - There are 23 unique organization industries
# - 99.965% skill sentences have job title data
# - 99.965% skill sentences have organization industry data
#

# %%
1780/65219

# %%
20000/746504

# %%
job_titles = load_s3_data(
    s3,
    bucket_name,
    "outputs/tk_data_analysis_new_method/metadata_job/14.01.22/sample_filtered.json"
)

# %%
len(job_titles)

# %%
# May need adjustment since there are multiple answer for duplicate job ids
# e.g. of one:
# [[['Telephone Sales Executive', 'Staffing / Employment Agencies']],
#  [['Telephone Sales Executive', 'Staffing / Employment Agencies'],
#   [None, None]],
#  [['Telephone Sales Executive', 'Staffing / Employment Agencies']]]
job_titles_one = {}
for job_id, job_ad_jobs in job_titles.items():
    job_title = [k[0] for j in job_ad_jobs for k in j if k[0]]
    org_industry = [k[1] for j in job_ad_jobs for k in j if k[1]]
    if len(job_title) !=0:
        if len(org_industry) !=0:
            job_titles_one[job_id] = [job_title[0], org_industry[0]]
        else:
            job_titles_one[job_id] = [job_title[0], None]

# %%
del job_titles

# %%
sentence_data_df["Job title"] = sentence_data_df["job id"].apply(
    lambda x: job_titles_one.get(x, [None])[0]
)
sentence_data_df["Organization industry"] = sentence_data_df["job id"].apply(
    lambda x: job_titles_one.get(x, [None, None])[1]
)

# %%
next(iter(job_titles_one.values()))

# %%

print(f"There are {sentence_data_df['Job title'].nunique()} unique job titles")
print(
    f"There are {sentence_data_df['Organization industry'].nunique()} unique organization industries"
)
print(
    f"{round(sum(sentence_data_df['Job title'].notnull())*100/len(sentence_data_df),3)}% skill sentences have job title data"
)
print(
    f"{round(sum(sentence_data_df['Organization industry'].notnull())*100/len(sentence_data_df),3)}% skill sentences have organization industry data"
)

# %% [markdown]
# ### Which skill group levels does each job have the highest proportions in

# %%
from tqdm import tqdm

# %%
grouped_jobtitle = sentence_data_df.groupby(["Job title"])

job_skill_leva_props = {}
job_skill_levb_props = {}
job_skill_levc_props = {}
for job_name, job_group in tqdm(grouped_jobtitle):
    ## Level A
    levas = job_group["Hierarchy level A"].value_counts(sort=True)
    leva_props_dict = (levas / sum(levas)).to_dict()
    leva_props_dict["Number of unique job ids"] = job_group["job id"].nunique()
    job_skill_leva_props[job_name] = leva_props_dict

    ## Level B
    levbs = job_group["Hierarchy level B"].value_counts(sort=True)
    levb_props_dict = (levbs / sum(levbs)).to_dict()
    levb_props_dict["Number of unique job ids"] = job_group["job id"].nunique()
    job_skill_levb_props[job_name] = levb_props_dict

    ## Level C
    levcs = job_group["Hierarchy level C"].value_counts(sort=True)
    levc_props_dict = (levcs / sum(levcs)).to_dict()
    levc_props_dict["Number of unique job ids"] = job_group["job id"].nunique()
    job_skill_levc_props[job_name] = levc_props_dict

job_skill_leva_props_df = pd.DataFrame(job_skill_leva_props).T
job_skill_levb_props_df = pd.DataFrame(job_skill_levb_props).T
job_skill_levc_props_df = pd.DataFrame(job_skill_levc_props).T

# %%
sentence_data_df["Job title"].nunique()

# %%
sum(sentence_data_df["Job title"].value_counts() > 10)

# %%
sum(sentence_data_df["Job title"].value_counts() > 100)

# %%
min_num_job_ids = 100

# %% [markdown]
# ### Level A skills

# %%
lev_a_top_jobs = []
for level_a_name, leve_a_group in sentence_data_df.groupby(["Hierarchy level A"]):
    job_titles = leve_a_group["Job title"].value_counts(sort=True)
    org_inds = leve_a_group["Organization industry"].value_counts(sort=True)

    top_job_titles_by_prop = job_skill_leva_props_df[
        job_skill_leva_props_df["Number of unique job ids"] > min_num_job_ids
    ][level_a_name].sort_values(ascending=False)[0:10]
    
    lev_a_top_jobs.append(
        {
            "Level A name": leve_a_group['Hierarchy level A name'].unique()[0],
            "Number of unique skills": leve_a_group["Cluster number predicted"].nunique(),
            "Number of unique job ids": leve_a_group["job id"].nunique(),
            "Number of skill sentences": len(leve_a_group),
            "Top 5 most common skills": list(
                leve_a_group["Cluster number predicted"]
                .apply(lambda x: skill_hierarchy[str(x)]["Skill name"])
                .value_counts()[0:5]
                .index
            ),
            "Top 10 job titles": [
                f"{j} ({n})"
                for j, n in list(zip(job_titles.index[0:10], job_titles[0:10]))
            ],
            "Top 10 organization industries": [
                f"{j} ({n})" for j, n in list(zip(org_inds.index[0:10], org_inds[0:10]))
            ],
            f"Top 10 job titles with >{min_num_job_ids} job adverts, with highest proportion in this level": [
                f"{j} ({round(n*100,2)}%)"
                for j, n in list(
                    zip(top_job_titles_by_prop.index, top_job_titles_by_prop)
                )
            ],
        }
    )

# %%
pd.DataFrame(lev_a_top_jobs).to_csv(
    f"outputs/skills_taxonomy/evaluation/{hier_date}/top_10_jobs_levela_20_over100.csv"
)

# %% [markdown]
# ### Level B

# %%
lev_a_name_dict = {}
lev_b_name_dict = {}
lev_c_name_dict = {}
lev_d_name_dict = {}
for lev_a_id, lev_a in hier_structure.items():
    lev_a_name_dict[lev_a_id] = lev_a["Name"]
    for lev_b_id, lev_b in lev_a["Level B"].items():
        lev_b_name_dict[lev_b_id] = lev_b["Name"]
        for lev_c_id, lev_c in lev_b["Level C"].items():
            lev_c_name_dict[lev_c_id] = lev_c["Name"]
del hier_structure

# %%
lev_b_top_jobs = []
for level_b_name, leve_b_group in sentence_data_df.groupby(["Hierarchy level B"]):
    job_titles = leve_b_group["Job title"].value_counts(sort=True)
    org_inds = leve_b_group["Organization industry"].value_counts(sort=True)

    top_job_titles_by_prop = job_skill_levb_props_df[
        job_skill_levb_props_df["Number of unique job ids"] > min_num_job_ids
    ][level_b_name].sort_values(ascending=False)[0:10]

    lev_b_top_jobs.append(
        {
            "Level B name": lev_b_name_dict[str(level_b_name)],
            "Number of unique skills": leve_b_group["Cluster number predicted"].nunique(),
            "Number of unique job ids": leve_b_group["job id"].nunique(),
            "Number of skill sentences": len(leve_b_group),
            "Top 5 most common skills": list(
                leve_b_group["Cluster number predicted"]
                .apply(lambda x: skill_hierarchy[str(x)]["Skill name"])
                .value_counts()[0:5]
                .index
            ),
            "Top 10 job titles": [
                f"{j} ({n})"
                for j, n in list(zip(job_titles.index[0:10], job_titles[0:10]))
            ],
            "Top 10 organization industries": [
                f"{j} ({n})" for j, n in list(zip(org_inds.index[0:10], org_inds[0:10]))
            ],
            f"Top 10 job titles with >{min_num_job_ids} job adverts, with highest proportion in this level": [
                f"{j} ({round(n*100,2)}%)"
                for j, n in list(
                    zip(top_job_titles_by_prop.index, top_job_titles_by_prop)
                )
            ],
        }
    )

# %%
pd.DataFrame(lev_b_top_jobs).to_csv(
    f"outputs/skills_taxonomy/evaluation/{hier_date}/top_10_jobs_levelb.csv"
)

# %% [markdown]
# ### Level C
# Too many - so just the most common?

# %%
lev_c_top_jobs = []
for level_c_name, leve_c_group in sentence_data_df.groupby(["Hierarchy level C"]):
    job_titles = leve_c_group["Job title"].value_counts(sort=True)
    org_inds = leve_c_group["Organization industry"].value_counts(sort=True)

    top_job_titles_by_prop = job_skill_levc_props_df[
        job_skill_levc_props_df["Number of unique job ids"] > min_num_job_ids
    ][level_c_name].sort_values(ascending=False)[0:10]

    lev_c_top_jobs.append(
        {
            "Level C name": lev_c_name_dict[str(level_c_name)],
            "Number of unique skills": leve_c_group["Cluster number predicted"].nunique(),
            "Number of unique job ids": leve_c_group["job id"].nunique(),
            "Number of skill sentences": len(leve_c_group),
            "Top 5 most common skills": list(
                leve_c_group["Cluster number predicted"]
                .apply(lambda x: skill_hierarchy[str(x)]["Skill name"])
                .value_counts()[0:5]
                .index
            ),
            "Top 10 job titles": [
                f"{j} ({n})"
                for j, n in list(zip(job_titles.index[0:10], job_titles[0:10]))
            ],
            "Top 10 organization industries": [
                f"{j} ({n})" for j, n in list(zip(org_inds.index[0:10], org_inds[0:10]))
            ],
            f"Top 10 job titles with >{min_num_job_ids} job adverts, with highest proportion in this level": [
                f"{j} ({round(n*100,2)}%)"
                for j, n in list(
                    zip(top_job_titles_by_prop.index, top_job_titles_by_prop)
                )
            ],
        }
    )

# %%
pd.DataFrame(lev_c_top_jobs).to_csv(
    f"outputs/skills_taxonomy/evaluation/{hier_date}/top_10_jobs_levelc.csv"
)

# %% [markdown]
# ## For each job, most common skill level C

# %%
top_jobs = sentence_data_df["Job title"].value_counts()[0:100].index.tolist()

# %%
len(top_jobs)

# %%
sentence_data_df["Skill name"] = sentence_data_df["Cluster number predicted"].apply(
    lambda x: skill_hierarchy[str(x)]["Skill name"]
)
sentence_data_df["Hierarchy level B name"] = sentence_data_df["Hierarchy level B"].apply(
    lambda x: lev_b_name_dict[str(x)]
)
sentence_data_df["Hierarchy level C name"] = sentence_data_df["Hierarchy level C"].apply(
    lambda x: lev_c_name_dict[str(x)]
)

# %%
per_top_job_levels = []
for job_title in top_jobs:
    job_df = sentence_data_df[sentence_data_df["Job title"] == job_title]
    levas = round(
        job_df["Hierarchy level A name"].value_counts(sort=True) * 100 / len(job_df)
    )
    levbs = round(
        job_df["Hierarchy level B name"].value_counts(sort=True) * 100 / len(job_df)
    )
    levcs = round(
        job_df["Hierarchy level C name"].value_counts(sort=True) * 100 / len(job_df)
    )
    skills_d = round(job_df["Skill name"].value_counts(sort=True) * 100 / len(job_df))
    per_top_job_levels.append(
        {
            "Job title": job_title,
            "Number of skill sentences": len(job_df),
            "Most common level A skill groups": [
                f"{j} ({round(n)}%)" for j, n in list(zip(levas.index[0:1], levas[0:1]))
            ],
            "5 most common level B skill groups": [
                f"{j} ({round(n)}%)" for j, n in list(zip(levbs.index[0:5], levbs[0:5]))
            ],
            "5 most common level C skill groups": [
                f"{j} ({round(n)}%)" for j, n in list(zip(levcs.index[0:5], levcs[0:5]))
            ],
            "5 most common skills": [
                f"{j} ({round(n)}%)" for j, n in list(zip(skills_d.index[0:10], skills_d[0:5]))
            ],
        }
    )


# %%
pd.DataFrame(per_top_job_levels).to_csv(
    f"outputs/skills_taxonomy/evaluation/{hier_date}/jobs_top_skill_groups.csv"
)

# %%
sentence_data_df[sentence_data_df['Job title']=='Member Pioneer']['job id'].nunique()


# %% [markdown]
# ## Breakdown pie charts of a few jobs

# %%
def get_job_title_info(job_title):
    job_df = sentence_data_df[sentence_data_df["Job title"] == job_title]
    levas = round(
        job_df["Hierarchy level A name"].value_counts(sort=True) * 100 / len(job_df)
    )
    levbs = round(
        job_df["Hierarchy level B name"].value_counts(sort=True) * 100 / len(job_df)
    )
    levcs = round(
        job_df["Hierarchy level C name"].value_counts(sort=True) * 100 / len(job_df)
    )
    skill_s = round(
        job_df["Skill name"].value_counts(sort=True) * 100 / len(job_df)
    )
    
    print(levas)
    print(levbs[0:5])
    print(levcs[0:5])
    print(skill_s[0:5])


# %%
get_job_title_info('Cleaner')

# %%
get_job_title_info('Accountant')

# %%
get_job_title_info('Data Scientist')

# %%
get_job_title_info('Data Analyst')

# %%
(sentence_data_df["Hierarchy level A name"].value_counts(sort=True)* 100 / len(sentence_data_df))[0:10]

# %%
(sentence_data_df["Hierarchy level B name"].value_counts(sort=True)* 100 / len(sentence_data_df))[0:10]

# %%
(sentence_data_df["Hierarchy level C name"].value_counts(sort=True)* 100 / len(sentence_data_df))[0:10]

# %% [markdown]
# ## Common job types in 5 million data and in sample used to extract skills from
# Is it a similar distribution?
# We don't have the data in a handy form from all TK data.

# %%
count_job_titles_all = Counter([j[0] for j in job_titles_one.values()])

# %%
skills_job_ids = set(sentence_data_df['job id'].unique().tolist())
count_job_titles_skills = Counter([j[0] for k,j in job_titles_one.items() if k in skills_job_ids])


# %%
len(skills_job_ids)

# %%
skill_job_titles = count_job_titles_skills.keys()
len(skill_job_titles)

# %%
print(len(count_job_titles_all))
print(len(count_job_titles_skills))

# %%
num_job_adverts_all = len(job_titles_one)
num_job_adverts_skills = len(skills_job_ids)

allsample_vs_skills_jobtitles = pd.DataFrame({
    'Job title': skill_job_titles,
    'Proportion in all sample': [count_job_titles_all[s]/num_job_adverts_all for s in skill_job_titles],
    'Proportion in skills sample': [count_job_titles_skills[s]/num_job_adverts_skills for s in skill_job_titles],
    'Number in all sample': [count_job_titles_all[s] for s in skill_job_titles],
    'Number in skills sample': [count_job_titles_skills[s] for s in skill_job_titles]
})

# %%
allsample_vs_skills_jobtitles.head(2)

# %%
high_props = allsample_vs_skills_jobtitles[allsample_vs_skills_jobtitles['Proportion in skills sample']>0.0001]
print(len(high_props))
plot_samp = high_props.sample(min(len(high_props),50), random_state=42)

plot_samp.plot.bar(
    x='Job title',
    y=['Proportion in all sample','Proportion in skills sample'],
    color=["red","blue"], figsize=(15,3)
)

# %%
sort_all = allsample_vs_skills_jobtitles.sort_values(by='Proportion in all sample', ascending=False)

# %%
sort_all[0:50].plot.bar(
    x='Job title',
    y=['Proportion in all sample','Proportion in skills sample'],
    color=["red","blue"], figsize=(15,3)
)

# %%
# At what point do jobs account for 10% of the entire sample?
for i in range(0,200):
    num_jobs = sort_all[0:i]['Number in all sample'].sum()
    if num_jobs > 0.1*num_job_adverts_all:
        print(i)
        break
top_all_jobs = set(sort_all[0:i]['Job title'].tolist())
print(len(top_all_jobs))

# %%
sort_skills = allsample_vs_skills_jobtitles.sort_values(by='Proportion in skills sample', ascending=False)
# At what point do jobs account for 10% of the skills sample?
for i in range(0,200):
    num_jobs = sort_skills[0:i]['Number in skills sample'].sum()
    if num_jobs > 0.1*num_job_adverts_skills:
        print(i)
        break
top_skills_jobs = set(sort_skills[0:i]['Job title'].tolist())
print(len(top_skills_jobs))

# %%
print(len(top_all_jobs.intersection(top_skills_jobs)))
print(len(top_skills_jobs.difference(top_all_jobs)))
print(len(top_all_jobs.difference(top_skills_jobs)))


# %%
def print_int_len(i):
    print(len(set(sort_all[0:i]['Job title'].tolist()).intersection(sort_skills[0:i]['Job title'].tolist()))/i)
print_int_len(5)
print_int_len(10)
print_int_len(15)
print_int_len(20)
print_int_len(50)
print_int_len(100)
print_int_len(200)

# %%
i=20
print(set(sort_all[0:i]['Job title'].tolist()).intersection(sort_skills[0:i]['Job title'].tolist()))
print(set(sort_all[0:i]['Job title'].tolist()).difference(sort_skills[0:i]['Job title'].tolist()))
print(set(sort_skills[0:i]['Job title'].tolist()).difference(sort_all[0:i]['Job title'].tolist()))

# %%
top_intersection = {}
for i in range(1,1000):
    top_intersection[i] = len(set(sort_all[0:i]['Job title'].tolist()).intersection(sort_skills[0:i]['Job title'].tolist()))

# %%
plt.plot(top_intersection.keys(),[v/i for i,v in top_intersection.items()])

# %% [markdown]
# ## By job area

# %%
count_job_area_all = Counter([j[1] for j in job_titles_one.values()])

# %%
count_job_area_skills = Counter([j[1] for k,j in job_titles_one.items() if k in skills_job_ids])


# %%
skill_job_area = count_job_area_skills.keys()
len(skill_job_area)

# %%
allsample_vs_skills_jobareas = pd.DataFrame({
    'Organization industry': skill_job_area,
    f'Proportion in all sample': [count_job_area_all[s]/num_job_adverts_all for s in skill_job_area],
    f'Proportion in skills sample': [count_job_area_skills[s]/num_job_adverts_skills for s in skill_job_area]
})

# %%
plt.figure()
allsample_vs_skills_jobareas.plot.bar(
    x='Organization industry',
    y=['Proportion in all sample','Proportion in skills sample'],
    color=["red","blue"],
    figsize=(10, 3),
    label = [
        f'Proportion in random sample (n={num_job_adverts_all})',
        f'Proportion in skills sample (n={num_job_adverts_skills})'],
    title='Job advert organisation industries from random sample of job adverts\n and those adverts from the sample which were used to extract skills from'
);
plt.savefig(
    f"outputs/skills_taxonomy/evaluation/{hier_date}/job_org_industry_skills.pdf", bbox_inches="tight"
)

# %%
num_job_adverts_skills/num_job_adverts_all

# %%
