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
hier_structure_file = "outputs/skills_hierarchy/2021.09.06_hierarchy_structure.json"
hier_structure = load_s3_data(s3, bucket_name, hier_structure_file)

# %%
skill_hierarchy_file = "outputs/skills_hierarchy/2021.09.06_skills_hierarchy.json"
skill_hierarchy = load_s3_data(s3, bucket_name, skill_hierarchy_file)

# %%
sentence_data = load_s3_data(
    s3,
    bucket_name,
    "outputs/skills_extraction/extracted_skills/2021.08.31_sentences_data.json",
)

# %%
skills_data = load_s3_data(
    s3,
    bucket_name,
    "outputs/skills_extraction/extracted_skills/2021.08.31_skills_data.json",
)

# %%
with open("skills_taxonomy_v2/utils/2021.09.06_level_a_rename_dict.json", "r") as f:
    level_a_rename_dict = json.load(f)

# %% [markdown]
# ### Join the sentence data with hierarchy information

# %%
sentence_data = pd.DataFrame(sentence_data)
sentence_data = sentence_data[sentence_data["Cluster number"] != -1]

sentence_data["Hierarchy level A"] = (
    sentence_data["Cluster number"]
    .astype(str)
    .apply(lambda x: skill_hierarchy[x]["Hierarchy level A"])
)
sentence_data["Hierarchy level A name"] = (
    sentence_data["Hierarchy level A"]
    .astype(str)
    .apply(lambda x: level_a_rename_dict[x])
)
sentence_data["Hierarchy level B"] = (
    sentence_data["Cluster number"]
    .astype(str)
    .apply(lambda x: skill_hierarchy[x]["Hierarchy level B"])
)
sentence_data["Hierarchy level C"] = (
    sentence_data["Cluster number"]
    .astype(str)
    .apply(lambda x: skill_hierarchy[x]["Hierarchy level C"])
)
sentence_data["Hierarchy level D"] = (
    sentence_data["Cluster number"]
    .astype(str)
    .apply(lambda x: skill_hierarchy[x]["Hierarchy level D"])
)
sentence_data["Hierarchy ID"] = (
    sentence_data["Cluster number"]
    .astype(str)
    .apply(lambda x: skill_hierarchy[x]["Hierarchy ID"])
)

# %% [markdown]
# ## Evaluate
# - Look at skills for individual jobs - each job's skills should be roughly in the same branch of the hierarchy
# - Warning: sometimes there is only one skill in a job id, so this would sounds better than it is

# %%
nesta_orange = [255 / 255, 90 / 255, 0 / 255]

# %%
job_id_levels = sentence_data.groupby("job id").agg(
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
    "outputs/skills_taxonomy/figures/2021.09.06/evaluate_mean_nums.pdf",
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
    "outputs/skills_taxonomy/figures/2021.09.06/evaluate_nums_normalised.pdf",
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
    "outputs/skills_taxonomy/figures/2021.09.06/evaluate_nums_raw.pdf",
    bbox_inches="tight",
)


# %% [markdown]
# ## Raw number for when number sentences==10

# %%
len(job_id_levels[job_id_levels["sentence id"] == 6])

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
    "outputs/skills_taxonomy/figures/2021.09.06/evaluate_nums_6.pdf",
    bbox_inches="tight",
)


# %%
job_id_levels["sentence id"].plot.hist(bins=100)

# %% [markdown]
# ## Most common job titles for skills

# %%
job_titles = load_s3_data(
    s3, bucket_name, "outputs/tk_data_analysis/metadata_job/sample_filtered.json"
)

# %%
len(job_titles)

# %%
sentence_data["Job title"] = sentence_data["job id"].apply(
    lambda x: job_titles.get(x, [None])[0]
)
sentence_data["Organization industry"] = sentence_data["job id"].apply(
    lambda x: job_titles.get(x, [None, None])[1]
)

# %%
print(f"There are {sentence_data['Job title'].nunique()} unique job titles")
print(
    f"There are {sentence_data['Organization industry'].nunique()} unique organization industries"
)

# %%
print(
    f"{round(sum(sentence_data['Job title'].notnull())*100/len(sentence_data),3)}% skill sentences have job title data"
)
print(
    f"{round(sum(sentence_data['Organization industry'].notnull())*100/len(sentence_data),3)}% skill sentences have organization industry data"
)


# %% [markdown]
# ### Which skill group levels does each job have the highest proportions in

# %%
job_skill_leva_props = {}
job_skill_levb_props = {}
job_skill_levc_props = {}
for job_name, job_group in sentence_data.groupby(["Job title"]):
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
sentence_data["Job title"].nunique()

# %%
sum(sentence_data["Job title"].value_counts() > 10)

# %%
min_num_job_ids = 10

# %% [markdown]
# ### Level A skills

# %%
leve_a_group["Cluster number"].apply(
    lambda x: skills_data[str(x)]["Skills name"]
).value_counts()[0:5].index

# %%
leve_a_group["Cluster number"].nunique()

# %%
leve_a_group["job id"].nunique()

# %%
lev_a_top_jobs = []
for level_a_name, leve_a_group in sentence_data.groupby(["Hierarchy level A"]):
    job_titles = leve_a_group["Job title"].value_counts(sort=True)
    org_inds = leve_a_group["Organization industry"].value_counts(sort=True)

    top_job_titles_by_prop = job_skill_leva_props_df[
        job_skill_leva_props_df["Number of unique job ids"] > min_num_job_ids
    ][level_a_name].sort_values(ascending=False)[0:10]

    lev_a_top_jobs.append(
        {
            "Level A name": level_a_rename_dict[str(level_a_name)],
            "Number of unique skills": leve_a_group["Cluster number"].nunique(),
            "Number of unique job ids": leve_a_group["job id"].nunique(),
            "Number of skill sentences": len(leve_a_group),
            "Top 5 most common skills": list(
                leve_a_group["Cluster number"]
                .apply(lambda x: skills_data[str(x)]["Skills name"])
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
    "outputs/skills_taxonomy/evaluation/top_10_jobs_levela.csv"
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
            for lev_d_id, lev_d in lev_c["Level D"].items():
                lev_d_name_dict[lev_d_id] = lev_d["Name"]

# %%
lev_b_top_jobs = []
for level_b_name, leve_b_group in sentence_data.groupby(["Hierarchy level B"]):
    job_titles = leve_b_group["Job title"].value_counts(sort=True)
    org_inds = leve_b_group["Organization industry"].value_counts(sort=True)

    top_job_titles_by_prop = job_skill_levb_props_df[
        job_skill_levb_props_df["Number of unique job ids"] > min_num_job_ids
    ][level_b_name].sort_values(ascending=False)[0:10]

    lev_b_top_jobs.append(
        {
            "Level B name": lev_b_name_dict[str(level_b_name)],
            "Number of unique skills": leve_b_group["Cluster number"].nunique(),
            "Number of unique job ids": leve_b_group["job id"].nunique(),
            "Number of skill sentences": len(leve_b_group),
            "Top 5 most common skills": list(
                leve_b_group["Cluster number"]
                .apply(lambda x: skills_data[str(x)]["Skills name"])
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
    "outputs/skills_taxonomy/evaluation/top_10_jobs_levelb.csv"
)

# %% [markdown]
# ### Level C
# Too many - so just the most common?

# %%
lev_c_top_jobs = []
for level_c_name, leve_c_group in sentence_data.groupby(["Hierarchy level C"]):
    job_titles = leve_c_group["Job title"].value_counts(sort=True)
    org_inds = leve_c_group["Organization industry"].value_counts(sort=True)

    top_job_titles_by_prop = job_skill_levc_props_df[
        job_skill_levc_props_df["Number of unique job ids"] > min_num_job_ids
    ][level_c_name].sort_values(ascending=False)[0:10]

    lev_c_top_jobs.append(
        {
            "Level C name": lev_c_name_dict[str(level_c_name)],
            "Number of unique skills": leve_c_group["Cluster number"].nunique(),
            "Number of unique job ids": leve_c_group["job id"].nunique(),
            "Number of skill sentences": len(leve_c_group),
            "Top 5 most common skills": list(
                leve_c_group["Cluster number"]
                .apply(lambda x: skills_data[str(x)]["Skills name"])
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
    "outputs/skills_taxonomy/evaluation/top_10_jobs_levelc.csv"
)

# %% [markdown]
# ## For each job, most common skill level C

# %%
top_jobs = sentence_data["Job title"].value_counts()[0:100].index.tolist()

# %%
sentence_data["Skill name"] = sentence_data["Cluster number"].apply(
    lambda x: skills_data[str(x)]["Skills name"]
)
sentence_data["Hierarchy level B name"] = sentence_data["Hierarchy level B"].apply(
    lambda x: lev_b_name_dict[str(x)]
)
sentence_data["Hierarchy level C name"] = sentence_data["Hierarchy level C"].apply(
    lambda x: lev_c_name_dict[str(x)]
)

# %%
per_top_job_levels = []
for job_title in top_jobs:
    job_df = sentence_data[sentence_data["Job title"] == job_title]
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
                f"{j} ({n})" for j, n in list(zip(levas.index[0:1], levas[0:1]))
            ],
            "5 most common level B skill groups": [
                f"{j} ({n})" for j, n in list(zip(levbs.index[0:5], levbs[0:5]))
            ],
            "5 most common level C skill groups": [
                f"{j} ({n})" for j, n in list(zip(levcs.index[0:5], levcs[0:5]))
            ],
            "5 most common skills": [
                f"{j} ({n})" for j, n in list(zip(skills_d.index[0:10], skills_d[0:5]))
            ],
        }
    )


# %%
pd.DataFrame(per_top_job_levels).to_csv(
    "outputs/skills_taxonomy/evaluation/jobs_top_skill_groups.csv"
)

# %%
