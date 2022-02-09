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
# ## Pre-and post-COVID changes in skill demands
# - How have skill groups changed over time
# - Pre and post covid

# %%
# cd ../../..

# %%
from bokeh.palettes import Turbo256
from skills_taxonomy_v2.getters.s3_data import load_s3_data

# %%
from collections import Counter, defaultdict
import random
from tqdm import tqdm
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import boto3

# %%
bucket_name = "skills-taxonomy-v2"
s3 = boto3.resource("s3")

# %%
hier_date = '2022.01.21'
skills_date = '2022.01.14'

# %%
skill_hierarchy_file = f"outputs/skills_taxonomy/{hier_date}_skills_hierarchy_named.json"
skill_hierarchy = load_s3_data(s3, bucket_name, skill_hierarchy_file)

# %%
sentence_data = load_s3_data(
    s3,
    bucket_name,
    f"outputs/skills_extraction/extracted_skills/{skills_date}_sentences_skills_data_lightweight.json",
)
sentence_data = pd.DataFrame(sentence_data, columns=['job id', 'sentence id',  'Cluster number predicted'])
sentence_data = sentence_data[sentence_data["Cluster number predicted"] >=0]

# %% [markdown]
# ### Add duplicated sentences

# %%
dupe_words_id = load_s3_data(
    s3,
    bucket_name,
    f"outputs/skills_extraction/word_embeddings/data/2022.01.14_unique_words_id_list.json",
)

# %%
# The job ids in the skill sentences which have duplicates
dupe_job_ids = set(sentence_data['job id'].tolist()).intersection(set(dupe_words_id.keys()))
# What are the word ids for these?
skill_job_ids_with_dupes_list = [(job_id, sent_id, word_id) for job_id, s_w_list in dupe_words_id.items() for (word_id, sent_id) in s_w_list if job_id in dupe_job_ids]
skill_job_ids_with_dupes_df = pd.DataFrame(skill_job_ids_with_dupes_list, columns = ['job id', 'sentence id', 'words id'])
del skill_job_ids_with_dupes_list
# Get the words id for the existing deduplicated sentence data
sentence_data_ehcd = sentence_data.merge(skill_job_ids_with_dupes_df, how='left', on=['job id', 'sentence id'])
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
del sentence_data_ehcd, dupe_sentence_data


# %%
# Add new duplicates to sentence data
print(len(sentence_data))
sentence_data = pd.concat([sentence_data, dupe_sentence_data_df])
sentence_data.drop_duplicates(inplace=True)
sentence_data.reset_index(inplace=True)
print(len(sentence_data))
del dupe_sentence_data_df

# %% [markdown]
# ### Add hierarchy information to this df

# %%

sentence_data["Hierarchy level A name"] = sentence_data["Cluster number predicted"].apply(
    lambda x: skill_hierarchy[str(x)]["Hierarchy level A name"]
)
sentence_data["Hierarchy level B name"] = sentence_data["Cluster number predicted"].apply(
    lambda x: skill_hierarchy[str(x)]["Hierarchy level B name"]
)
sentence_data["Hierarchy level C name"] = sentence_data["Cluster number predicted"].apply(
    lambda x: skill_hierarchy[str(x)]["Hierarchy level C name"]
)


# %%
level_a_mapper = {}
for skill in skill_hierarchy.values():
    level_a_mapper[skill['Hierarchy level A name']] = skill['Hierarchy level A']
level_a_mapper

# %% [markdown]
# ### Import the job advert year data

# %%
job_dates = load_s3_data(s3, bucket_name, "outputs/tk_data_analysis_new_method/metadata_date/14.01.22/sample_filtered.json")

# %%
job_dates_job_ids = set(job_dates.keys())

# %%
from collections import defaultdict
# Each job id's dates found
unique_dates_lists = defaultdict(list)
for job_id, dates_lists in job_dates.items():
    for date_list in dates_lists:
        for date in date_list:
            unique_dates_lists[job_id].append(date)
        
sample_job_dates_dupes = {k:list(set(v)) for k, v in unique_dates_lists.items()}

# Just one date per job id
sample_job_dates = {}
weird_jobs = {}
for job_id, date_list in unique_dates_lists.items():
    dates = [date for date in list(set(date_list)) if date]
    if len(dates)==0:
        sample_job_dates[job_id] = None
    elif len(dates)==1:
        # Majority of cases
        sample_job_dates[job_id] = dates[0]
    else:
        weird_jobs[job_id] = dates
        sample_job_dates[job_id] = dates[0]

# %%
print(len(sentence_data))
sentence_data_with_meta = sentence_data.copy()[
    sentence_data["job id"].isin(job_dates_job_ids)
]
print(len(sentence_data_with_meta))
sentence_data_with_meta["date"] = sentence_data_with_meta["job id"].apply(
    lambda x: sample_job_dates.get(x)
)
sentence_data_with_meta = sentence_data_with_meta[
    sentence_data_with_meta["date"].notnull()
]
print(len(sentence_data_with_meta))

# %%
num_job_year = sentence_data_with_meta["job id"].nunique()
num_all_job = sentence_data["job id"].nunique()
print(
    f"{num_job_year} of {num_all_job} ({round(num_job_year*100/num_all_job,2)}%) job adverts have date metadata"
)

# %%
sentence_data_with_meta.head(2)

# %%
sentence_data_with_meta["year"] = pd.DatetimeIndex(sentence_data_with_meta["date"]).year
sentence_data_with_meta["month"] = pd.DatetimeIndex(
    sentence_data_with_meta["date"]
).month
sentence_data_with_meta["covid"] = sentence_data_with_meta["date"].apply(
    lambda x: "Pre-COVID"
    if float(x[0:7].replace("-", ".")) <= 2020.02
    else "Post-COVID"
)


# %%
sentence_data_with_meta["date"].min()

# %%
sentence_data_with_meta["date"].max()

# %% [markdown]
# ## Colours

# %%
nesta_orange = [255 / 255, 90 / 255, 0 / 255]
nesta_grey = [165 / 255, 148 / 255, 130 / 255]

# %%
from bokeh.models import (
    LinearColorMapper,
)

# %%
num_lev_as = sentence_data['Hierarchy level A name'].nunique()
levela_cols = []
for i in range(0, num_lev_as):
    levela_cols.append(Turbo256[i * round(len(Turbo256) / num_lev_as)])
# levela_cols = levela_cols[0:-1]
random.seed(15)
random.shuffle(levela_cols)
levela_cols[10]="grey"
levela_cols

# %%
level_a_mapper_names = list(level_a_mapper.keys())
level_a_mapper_names.sort()
levela_cols = [levela_cols[level_a_mapper[level_a_name]] for level_a_name in level_a_mapper_names]

# %% [markdown]
# ## Hist of years

# %%
print(len(sentence_data_with_meta))
unique_job_adverts_df = sentence_data_with_meta[
    ["job id", "covid", "date"]
].drop_duplicates()
len(unique_job_adverts_df)

# %%
pd.to_datetime(
    unique_job_adverts_df[unique_job_adverts_df["covid"] == "Post-COVID"]["date"]
).hist(bins=12, grid=False, color=nesta_orange, label="Post-COVID", alpha=0.9, figsize=(15, 4))
pd.to_datetime(
    unique_job_adverts_df[unique_job_adverts_df["covid"] == "Pre-COVID"]["date"]
).hist(bins=50, grid=False, color=nesta_grey, label="Pre-COVID", alpha=0.9)
plt.xlabel("Date of job advert")
plt.ylabel("Number of job adverts")
plt.legend()
plt.savefig(
    f"outputs/skills_taxonomy_application/covid_application/{hier_date}/num_job_adverts_date.pdf",
    bbox_inches="tight",
)

# %% [markdown]
# ## Proportion of year from each year

# %%
prop_level_a_year = sentence_data_with_meta.groupby("year")[
    "Hierarchy level A name"
].apply(lambda x: x.value_counts() / len(x))

prop_level_a_year.unstack().plot.barh(
    stacked=True,
    title="Proportion of level A skill groups for each year",
    ylabel="",
    xlabel="",
    color=levela_cols,
)
plt.legend(bbox_to_anchor=(1.05, 1))

plt.savefig(
    f"outputs/skills_taxonomy_application/covid_application/{hier_date}/year_prop_a.pdf",
    bbox_inches="tight",
)


# %%
prop_level_a_covid = sentence_data_with_meta.groupby("covid")[
    "Hierarchy level A name"
].apply(lambda x: x.value_counts() / len(x))

prop_level_a_covid.unstack().plot.barh(
    stacked=True,
    title="Proportion of level A skill groups\nfor pre- and post- COVID job adverts",
    ylabel="",
    xlabel="",
    color=levela_cols,
)
plt.legend(bbox_to_anchor=(1.05, 1))

plt.savefig(
    f"outputs/skills_taxonomy_application/covid_application/{hier_date}/covid_prop_a.pdf",
    bbox_inches="tight",
)


# %%
# Another view type
prop_level_a_covid.unstack().plot.bar(
    stacked=False,
    title="Proportion of level A skill groups for pre- and post- COVID job adverts",
    ylabel="",
    xlabel="",
    figsize=(10, 4),
    color=levela_cols,
)
plt.legend(bbox_to_anchor=(1.05, 1))

plt.savefig(
    f"outputs/skills_taxonomy_application/covid_application/{hier_date}/covid_prop_a_T.pdf",
    bbox_inches="tight",
)


# %%
prop_level_a_covid.reset_index().groupby(["level_1", "covid"]).apply(
    lambda x: x["Hierarchy level A name"].iloc[0]
).unstack().plot.barh(
    stacked=False,
    title="Proportion of level A skill groups for pre- and post- COVID job adverts",
    figsize=(8, 3),
    color=[nesta_grey, "black"],
)
plt.legend(bbox_to_anchor=(1.05, 1))
plt.ylabel("")
plt.savefig(
    f"outputs/skills_taxonomy_application/covid_application/{hier_date}/covid_prop_a_T2.pdf",
    bbox_inches="tight",
)


# %%
sentence_data_with_meta_filter = sentence_data_with_meta[
    sentence_data_with_meta["covid"] == "Post-COVID"
]
level_a_prop_post_covid = (
    sentence_data_with_meta_filter["Hierarchy level A name"].value_counts()
    * 100
    / len(sentence_data_with_meta_filter)
)

sentence_data_precovid = sentence_data_with_meta[
    sentence_data_with_meta["covid"] == "Pre-COVID"
]
level_a_prop_pre_covid = (
    sentence_data_precovid["Hierarchy level A name"].value_counts()
    * 100
    / len(sentence_data_precovid)
)

df = pd.concat(
    [
        pd.DataFrame(level_a_prop_pre_covid).rename(
            columns={
                "Hierarchy level A name": "Percentage of level A skill group in pre-covid job adverts only"
            }
        ),
        pd.DataFrame(level_a_prop_post_covid).rename(
            columns={
                "Hierarchy level A name": "Percentage of level A skill group in post-covid job adverts only"
            }
        ),
    ],
    axis=1,
)
df["Increase from before to after COVID"] = (
    df["Percentage of level A skill group in post-covid job adverts only"]
    / df["Percentage of level A skill group in pre-covid job adverts only"]
)
df.round(3).to_csv(
    f"outputs/skills_taxonomy_application/covid_application/{hier_date}/covid_prepost_leva.csv"
)


# %%
prop_level_a_covid.reset_index().groupby(["level_1", "covid"]).apply(
    lambda x: x["Hierarchy level A name"].iloc[0]
)

# %% [markdown]
# ## pre vs post covid quotients

# %%
## Only include skills groups that make up at least x% of skills
## to avoid large changes in very small groups.
level_b_prop_thresh = 0.01
level_b_grouped = sentence_data_with_meta.groupby("Hierarchy level B name")
print(len(level_b_grouped))
level_b_all_prop = level_b_grouped["Hierarchy level B name"].count() / len(
    sentence_data_with_meta
)
level_b_high_prob_names = level_b_all_prop[
    level_b_all_prop >= level_b_prop_thresh
].index.tolist()
print(len(level_b_high_prob_names))

sentence_data_with_meta_high_levb = sentence_data_with_meta[
    sentence_data_with_meta["Hierarchy level B name"].isin(level_b_high_prob_names)
]
len(sentence_data_with_meta_high_levb)

# %%
# level_b_prop_all = sentence_data_with_meta['Hierarchy level B name'].value_counts()/len(sentence_data_with_meta)

sentence_data_postcovid = sentence_data_with_meta_high_levb[
    sentence_data_with_meta_high_levb["covid"] == "Post-COVID"
]
level_b_prop_post_covid = sentence_data_postcovid[
    "Hierarchy level B name"
].value_counts() / len(sentence_data_postcovid)

sentence_data_precovid = sentence_data_with_meta_high_levb[
    sentence_data_with_meta_high_levb["covid"] == "Pre-COVID"
]
level_b_prop_pre_covid = sentence_data_precovid[
    "Hierarchy level B name"
].value_counts() / len(sentence_data_precovid)


covid_quotient = level_b_prop_post_covid / level_b_prop_pre_covid
covid_quotient = covid_quotient[pd.notnull(covid_quotient)].sort_values(ascending=True)

covid_quotient.plot.barh(
    figsize=(8, 10),
    ylabel="",
    xlabel="Level B hierarchy",
    title="Post-COVID compared to pre-COVID proportions of level B skill groups",
    color=nesta_orange,
)
plt.axvline(1, color="black")

plt.savefig(
    f"outputs/skills_taxonomy_application/covid_application/{hier_date}/covid_prepost_levb.pdf",
    bbox_inches="tight",
)


# %%
low_covid_quotient = covid_quotient.sort_values()[0:10].index.tolist()
high_covid_quotient = covid_quotient.sort_values()[-10:].index.tolist()

# %%
df = pd.concat(
    [
        pd.DataFrame(
            level_b_prop_pre_covid[low_covid_quotient + high_covid_quotient]
        ).rename(
            columns={
                "Hierarchy level B name": "Percentage of level B skill group in pre-covid job adverts only"
            }
        ),
        pd.DataFrame(
            level_b_prop_post_covid[low_covid_quotient + high_covid_quotient]
        ).rename(
            columns={
                "Hierarchy level B name": "Percentage of level B skill group in post-covid job adverts only"
            }
        ),
    ],
    axis=1,
)
df = df * 100
df["Change from before to after COVID"] = (
    df["Percentage of level B skill group in post-covid job adverts only"]
    * 100
    / df["Percentage of level B skill group in pre-covid job adverts only"]
)
df.round(2).to_csv(
    f"outputs/skills_taxonomy_application/covid_application/{hier_date}/covid_prepost_levb.csv"
)


# %%
df.round(2)

# %%
# Level C covid

# %%
## Only include skills groups that make up at least x% of skills
## to avoid large changes in very small groups.
level_c_prop_thresh = 0.001
level_c_grouped = sentence_data_with_meta.groupby("Hierarchy level C name")
print(len(level_c_grouped))
level_c_all_prop = level_c_grouped["Hierarchy level C name"].count() / len(
    sentence_data_with_meta
)
level_c_high_prob_names = level_c_all_prop[
    level_c_all_prop >= level_c_prop_thresh
].index.tolist()
print(len(level_c_high_prob_names))

sentence_data_with_meta_high_levc = sentence_data_with_meta[
    sentence_data_with_meta["Hierarchy level C name"].isin(level_c_high_prob_names)
]
len(sentence_data_with_meta_high_levc)

# %%
sentence_data_precovid = sentence_data_with_meta_high_levc[
    sentence_data_with_meta_high_levc["covid"] == "Pre-COVID"
]
sentence_data_postcovid = sentence_data_with_meta_high_levc[
    sentence_data_with_meta_high_levc["covid"] == "Post-COVID"
]

level_c_prop_post_covid = sentence_data_postcovid[
    "Hierarchy level C name"
].value_counts() / len(sentence_data_postcovid)
level_c_prop_pre_covid = sentence_data_precovid[
    "Hierarchy level C name"
].value_counts() / len(sentence_data_precovid)

covid_quotient_levc = level_c_prop_post_covid / level_c_prop_pre_covid
covid_quotient_levc = covid_quotient_levc[pd.notnull(covid_quotient_levc)].sort_values(
    ascending=True
)


# %%
low_covid_quotient_levc = covid_quotient_levc.sort_values()[0:10].index.tolist()
high_covid_quotient_levc = covid_quotient_levc.sort_values()[-10:].index.tolist()

# %%
df = pd.concat(
    [
        pd.DataFrame(
            level_c_prop_pre_covid[low_covid_quotient_levc + high_covid_quotient_levc]
        ).rename(
            columns={
                "Hierarchy level C name": "Percentage of level C skill group in pre-covid job adverts only"
            }
        ),
        pd.DataFrame(
            level_c_prop_post_covid[low_covid_quotient_levc + high_covid_quotient_levc]
        ).rename(
            columns={
                "Hierarchy level C name": "Percentage of level C skill group in post-covid job adverts only"
            }
        ),
    ],
    axis=1,
)
df = df * 100
df["Increase from before to after COVID"] = (
    df["Percentage of level C skill group in post-covid job adverts only"]
    * 100
    / df["Percentage of level C skill group in pre-covid job adverts only"]
)
df.round(2).to_csv(
    f"outputs/skills_taxonomy_application/covid_application/{hier_date}/covid_prepost_levc.csv"
)


# %%
df.round(2)

# %% [markdown]
# ## Most common job titles pre and post COVID

# %%
job_dates = load_s3_data(s3, bucket_name, "outputs/tk_data_analysis_new_method/metadata_date/14.01.22/sample_filtered.json")

# %%
job_dates_job_ids = set(job_dates.keys())

# %%
from collections import defaultdict
# Each job id's dates found
unique_dates_lists = defaultdict(list)
for job_id, dates_lists in job_dates.items():
    for date_list in dates_lists:
        for date in date_list:
            unique_dates_lists[job_id].append(date)
        
sample_job_dates_dupes = {k:list(set(v)) for k, v in unique_dates_lists.items()}

# Just one date per job id
sample_job_dates = {}
weird_jobs = {}
for job_id, date_list in unique_dates_lists.items():
    dates = [date for date in list(set(date_list)) if date]
    if len(dates)==0:
        sample_job_dates[job_id] = None
    elif len(dates)==1:
        # Majority of cases
        sample_job_dates[job_id] = dates[0]
    else:
        weird_jobs[job_id] = dates
        sample_job_dates[job_id] = dates[0]

# %%
job_titles = load_s3_data(
    s3,
    bucket_name,
    "outputs/tk_data_analysis_new_method/metadata_job/14.01.22/sample_filtered.json"
)

# %%
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
job_titles_one['90929d21a5b44dd6adc72e0ace1b5371']

# %%
sample_job_dates['90929d21a5b44dd6adc72e0ace1b5371']

# %%
len(job_titles_one)

# %%
precovid_job_counts = defaultdict(int)
postcovid_job_counts = defaultdict(int)
for job_id, job_title in job_titles_one.items():
    job_date = sample_job_dates.get(job_id)
    if job_date:
        if float(job_date[0:7].replace("-", ".")) <= 2020.02:
            precovid_job_counts[job_title[0]] += 1
        else:
            postcovid_job_counts[job_title[0]] += 1

# %%
len(precovid_job_counts) + len(postcovid_job_counts)

# %%
post_counts_sort = [(k,v) for k,v in postcovid_job_counts.items()]
post_counts_sort.sort(key=lambda x:x[1], reverse=True)
pre_counts_sort = [(k,v) for k,v in precovid_job_counts.items()]
pre_counts_sort.sort(key=lambda x:x[1], reverse=True)

# %%
post_counts_sort[0:10]

# %%
pre_counts_sort[0:10]

# %%
