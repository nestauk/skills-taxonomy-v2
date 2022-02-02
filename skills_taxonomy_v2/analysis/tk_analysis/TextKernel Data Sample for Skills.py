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
# ## Comparing the whole TextKernel dataset to the sample from which skills are extracted
#
# Compare the sample of TK job adverts used in the skills with all the TK data.

# %%
# cd ../../..

# %%
from skills_taxonomy_v2.getters.s3_data import load_s3_data, get_s3_data_paths

from collections import Counter

from datetime import datetime
from tqdm import tqdm
import pandas as pd
import boto3
import matplotlib.pyplot as plt

BUCKET_NAME = "skills-taxonomy-v2"
s3 = boto3.resource("s3")

# %%
output_folder = 'outputs/tk_analysis/2022.01.14/'

# %% [markdown]
# ## Load all TK counts

# %%
all_tk_year_month_counts = load_s3_data(s3, BUCKET_NAME, "outputs/tk_data_analysis_new_method/metadata_date/all_tk_date_count.json")

# %%
all_tk_region_counts = load_s3_data(s3, BUCKET_NAME, "outputs/tk_data_analysis_new_method/metadata_location/all_tk_region_count.json")
all_tk_subregion_counts = load_s3_data(s3, BUCKET_NAME, "outputs/tk_data_analysis_new_method/metadata_location/all_tk_subregion_count.json")

# %%
# all_tk_year_month_counts = pd.read_csv(
#     "outputs/tk_analysis/all_tk_year_month_counts.csv"
# )
# all_tk_count_region_df = pd.read_csv("outputs/tk_analysis/all_tk_regions_counts.csv")
# all_tk_count_subregion_df = pd.read_csv(
#     "outputs/tk_analysis/all_tk_subregions_counts.csv"
# )

# %% [markdown]
# ## Load sentences that went into skills

# %%
skill_sents_data = load_s3_data(s3, BUCKET_NAME, "outputs/skills_extraction/extracted_skills/2022.01.14_sentences_skills_data_lightweight.json")

# %%
skill_unclusteredtoo_job_ads = set([s[0] for s in skill_sents_data])
skill_job_ads = set([s[0] for s in skill_sents_data if s[2]>=0])

# %% [markdown]
# ## How many job adverts

# %%
total_number_jobadvs = 62892486  # Found in 'TextKernel Data.ipynb'

# %%
skill_num_jobadvs = len(skill_job_ads)

# %%
print(f"Sentences that make up skills are from {skill_num_jobadvs} job adverts")
print(
    f"This is {round(skill_num_jobadvs*100/total_number_jobadvs,2)}% of all job adverts"
)

# %%
print(f"Sentences that make up skills (plus unclustered sentences) are from {len(skill_unclusteredtoo_job_ads)} job adverts")
print(
    f"This is {round(len(skill_unclusteredtoo_job_ads)*100/total_number_jobadvs,2)}% of all job adverts"
)

# %% [markdown]
# ## Dates
# 'date', 'expiration_date'

# %%
sample_job_dates_from_metadata = load_s3_data(s3, BUCKET_NAME, "outputs/tk_data_analysis_new_method/metadata_date/14.01.22/sample_filtered.json")

# %%
len(sample_job_dates_from_metadata)

# %%
from collections import defaultdict

# %%
# Each job id's dates found
unique_dates_lists = defaultdict(list)
for job_id, dates_lists in sample_job_dates_from_metadata.items():
    for date_list in dates_lists:
        for date in date_list:
            unique_dates_lists[job_id].append(date)
        
sample_job_dates_dupes = {k:list(set(v)) for k, v in unique_dates_lists.items()}

# %%
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
len(weird_jobs)

# %%
all_sample_job_ids = set(sample_job_dates_from_metadata.keys())

# %%
all_sample_dates_count = defaultdict(int)
for job_id in all_sample_job_ids:
    all_sample_dates_count[sample_job_dates[job_id]] += 1

# %%
skill_job_ads_dates_count = defaultdict(int)
for job_id in skill_job_ads:
    skill_job_ads_dates_count[sample_job_dates[job_id]] += 1

# %%
skill_unclusteredtoo_job_ads_dates_count = defaultdict(int)
for job_id in skill_unclusteredtoo_job_ads:
    skill_unclusteredtoo_job_ads_dates_count[sample_job_dates[job_id]] += 1

# %%
skill_job_ads_dates_count['2015-06-10']


# %%
def count_year_month(counts):
    count_ym = defaultdict(int)
    for k, v in tqdm(counts.items()):
        if k:
            if k=="Not given":
                date = 2014
            elif k=="No date data found yet":
                date = 2013
            else:
                date = k[0:7]
                date = int(date.split("-")[0]) + int(date.split("-")[1]) / 12
        else:
            date = 2014
        count_ym[date] += v
    return count_ym



# %%
def count_year(counts):
    count_y = defaultdict(int)
    for k, v in tqdm(counts.items()):
        if k:
            if k=="Not given":
                year = 2014
            elif k=="No date data found yet":
                year = 2013
            else:
                year = k[0:4]
        else:
            year = 2014
        count_y[float(year)] += v
    return count_y



# %% [markdown]
# ### Plot dates with all TK dates

# %%
nesta_orange = [255 / 255, 90 / 255, 0]
nesta_purple = [155 / 255, 0, 195 / 255]
nesta_grey = [165 / 255, 148 / 255, 130 / 255]


# %%
def plot_prop_data(dates, no_none, plt, label, color, alpha=0.5,width=0.1,position=0):
    if no_none:
        dates = {k:v for k,v in dates.items() if k!=2014}
    plt.bar(
        [k+float(position) for k in dates.keys()],
        [v / sum(dates.values()) for v in dates.values()],
        width=width,
        alpha=alpha,
        color=color,
        label=label,
    )


# %%
plt.figure(figsize=(10, 4))
no_none=True
plot_prop_data(count_year_month(all_tk_year_month_counts), no_none, plt, label="All TK job adverts", color="black", alpha=0.3)
plot_prop_data(count_year_month(skill_job_ads_dates_count), no_none, plt, label="TK job adverts in sample", color=nesta_orange, alpha=0.4)
plt.legend()
plt.xlabel("Date of job advert")
plt.ylabel("Proportion")
plt.savefig(
    f"{output_folder}job_ad_date_sample_comparison.pdf", bbox_inches="tight"
)

# %%
skill_job_ads_dates_count_df = pd.DataFrame.from_dict(skill_job_ads_dates_count, orient='index')
skill_job_ads_dates_count_df['date'] = pd.to_datetime(skill_job_ads_dates_count_df.index)

alltk_job_ads_dates_count_df = pd.DataFrame.from_dict({k:v for k,v in all_tk_year_month_counts.items() if k!='Not given'}, orient='index')
alltk_job_ads_dates_count_df['date'] = pd.to_datetime(alltk_job_ads_dates_count_df.index)


# %%
df1 = alltk_job_ads_dates_count_df.groupby(alltk_job_ads_dates_count_df['date'].dt.to_period('M')).sum()
df1 = df1[0]/df1[0].sum()
# df1 = df1.resample('M').asfreq().fillna(0)
ax1 = df1.plot(kind='bar', width=1, figsize=(15, 4), color="black", alpha=0.5)

df2 = skill_job_ads_dates_count_df.groupby(skill_job_ads_dates_count_df['date'].dt.to_period('M')).sum()
df2 = df2[0]/df2[0].sum()
# df2 = df2.resample('M').asfreq().fillna(0)
df2.plot(kind='bar', width=1,color=nesta_orange, alpha=0.5, ax=ax1)

plt.legend(["All TK job adverts", "TK job adverts in sample"])
plt.xlabel("Date of job advert")
plt.ylabel("Proportion")

# %%
plt.figure(figsize=(10, 5))
no_none=True
plot_prop_data(count_year(all_tk_year_month_counts), no_none, plt,
               label="All TK job adverts", color="black", alpha=0.3, width=0.4,position=0.4)
plot_prop_data(count_year(skill_job_ads_dates_count), no_none, plt,
               label="TK job adverts in sample", color=nesta_orange, alpha=0.7, width=0.4,position=0)
plt.legend(loc = 'lower left')
plt.xlabel("Date of job advert")
plt.ylabel("Proportion")
plt.savefig(
    f"{output_folder}job_ad_date_year_sample_comparison.pdf", bbox_inches="tight"
)

# %%
plt.figure(figsize=(10, 5))
no_none=True
plot_prop_data(count_year(all_tk_year_month_counts), no_none, plt,
               label="All TK job adverts", color="black", alpha=0.3, width=0.2,position=0.4)
plot_prop_data(count_year(all_sample_dates_count), no_none, plt,
               label="Original TK sample job adverts", color="blue", alpha=0.3, width=0.2,position=0.2)
plot_prop_data(count_year(skill_job_ads_dates_count), no_none, plt,
               label="TK job adverts in sample", color=nesta_orange, alpha=0.7, width=0.2,position=0)
plt.legend(loc = 'lower left')
plt.xlabel("Date of job advert")
plt.ylabel("Proportion")
plt.savefig(
    f"{output_folder}job_ad_date_year_sample_comparison_original.pdf", bbox_inches="tight"
)

# %% [markdown]
# ## Location

# %%
all_tk_region_counts.keys()

# %%
sample_job_dates_from_metadata_loc = load_s3_data(s3, BUCKET_NAME, "outputs/tk_data_analysis_new_method/metadata_location/14.01.22/sample_filtered.json")

# %%
# Each job id's locs found
unique_locs_lists = defaultdict(list)
for job_id, locs_lists in sample_job_dates_from_metadata_loc.items():
    for loc_list in locs_lists:
        for loc in loc_list:
            unique_locs_lists[job_id].append(loc[2])
        
sample_job_dates_locs_dupes = {k:list(set(v)) for k, v in unique_locs_lists.items()}

# %%
# Just one location per job id
sample_job_locs = {}
weird_jobs_locs = {}
for job_id, locs_list in unique_locs_lists.items():
    locs = [loc for loc in list(set(locs_list)) if loc]
    if len(locs)==0:
        sample_job_locs[job_id] = None
    elif len(locs)==1:
        # Majority of cases
        sample_job_locs[job_id] = locs[0]
    else:
        weird_jobs_locs[job_id] = locs
        sample_job_locs[job_id] = locs[0]

# %%
all_sample_job_ids_locs = set(sample_job_dates_from_metadata_loc.keys())

# %%
all_sample_locs_count = defaultdict(int)
for job_id in all_sample_job_ids_locs:
    all_sample_locs_count[sample_job_locs[job_id]] += 1

# %%
skill_job_ads_locs_count = defaultdict(int)
for job_id in skill_job_ads:
    skill_job_ads_locs_count[sample_job_locs[job_id]] += 1

# %%
skill_unclusteredtoo_job_ads_locs_count = defaultdict(int)
for job_id in skill_unclusteredtoo_job_ads:
    skill_unclusteredtoo_job_ads_locs_count[sample_job_locs[job_id]] += 1


# %%
def plot_prop_data_locs(locs, no_none, plt, label, color, alpha=0.5,width=0.1,position=0):
    if no_none:
        locs = {k:v for k,v in locs.items() if (k and k!='Not given')}
        loc_i = {'England':0, 'Scotland':1, 'Wales':2, 'Northern Ireland':3}
    else:
        locs = {(k if k else 'Not given'):v for k,v in locs.items()}
        loc_i = {'England':0, 'Scotland':1, 'Wales':2, 'Northern Ireland':3, 'Not given':4}
    
    plt.bar(
        [i+position for i in loc_i.values()],
        [locs[k] / sum(locs.values()) for k in loc_i.keys()],
        width=width,
        alpha=alpha,
        color=color,
        label=label,
    )
    plt.xticks(list(loc_i.values()), list(loc_i.keys()))


# %%
skill_job_ads_locs_count

# %%
plt.figure(figsize=(10, 5))
no_none=True
plot_prop_data_locs(all_tk_region_counts, no_none, plt,
               label="All TK job adverts", color="black", alpha=0.3, width=0.4,position=0.4)
plot_prop_data_locs(skill_job_ads_locs_count, no_none, plt,
               label="TK job adverts in sample", color=nesta_orange, alpha=0.7, width=0.4,position=0)
plt.legend(loc = 'upper right')
plt.xlabel("Date of job advert")
plt.ylabel("Proportion")
plt.savefig(
    f"{output_folder}job_ad_date_year_sample_comparison.pdf", bbox_inches="tight"
)

# %% [markdown]
# ## Plots together

# %%
width = 0.3

plt.figure(figsize=(12, 8))

ax1 = plt.subplot(221)
no_none=True
plot_prop_data(count_year_month(all_tk_year_month_counts), no_none, plt, label="All TK job adverts", color="black", alpha=0.3)
plot_prop_data(count_year_month(skill_job_ads_dates_count), no_none, plt, label="TK job adverts in sample", color=nesta_orange, alpha=0.4)
plt.legend()
plt.xlabel("Date of job advert")
plt.ylabel("Proportion")
plt.savefig(
    f"{output_folder}job_ad_date_sample_comparison.pdf", bbox_inches="tight"
)

ax2 = plt.subplot(222)
no_none=True
plot_prop_data_locs(all_tk_region_counts, no_none, plt,
               label="All TK job adverts", color="black", alpha=0.3, width=0.4,position=0.4)
plot_prop_data_locs(skill_job_ads_locs_count, no_none, plt,
               label="TK job adverts in sample", color=nesta_orange, alpha=0.7, width=0.4,position=0)
plt.legend(loc = 'upper right')
plt.xlabel("Date of job advert")
plt.ylabel("Proportion")

plt.tight_layout()
plt.savefig(
    f"{output_folder}job_ad_together_sample_comparison_two.pdf", bbox_inches="tight"
)

# %%
