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

bucket_name = "skills-taxonomy-v2"
s3 = boto3.resource("s3")

# %% [markdown]
# ## Load all TK counts

# %%
all_tk_year_month_counts = pd.read_csv('outputs/tk_analysis/all_tk_year_month_counts.csv')
all_tk_count_region_df = pd.read_csv('outputs/tk_analysis/all_tk_regions_counts.csv')
all_tk_count_subregion_df = pd.read_csv('outputs/tk_analysis/all_tk_subregions_counts.csv')

# %% [markdown]
# ## Load sentences that went into skills

# %%
file_name_date = '2021.08.31'
sentence_clusters = load_s3_data(s3, bucket_name, f'outputs/skills_extraction/extracted_skills/{file_name_date}_sentences_data.json')
sentence_clusters = pd.DataFrame(sentence_clusters)
sentence_clusters.head(2)

# %%
skill_job_ads = set(sentence_clusters['job id'].unique())

# %% [markdown]
# ## How many job adverts

# %%
total_number_jobadvs = 62892486 # Found in 'TextKernel Data.ipynb'

# %%
skill_num_jobadvs = len(skill_job_ads)

# %%
print(f"Sentences that make up skills are from {skill_num_jobadvs} job adverts")
print(f"This is {round(skill_num_jobadvs*100/total_number_jobadvs,2)}% of all job adverts")

# %% [markdown]
# ## Dates
# 'date', 'expiration_date'

# %%
tk_dates = []
for file_name in tqdm(range(0,13)):
    file_date_dict = load_s3_data(
        s3,
        bucket_name,
        f'outputs/tk_data_analysis/metadata_date/{file_name}.json')
    tk_dates.extend([f[0] for job_id, f in file_date_dict.items() if job_id in skill_job_ads])
    
print(len(tk_dates))

# %%
pd.DataFrame(tk_dates).to_csv('tk_dates.csv')

# %%
skill_tk_dates = pd.DataFrame(tk_dates)
skill_tk_dates['date'] = pd.to_datetime(skill_tk_dates[0], format='%Y-%m-%d')

# %%
num_dates_null = sum(pd.isnull(skill_tk_dates[0]))
num_dates_null

# %%
print(len(skill_tk_dates))
skill_tk_dates = skill_tk_dates[pd.notnull(skill_tk_dates[0])]
print(len(skill_tk_dates))

# %%
skill_tk_dates['year'] = pd.DatetimeIndex(skill_tk_dates[0]).year
skill_tk_dates['month'] = pd.DatetimeIndex(skill_tk_dates[0]).month

# %%
year_month_counts = skill_tk_dates.groupby(['year','month'])[0].count()

# %%
year_month_counts = year_month_counts.sort_index().reset_index()
year_month_counts['year/month'] = year_month_counts[['year', 'month']].astype(str).agg('/'.join, axis=1)
year_month_counts

# %%
# Add a row for the None date counts and save
pd.concat([year_month_counts,
          pd.DataFrame([{'year': None, 'month': None, 0: num_dates_null, 'year/month': None}])
          ], ignore_index = True, axis = 0).to_csv('outputs/tk_analysis/skills_tk_year_month_counts.csv')

# %% [markdown]
# ### Get proportions for side by side comparison
# Not using no-date data

# %%
year_month_counts["Proportion"] = year_month_counts[0]/(year_month_counts[0].sum())

# %%
all_tk_year_month_counts_nonull = all_tk_year_month_counts[pd.notnull(all_tk_year_month_counts['year'])]
all_tk_year_month_counts_nonull["Proportion"] = all_tk_year_month_counts_nonull['0']/(all_tk_year_month_counts_nonull['0'].sum())


# %% [markdown]
# ### Plot dates with all TK dates

# %%
nesta_orange = [255/255,90/255,0]
nesta_purple = [155/255,0,195/255]
nesta_grey = [165/255,148/255,130/255]

# %%
ax = all_tk_year_month_counts_nonull.plot(x='year/month', y="Proportion",
    xlabel='Date of job advert', ylabel="Proportion of job adverts", c=nesta_grey)
ax = year_month_counts.plot(x='year/month', y="Proportion",
    xlabel='Date of job advert', ylabel="Proportion of job adverts", c=nesta_orange, ax=ax)

ax.legend(["All TK job adverts", "TK job adverts in sample"]);
ax.figure.savefig('outputs/tk_analysis/job_ad_date_sample_comparison.pdf',bbox_inches='tight')

# %%
all_tk_year_month_counts_nonull['year'] = all_tk_year_month_counts_nonull['year'].astype(int)

# %%
fig = plt.figure(figsize=(7,4)) # Create matplotlib figure

ax = fig.add_subplot(111) # Create matplotlib axes

width = 0.3

pd.DataFrame(all_tk_year_month_counts_nonull.groupby('year')['0'].sum()/sum(all_tk_year_month_counts_nonull['0'])).plot.bar(
    color=nesta_grey,ax=ax,width=width, position=1)


pd.DataFrame(year_month_counts.groupby('year')[0].sum()/sum(year_month_counts[0])).plot.bar(
    color=nesta_orange, ax=ax,width=width, position=0)


ax.set_ylabel('Proportion of job adverts')
ax.set_xlabel('Year of job advert')
ax.legend(["All TK job adverts", "TK job adverts in sample"],loc='upper right');

ax.figure.savefig('outputs/tk_analysis/job_ad_year_sample_comparison.pdf',bbox_inches='tight')

# %% [markdown]
# ## Location 

# %%
tk_region = []
tk_subregion = []
for file_name in tqdm(range(0,13)):
    file_dict = load_s3_data(
        s3,
        bucket_name,
        f'outputs/tk_data_analysis/metadata_location/{file_name}.json')
    tk_region.extend([f[2] for job_id,f in file_dict.items() if f and job_id in skill_job_ads])
    tk_subregion.extend([f[3] for job_id,f in file_dict.items() if f and job_id in skill_job_ads])
    
print(len(tk_region))
print(len(tk_subregion))

# %%
print(len(set(tk_region)))
print(len(set(tk_subregion)))

# %%
count_region_df = pd.DataFrame.from_dict(Counter(tk_region), orient='index')
count_region_df

# %%
count_region_df.to_csv('outputs/tk_analysis/skills_tk_regions_counts.csv')

# %%
print(count_region_df[0].sum())
count_region_df = count_region_df[pd.notnull(count_region_df.index)]
print(count_region_df[0].sum())

# %%
count_region_df

# %%
all_tk_count_region_df_nonull = all_tk_count_region_df[pd.notnull(all_tk_count_region_df['Unnamed: 0'])]
all_tk_count_region_df_nonull.index = all_tk_count_region_df_nonull['Unnamed: 0']
all_tk_count_region_df_nonull

# %%
fig = plt.figure(figsize=(7,4)) # Create matplotlib figure

ax = fig.add_subplot(111) # Create matplotlib axes

width = 0.3

ax = pd.DataFrame(all_tk_count_region_df_nonull['0']/sum(all_tk_count_region_df_nonull['0'])
                 ).sort_values(by=['0'], ascending=False).plot.bar(
    color=nesta_grey, legend=False, ax=ax,width=width, position=1)

ax = pd.DataFrame(count_region_df[0]/sum(count_region_df[0])).plot.bar(
    color=nesta_orange, legend=False, ax=ax,width=width, position=0)

ax.set_ylabel('Proportion of job adverts')
ax.set_xlabel('Region of job advert')
ax.legend(["All TK job adverts", "TK job adverts in sample"],loc='upper right');

ax.figure.savefig('outputs/tk_analysis/job_ad_region_sample_comparison.pdf',bbox_inches='tight')

# %%
count_subregion_df = pd.DataFrame.from_dict(Counter(tk_subregion), orient='index')

# %%
count_subregion_df.to_csv('outputs/tk_analysis/skills_tk_subregions_counts.csv')

# %%
print(count_subregion_df[0].sum())
count_subregion_df = count_subregion_df[pd.notnull(count_subregion_df.index)]
print(count_subregion_df[0].sum())

# %%
all_tk_count_subregion_df_nonull = all_tk_count_subregion_df[pd.notnull(all_tk_count_subregion_df['Unnamed: 0'])]
all_tk_count_subregion_df_nonull.index = all_tk_count_subregion_df_nonull['Unnamed: 0']

# %%
prop_subregions_all = pd.DataFrame(all_tk_count_subregion_df_nonull['0']/sum(all_tk_count_subregion_df_nonull['0']))
prop_subregions_sample = pd.DataFrame(count_subregion_df[0]/sum(count_subregion_df[0]))

# %%
top_50_subregions_all = prop_subregions_all.sort_values(by=['0'], ascending=False)[0:50].index

# %%
fig = plt.figure(figsize=(14,4)) # Create matplotlib figure

ax = fig.add_subplot(111) # Create matplotlib axes

width = 0.3


ax = prop_subregions_all.loc[top_50_subregions_all].plot.bar(
    color=nesta_grey, legend=False, ax=ax,width=width, position=1)

ax = prop_subregions_sample.loc[top_50_subregions_all].plot.bar(
    color=nesta_orange, legend=False, ax=ax,width=width, position=0)

ax.set_ylabel('Proportion of job adverts')
ax.set_xlabel('Subregion of job advert')
ax.legend(["All TK job adverts", "TK job adverts in sample"],loc='upper right');

ax.figure.savefig('outputs/tk_analysis/job_ad_subregion_sample_comparison.pdf',bbox_inches='tight')

# %% [markdown]
# ## Plots together

# %%
width = 0.3

plt.figure(figsize=(12,8))

ax3 = plt.subplot(212)
prop_subregions_all.loc[top_50_subregions_all].plot.bar(
    color=nesta_grey, legend=False, ax=ax3,width=width, position=1)
prop_subregions_sample.loc[top_50_subregions_all].plot.bar(
    color=nesta_orange, legend=False, ax=ax3,width=width, position=0)
ax3.set_ylabel('Proportion of job adverts')
ax3.set_xlabel('Subregion of job advert')

ax1 = plt.subplot(221)
pd.DataFrame(all_tk_year_month_counts_nonull.groupby('year')['0'].sum()/sum(all_tk_year_month_counts_nonull['0'])).plot.bar(
    color=nesta_grey,ax=ax1,width=width, position=1, legend=False)
pd.DataFrame(year_month_counts.groupby('year')[0].sum()/sum(year_month_counts[0])).plot.bar(
    color=nesta_orange, ax=ax1,width=width, position=0, legend=False)
ax1.set_ylabel('Proportion of job adverts')
ax1.set_xlabel('Year of job advert')

ax2 = plt.subplot(222)
pd.DataFrame(all_tk_count_region_df_nonull['0']/sum(all_tk_count_region_df_nonull['0'])
                 ).sort_values(by=['0'], ascending=False).plot.bar(
    color=nesta_grey, legend=False, ax=ax2,width=width, position=1)
pd.DataFrame(count_region_df[0]/sum(count_region_df[0])).plot.bar(
    color=nesta_orange, legend=False, ax=ax2,width=width, position=0)
ax2.set_ylabel('Proportion of job adverts')
ax2.set_xlabel('Region of job advert')
ax2.legend(["All TK job adverts", "TK job adverts in sample"],loc='upper right');

plt.tight_layout()
plt.savefig('outputs/tk_analysis/job_ad_together_sample_comparison.pdf',bbox_inches='tight')

# %%
width = 0.3

plt.figure(figsize=(12,8))

ax1 = plt.subplot(221)
pd.DataFrame(all_tk_year_month_counts_nonull.groupby('year')['0'].sum()/sum(all_tk_year_month_counts_nonull['0'])).plot.bar(
    color=nesta_grey,ax=ax1,width=width, position=1, legend=False)
pd.DataFrame(year_month_counts.groupby('year')[0].sum()/sum(year_month_counts[0])).plot.bar(
    color=nesta_orange, ax=ax1,width=width, position=0, legend=False)
ax1.set_ylabel('Proportion of job adverts')
ax1.set_xlabel('Year of job advert')

ax2 = plt.subplot(222)
pd.DataFrame(all_tk_count_region_df_nonull['0']/sum(all_tk_count_region_df_nonull['0'])
                 ).sort_values(by=['0'], ascending=False).plot.bar(
    color=nesta_grey, legend=False, ax=ax2,width=width, position=1)
pd.DataFrame(count_region_df[0]/sum(count_region_df[0])).plot.bar(
    color=nesta_orange, legend=False, ax=ax2,width=width, position=0)
ax2.set_ylabel('Proportion of job adverts')
ax2.set_xlabel('Region of job advert')
ax2.legend(["All TK job adverts", "TK job adverts in sample"],loc='upper right');

plt.tight_layout()
plt.savefig('outputs/tk_analysis/job_ad_together_sample_comparison_two.pdf',bbox_inches='tight')

# %%
