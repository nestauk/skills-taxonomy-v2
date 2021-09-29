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
skill_hierarchy_file = 'outputs/skills_hierarchy/2021.09.06_skills_hierarchy.json'
skill_hierarchy = load_s3_data(s3, bucket_name, skill_hierarchy_file)

# %%
sentence_data = load_s3_data(s3, bucket_name, 'outputs/skills_extraction/extracted_skills/2021.08.31_sentences_data.json')
sentence_data = pd.DataFrame(sentence_data)
sentence_data = sentence_data[sentence_data['Cluster number']!=-1]


# %%
# Manual level A names
with open('skills_taxonomy_v2/utils/2021.09.06_level_a_rename_dict.json', 'r') as f:
    level_a_rename_dict = json.load(f)

# %%
# Add hierarchy information to this df
sentence_data['Hierarchy level A name'] = sentence_data['Cluster number'].apply(lambda x: level_a_rename_dict[str(
    skill_hierarchy[str(x)]['Hierarchy level A'])])
sentence_data['Hierarchy level B name'] = sentence_data['Cluster number'].apply(lambda x: skill_hierarchy[str(x)]['Hierarchy level B name'])
sentence_data['Hierarchy level C name'] = sentence_data['Cluster number'].apply(lambda x: skill_hierarchy[str(x)]['Hierarchy level C name'])


# %% [markdown]
# ### Import the job advert year data

# %%
job_dates = load_s3_data(s3, bucket_name,
                             'outputs/tk_data_analysis/metadata_date/sample_filtered.json')

# %%
sentence_data_with_meta = sentence_data.copy()[sentence_data['job id'].isin(job_dates.keys())]
print(len(sentence_data_with_meta))
sentence_data_with_meta['date'] = sentence_data_with_meta['job id'].apply(
    lambda x: job_dates.get(x))
sentence_data_with_meta = sentence_data_with_meta[sentence_data_with_meta['date'].notnull()]
print(len(sentence_data_with_meta))

# %%
num_job_year = sentence_data_with_meta['job id'].nunique()
num_all_job = sentence_data['job id'].nunique()
print(f"{num_job_year} of {num_all_job} ({round(num_job_year*100/num_all_job,2)}%) job adverts have date metadata")

# %%
sentence_data_with_meta['year'] = pd.DatetimeIndex(sentence_data_with_meta['date']).year
sentence_data_with_meta['month'] = pd.DatetimeIndex(sentence_data_with_meta['date']).month
sentence_data_with_meta['covid'] = sentence_data_with_meta['date'].apply(
    lambda x: 'Pre-COVID' if float(x[0:7].replace('-','.'))<=2020.02 else 'Post-COVID')


# %% [markdown]
# ## Colours

# %%
nesta_orange = [255/255,90/255,0/255]
nesta_grey = [165/255,148/255,130/255]

# %%
from bokeh.palettes import Turbo256
levela_cols = []
for i in range(0,7):
    levela_cols.append(Turbo256[i*round(len(Turbo256)/7)])
levela_cols = levela_cols[0:6]

# %% [markdown]
# ## Proportion of year from each year

# %%
prop_level_a_year = sentence_data_with_meta.groupby('year')['Hierarchy level A name'].apply(
    lambda x: x.value_counts()/len(x))

prop_level_a_year.unstack().plot.barh(
    stacked=True,
    title ="Proportion of level A skill groups for each year",
    ylabel="",
    xlabel="",
    color = levela_cols)
plt.legend(bbox_to_anchor=(1.05, 1))

plt.savefig('outputs/skills_taxonomy_application/covid_application/year_prop_a.pdf',bbox_inches='tight')


# %%
prop_level_a_covid = sentence_data_with_meta.groupby('covid')['Hierarchy level A name'].apply(
    lambda x: x.value_counts()/len(x))

prop_level_a_covid.unstack().plot.barh(
    stacked=True,
    title ="Proportion of level A skill groups\nfor pre- and post- COVID job adverts",
    ylabel="",
    xlabel="",
    color = levela_cols)
plt.legend(bbox_to_anchor=(1.05, 1))

plt.savefig('outputs/skills_taxonomy_application/covid_application/covid_prop_a.pdf',bbox_inches='tight')


# %%
# Another view type
prop_level_a_covid.unstack().plot.bar(
    stacked=False,
    title ="Proportion of level A skill groups for pre- and post- COVID job adverts",
    ylabel="",
    xlabel="",
    figsize=(10,4), color = levela_cols)
plt.legend(bbox_to_anchor=(1.05, 1))

plt.savefig('outputs/skills_taxonomy_application/covid_application/covid_prop_a_T.pdf',bbox_inches='tight')



# %%
prop_level_a_covid.reset_index().groupby(['level_1','covid']).apply(
    lambda x: x['Hierarchy level A name'].iloc[0]
).unstack().plot.barh(stacked=False,
                     title ="Proportion of level A skill groups for pre- and post- COVID job adverts",
                     figsize=(8,3),
                     color = [nesta_grey, 'black']
                    )
plt.legend(bbox_to_anchor=(1.05, 1))
plt.ylabel("")
plt.savefig('outputs/skills_taxonomy_application/covid_application/covid_prop_a_T2.pdf',bbox_inches='tight')


# %%
sentence_data_with_meta_filter = sentence_data_with_meta[sentence_data_with_meta[
    'covid']=='Post-COVID']
level_a_prop_post_covid = sentence_data_with_meta_filter['Hierarchy level A name'].value_counts()/len(sentence_data_with_meta_filter)

sentence_data_precovid = sentence_data_with_meta[sentence_data_with_meta[
    'covid']=='Pre-COVID']
level_a_prop_pre_covid = sentence_data_precovid['Hierarchy level A name'].value_counts()/len(sentence_data_precovid)

df = pd.concat([
    pd.DataFrame(level_a_prop_pre_covid).rename(
        columns={'Hierarchy level A name':'Proportion of level A skill group in pre-covid job adverts only'}),
    pd.DataFrame(level_a_prop_post_covid).rename(
        columns={'Hierarchy level A name':'Proportion of level A skill group in post-covid job adverts only'})
], axis=1)
df['Increase from before to after COVID'] = df['Proportion of level A skill group in post-covid job adverts only']/df['Proportion of level A skill group in pre-covid job adverts only']
df.round(3).to_csv('outputs/skills_taxonomy_application/covid_application/covid_prepost_leva.csv')



# %%
prop_level_a_covid.reset_index().groupby(['level_1','covid']).apply(
    lambda x: x['Hierarchy level A name'].iloc[0]
)

# %% [markdown]
# ## pre vs post covid quotients

# %%
# level_b_prop_all = sentence_data_with_meta['Hierarchy level B name'].value_counts()/len(sentence_data_with_meta)

sentence_data_with_meta_filter = sentence_data_with_meta[sentence_data_with_meta[
    'covid']=='Post-COVID']
level_b_prop_post_covid = sentence_data_with_meta_filter['Hierarchy level B name'].value_counts()/len(sentence_data_with_meta_filter)

sentence_data_precovid = sentence_data_with_meta[sentence_data_with_meta[
    'covid']=='Pre-COVID']
level_b_prop_pre_covid = sentence_data_precovid['Hierarchy level B name'].value_counts()/len(sentence_data_precovid)


covid_quotient = level_b_prop_post_covid/level_b_prop_pre_covid
covid_quotient = covid_quotient[pd.notnull(covid_quotient)].sort_values(ascending=True)

covid_quotient.plot.barh(figsize=(8,10),
                        ylabel='',
                        xlabel='Level B hierarchy',
                       title = 'Post-COVID compared to pre-COVID proportions of level B skill groups',
                         color = nesta_orange)
plt.axvline(1, color="black")

plt.savefig('outputs/skills_taxonomy_application/covid_application/covid_prepost_levb.pdf',bbox_inches='tight')


# %%
low_covid_quotient = covid_quotient.sort_values()[0:10].index.tolist()
high_covid_quotient = covid_quotient.sort_values()[-10:].index.tolist()

# %%
df = pd.concat([
    pd.DataFrame(level_b_prop_pre_covid[low_covid_quotient+high_covid_quotient]).rename(
        columns={'Hierarchy level B name':'Proportion of level B skill group in pre-covid job adverts only'}),
    pd.DataFrame(level_b_prop_post_covid[low_covid_quotient+high_covid_quotient]).rename(
        columns={'Hierarchy level B name':'Proportion of level B skill group in post-covid job adverts only'})
], axis=1)
df['Increase from before to after COVID'] = df['Proportion of level B skill group in post-covid job adverts only']/df['Proportion of level B skill group in pre-covid job adverts only']
df.round(3).to_csv('outputs/skills_taxonomy_application/covid_application/covid_prepost_levb.csv')



# %%
df.round(3)

# %%
## Level C covid

# %%
level_c_prop_post_covid = sentence_data_with_meta_filter['Hierarchy level C name'].value_counts()/len(sentence_data_with_meta_filter)
level_c_prop_pre_covid = sentence_data_precovid['Hierarchy level C name'].value_counts()/len(sentence_data_precovid)

covid_quotient_levc = level_c_prop_post_covid/level_c_prop_pre_covid
covid_quotient_levc = covid_quotient_levc[pd.notnull(covid_quotient_levc)].sort_values(ascending=True)


# %%
low_covid_quotient_levc = covid_quotient_levc.sort_values()[0:10].index.tolist()
high_covid_quotient_levc = covid_quotient_levc.sort_values()[-10:].index.tolist()

# %%
df = pd.concat([
    pd.DataFrame(level_c_prop_pre_covid[low_covid_quotient_levc+high_covid_quotient_levc]).rename(
        columns={'Hierarchy level C name':'Proportion of level C skill group in pre-covid job adverts only'}),
    pd.DataFrame(level_c_prop_post_covid[low_covid_quotient_levc+high_covid_quotient_levc]).rename(
        columns={'Hierarchy level C name':'Proportion of level C skill group in post-covid job adverts only'})
], axis=1)
df['Increase from before to after COVID'] = df['Proportion of level C skill group in post-covid job adverts only']/df['Proportion of level C skill group in pre-covid job adverts only']
df.round(3).to_csv('outputs/skills_taxonomy_application/covid_application/covid_prepost_levc.csv')



# %%
df.round(3)

# %%
