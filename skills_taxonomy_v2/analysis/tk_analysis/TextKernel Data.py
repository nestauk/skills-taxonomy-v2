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
# ## Some metadata about the TextKernel dataset
#
# We already processed metadata from the textkernel dataset in `get_bulk_metadata.py` - this script saves out the textkernel data into more useful (and smaller) files rather than 686 massive files containing all the data. The files saved are:
# 1. `metadata_file/` : 13 json files of {'job id': 'file name'} useful for knowing which file a job advert is in.
# 2. `metadata_date/` : 13 json files of {'job id': ['date', 'expiration_date']}
# 3. `metadata_job/` : 13 json files of {'job id': ['job_title', 'organization_industry']}
# 4. `metadata_location/` : 13 json files of {'job id': ['location_name', 'location_coordinates', 'region', 'subregion']}
# 5. `metadata_meta/` : 13 json files of {'job id': ['source_website', 'language']}
#
# Note: not every job advert has all the metadata.
#
# These files take quite a while to load (70 secs per file)
#
# - How many files
# - How many job adverts
# - How many job adverts per year

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
# ## How many files

# %%
tk_input_dir = "inputs/data/textkernel-files/"
tk_data_paths = get_s3_data_paths(s3, bucket_name, tk_input_dir, file_types=["*.json*"])
len(tk_data_paths)

# %% [markdown]
# ## How many job adverts

# %%
num_tk_job_ads = 0
for file_name in tqdm(range(0, 13)):
    file_loc_dict = load_s3_data(
        s3, bucket_name, f"outputs/tk_data_analysis/metadata_file/{file_name}.json"
    )
    num_tk_job_ads += len(file_loc_dict)

# %%
print(num_tk_job_ads)

# %% [markdown]
# ## Dates
# 'date', 'expiration_date'

# %%
tk_dates = []
for file_name in tqdm(range(0, 13)):
    file_date_dict = load_s3_data(
        s3, bucket_name, f"outputs/tk_data_analysis/metadata_date/{file_name}.json"
    )
    tk_dates.extend([f[0] for f in file_date_dict.values()])

print(len(tk_dates))

# %%
df = pd.DataFrame(tk_dates)
df["date"] = pd.to_datetime(df[0], format="%Y-%m-%d")

# %%
len(df)

# %%
num_dates_null = sum(pd.isnull(df[0]))
num_dates_null

# %%
df_date = df[pd.notnull(df[0])]

# %%
len(df_date)

# %%
df_date["year"] = pd.DatetimeIndex(df_date[0]).year
df_date["month"] = pd.DatetimeIndex(df_date[0]).month

# %%
year_month_counts = df_date.groupby(["year", "month"])[0].count()

# %%
year_month_counts = year_month_counts.sort_index().reset_index()
year_month_counts["year/month"] = (
    year_month_counts[["year", "month"]].astype(str).agg("/".join, axis=1)
)
year_month_counts

# %%
# Add a row for the None date counts and save
pd.concat(
    [
        year_month_counts,
        pd.DataFrame(
            [{"year": None, "month": None, 0: num_dates_null, "year/month": None}]
        ),
    ],
    ignore_index=True,
    axis=0,
).to_csv("outputs/tk_analysis/all_tk_year_month_counts.csv")

# %%
ax = year_month_counts.plot(
    x="year/month",
    y=0,
    xlabel="Date of job advert",
    ylabel="Number of job adverts",
    c="k",
)
ax.figure.savefig("outputs/tk_analysis/job_ad_date.pdf", bbox_inches="tight")

# %%
ax = (
    df_date["year"]
    .value_counts()
    .sort_index()
    .plot.bar(
        xlabel="Year of job advert",
        ylabel="Number of job adverts",
        color=[255 / 255, 90 / 255, 0 / 255],
    )
)
ax.figure.savefig("outputs/tk_analysis/job_ad_year.pdf", bbox_inches="tight")

# %% [markdown]
# ## Location

# %%
tk_region = []
tk_subregion = []
for file_name in tqdm(range(0, 13)):
    file_dict = load_s3_data(
        s3, bucket_name, f"outputs/tk_data_analysis/metadata_location/{file_name}.json"
    )
    tk_region.extend([f[2] for f in file_dict.values() if f])
    tk_subregion.extend([f[3] for f in file_dict.values() if f])

print(len(tk_region))
print(len(tk_subregion))

# %%
print(len(set(tk_region)))
print(len(set(tk_subregion)))

# %%
count_region_df = pd.DataFrame.from_dict(Counter(tk_region), orient="index")
count_region_df

# %%
count_region_df.to_csv("outputs/tk_analysis/all_tk_regions_counts.csv")

# %%
print(count_region_df[0].sum())
count_region_df = count_region_df[pd.notnull(count_region_df.index)]
print(count_region_df[0].sum())

# %%
count_region_df.loc["England"][0] / count_region_df[0].sum()

# %%
ax = count_region_df.sort_values(by=[0], ascending=False).plot.bar(
    xlabel="Region of job advert",
    ylabel="Number of job adverts",
    color=[255 / 255, 90 / 255, 0 / 255],
    legend=False,
)
ax.figure.savefig("outputs/tk_analysis/job_ad_region.pdf", bbox_inches="tight")

# %%
count_subregion_df = pd.DataFrame.from_dict(Counter(tk_subregion), orient="index")

# %%
count_subregion_df.to_csv("outputs/tk_analysis/all_tk_subregions_counts.csv")

# %%
print(count_subregion_df[0].sum())
count_subregion_df = count_subregion_df[pd.notnull(count_subregion_df.index)]
print(count_subregion_df[0].sum())

# %%
ax = count_subregion_df.sort_values(by=[0], ascending=False)[0:50].plot.bar(
    xlabel="Subregion of job advert",
    ylabel="Number of job adverts",
    title="Job advert subregions for 50 most common subregions",
    color=[255 / 255, 90 / 255, 0 / 255],
    legend=False,
    figsize=(12, 4),
    fontsize=8,
)
ax.figure.savefig("outputs/tk_analysis/job_ad_subregion.pdf", bbox_inches="tight")

# %%
count_subregion_df.loc["Greater London"][0] / count_subregion_df[0].sum()

# %% [markdown]
# ## Plots together

# %%
plt.figure(figsize=(12, 8))

ax3 = plt.subplot(212)
count_subregion_df.sort_values(by=[0], ascending=False)[0:50].plot.bar(
    xlabel="Subregion of job advert for 50 most common subregions",
    ylabel="Number of job adverts",
    color=[255 / 255, 90 / 255, 0 / 255],
    legend=False,
    fontsize=8,
    ax=ax3,
    layout="tight",
)

ax1 = plt.subplot(221)
df_date["year"].value_counts().sort_index().plot.bar(
    xlabel="Year of job advert",
    ylabel="Number of job adverts",
    color=[255 / 255, 90 / 255, 0 / 255],
    ax=ax1,
    layout="tight",
)

ax2 = plt.subplot(222)
count_region_df.sort_values(by=[0], ascending=False).plot.bar(
    xlabel="Region of job advert",
    ylabel="Number of job adverts",
    color=[255 / 255, 90 / 255, 0 / 255],
    legend=False,
    ax=ax2,
    layout="tight",
)

plt.tight_layout()
plt.savefig("outputs/tk_analysis/job_ad_together.pdf", bbox_inches="tight")

# %%
