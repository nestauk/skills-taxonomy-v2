# -*- coding: utf-8 -*-
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
# # The geographical application
#
# - We have 232394 skill sentences
# - We have 18893 skills
# - We have skills identified from 107434 unique job adverts
# - Of these job adverts 104801 have location information
# - We don't have all skills from each job advert
#
#

# %%
# cd ../../..

# %%
from geopandas import GeoDataFrame
from shapely.geometry import Point
from bokeh.palettes import Turbo256
from skills_taxonomy_v2.getters.s3_data import load_s3_data

# %%
from collections import Counter, defaultdict
import random
from tqdm import tqdm
import json

import plotly.express as px
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import boto3
import geopandas as gpd

# %%
bucket_name = "skills-taxonomy-v2"
s3 = boto3.resource("s3")

# %% [markdown]
# ### Load data

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
sentence_data = pd.DataFrame(sentence_data)
sentence_data = sentence_data[sentence_data["Cluster number"] != -1]

# %%
sentence_data["Cluster number"].nunique()

# %%
# Manual level A names
with open("skills_taxonomy_v2/utils/2021.09.06_level_a_rename_dict.json", "r") as f:
    level_a_rename_dict = json.load(f)

# %%
# Add hierarchy information to this df
sentence_data["Hierarchy level A name"] = sentence_data["Cluster number"].apply(
    lambda x: level_a_rename_dict[str(skill_hierarchy[str(x)]["Hierarchy level A"])]
)
sentence_data["Hierarchy level B name"] = sentence_data["Cluster number"].apply(
    lambda x: skill_hierarchy[str(x)]["Hierarchy level B name"]
)
sentence_data["Hierarchy level C name"] = sentence_data["Cluster number"].apply(
    lambda x: skill_hierarchy[str(x)]["Hierarchy level C name"]
)


# %%
sentence_data.head(1)

# %%
job_ids = set(sentence_data["job id"].tolist())

# %%
len(job_ids)

# %%
sentence_data["Hierarchy level B name"].nunique()

# %%
sentence_data["Hierarchy level C name"].nunique()

# %% [markdown]
# ## Import locations for jobs in this data
# This was created in `skills_taxonomy_v2/analysis/skills_taxonomy_application/locations_to_nuts2.py`
#
# There are 107434 job IDs but not all have locations.
# Should be about 104801 long.

# %%
jobs_with_nuts_codes = load_s3_data(
    s3,
    bucket_name,
    "outputs/tk_data_analysis/metadata_location/2021.09.14_jobs_with_nuts_codes.json",
)


# %%
len(jobs_with_nuts_codes)

# %%
nuts_uniq = list()
for k, v in jobs_with_nuts_codes.items():
    if len(v) == 6:
        nuts_uniq.append(v[5])

# %%
Counter(nuts_uniq)

# %% [markdown]
# ## Merge the metadata with the sentence data

# %%
job_id_loc_dict = jobs_with_nuts_codes

# %%
sentence_data_with_meta = sentence_data.copy()[
    sentence_data["job id"].isin(job_id_loc_dict)
]
print(len(sentence_data_with_meta))
sentence_data_with_meta["long-lat"] = sentence_data_with_meta["job id"].apply(
    lambda x: job_id_loc_dict.get(x)[1]
)
sentence_data_with_meta["latitude"] = sentence_data_with_meta["job id"].apply(
    lambda x: float(job_id_loc_dict.get(x)[1].split(",")[0])
    if job_id_loc_dict.get(x)[1]
    else None
)
sentence_data_with_meta["longitude"] = sentence_data_with_meta["job id"].apply(
    lambda x: float(job_id_loc_dict.get(x)[1].split(",")[1])
    if job_id_loc_dict.get(x)[1]
    else None
)
sentence_data_with_meta["location_name"] = sentence_data_with_meta["job id"].apply(
    lambda x: job_id_loc_dict.get(x)[0]
)
sentence_data_with_meta["region"] = sentence_data_with_meta["job id"].apply(
    lambda x: job_id_loc_dict.get(x)[2]
)
sentence_data_with_meta["subregion"] = sentence_data_with_meta["job id"].apply(
    lambda x: job_id_loc_dict.get(x)[3]
)
sentence_data_with_meta["NUTs region"] = sentence_data_with_meta["job id"].apply(
    lambda x: job_id_loc_dict.get(x)[5] if len(job_id_loc_dict.get(x)) == 6 else None
)

# %%

sentence_data_with_meta.head(2)

# %%
nesta_orange = [255 / 255, 90 / 255, 0 / 255]

# %%

# %%
levela_cols = []
for i in range(0, 7):
    levela_cols.append(Turbo256[i * round(len(Turbo256) / 7)])
levela_cols = levela_cols[0:6]

# %% [markdown]
# ## Number of data points per location

# %%
sentence_data_with_meta["NUTs region"].value_counts().plot.bar(
    xlabel="NUTs region",
    ylabel="Number of data points (skill sentences)",
    title="Number of data points by NUTs region",
    color=nesta_orange,
)

# %%
sentence_data_with_meta["NUTs region"].value_counts()

# %% [markdown]
# ## Plot location proportions of hier A

# %%
prop_level_a_region = sentence_data_with_meta.groupby("region")[
    "Hierarchy level A name"
].apply(lambda x: x.value_counts() / len(x))

prop_level_a_region.unstack().plot.barh(
    stacked=True,
    title="Proportion of skill types in each region",
    ylabel="",
    xlabel="",
    color=levela_cols,
)
plt.legend(bbox_to_anchor=(1.05, 1))

plt.savefig(
    "outputs/skills_taxonomy_application/region_application/region_levela_props.pdf",
    bbox_inches="tight",
)


# %%
prop_level_a_nuts = sentence_data_with_meta.groupby("NUTs region")[
    "Hierarchy level A name"
].apply(lambda x: x.value_counts() / len(x))

prop_level_a_nuts.unstack().plot.barh(
    stacked=True,
    title="Proportion of skill types in each NUTs region",
    ylabel="",
    xlabel="",
    color=levela_cols,
)
plt.legend(bbox_to_anchor=(1.05, 1))

plt.savefig(
    "outputs/skills_taxonomy_application/region_application/nuts_levela_props.pdf",
    bbox_inches="tight",
)


# %%
# Another view type
prop_level_a_region.unstack().plot.bar(
    stacked=False,
    title="Proportion of skill types in each region",
    ylabel="",
    xlabel="",
    figsize=(10, 4),
    color=levela_cols,
)
plt.legend(bbox_to_anchor=(1.05, 1))

plt.savefig(
    "outputs/skills_taxonomy_application/region_application/region_levela_props_t.pdf",
    bbox_inches="tight",
)


# %%
prop_level_a_nuts.reset_index().groupby(["level_1", "NUTs region"]).apply(
    lambda x: x["Hierarchy level A name"].iloc[0]
).unstack().plot.bar(
    stacked=False, title="Proportion of skill types in each region", figsize=(10, 4)
)
plt.legend(bbox_to_anchor=(1.05, 1))

plt.savefig(
    "outputs/skills_taxonomy_application/region_application/nuts_levela_props_t.pdf",
    bbox_inches="tight",
)


# %% [markdown]
# ## Proportion of each level A in a box plot

# %%
props = dict(boxes="orange", whiskers="black", medians="black", caps="black")
axes = prop_level_a_nuts.reset_index().boxplot(
    by=["level_1"],
    column=["Hierarchy level A name"],
    vert=False,
    figsize=(10, 4),
    color=props,
    patch_artist=True,
)
axes.set_title("Spread of proportions of skill types over all NUTs regions")
axes.set_xlabel("")
plt.suptitle("")
plt.savefig(
    "outputs/skills_taxonomy_application/region_application/nuts_levela_props_box.pdf",
    bbox_inches="tight",
)


# %%
name2num_dict = {v: int(k) for k, v in level_a_rename_dict.items()}

# %%
# Only the ones with the biggest spread:
level_a_spread = [
    "Information technology and languages",
    "Teaching and care",
    "Safety, finance, maintenance and service",
    "Business administration and management",
]
fig, axs = plt.subplots(1, 4, figsize=(20, 3))

coli = 0
rowi = 0
for i, level_a_name in enumerate(level_a_spread):
    color = levela_cols[name2num_dict[level_a_name]]
    df = prop_level_a_nuts.reset_index()
    sort_props = (
        df[df["level_1"] == level_a_name]
        .set_index("NUTs region")
        .sort_values(by="Hierarchy level A name")
    )
    sort_props.plot.bar(ax=axs[i], title=level_a_name, legend=False, color=color)
    coli += 1
fig.subplots_adjust(hspace=1.3)

plt.savefig(
    "outputs/skills_taxonomy_application/region_application/nuts_levela_props_separate_top.pdf",
    bbox_inches="tight",
)


# %%
fig, axs = plt.subplots(2, 3, figsize=(20, 10))

coli = 0
rowi = 0
for i, level_a_name in enumerate(
    sentence_data_with_meta["Hierarchy level A name"].unique()
):
    color = levela_cols[name2num_dict[level_a_name]]
    if i != 0 and i % 3 == 0:
        rowi += 1
        coli = 0
    df = prop_level_a_nuts.reset_index()
    sort_props = (
        df[df["level_1"] == level_a_name]
        .set_index("NUTs region")
        .sort_values(by="Hierarchy level A name")
    )
    sort_props.plot.bar(
        ax=axs[rowi, coli], title=level_a_name, legend=False, color=color
    )
    coli += 1
fig.subplots_adjust(hspace=1.3)
plt.savefig(
    "outputs/skills_taxonomy_application/region_application/nuts_levela_props_separate_all.pdf",
    bbox_inches="tight",
)


# %% [markdown]
# ## Plot circles on map for
# - How many job avderts
# - How mnay from the 4 most diverging level A code

# %%


# %%
df = sentence_data_with_meta.groupby("NUTs region")[["latitude", "longitude"]].mean()
# Add the number of unique job ids
df = pd.concat(
    [df, sentence_data_with_meta.groupby(["NUTs region"])["job id"].nunique()], axis=1
)
# Add the proportions of each level A
df = pd.concat(
    [
        df,
        prop_level_a_nuts.reset_index().pivot(
            index="NUTs region", columns="level_1", values="Hierarchy level A name"
        ),
    ],
    axis=1,
)

# %%
# Normalise the proportions, otherwise they are all v similar on plot
for col_name in sentence_data_with_meta["Hierarchy level A name"].unique():
    df[f"{col_name} - normalised"] = (df[col_name] - df[col_name].min()) / (
        df[col_name].max() - df[col_name].min()
    )

# %%
geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
gdf = GeoDataFrame(df, geometry=geometry)

# %%
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

fig, ax_map = plt.subplots(2, 3, figsize=(20, 10))

coli = 0
rowi = 0
for i, col_name in enumerate(
    sentence_data_with_meta["Hierarchy level A name"].unique()
):
    color = levela_cols[name2num_dict[col_name]]
    if name2num_dict[col_name] == 0:
        color = "white"
    if i != 0 and i % 3 == 0:
        rowi += 1
        coli = 0
    world[world.name == "United Kingdom"].plot(ax=ax_map[rowi, coli], color="black")
    gdf.plot(
        ax=ax_map[rowi, coli],
        marker="o",
        color=color,
        markersize=gdf[f"{col_name} - normalised"] * 500,
        alpha=0.8,
    )
    ax_map[rowi, coli].set_title(f"{col_name}")
    ax_map[rowi, coli].set_axis_off()

    coli += 1

plt.savefig(
    "outputs/skills_taxonomy_application/region_application/nuts_levela_props_maps.pdf",
    bbox_inches="tight",
)


# %%
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

fig, ax_map = plt.subplots(1, 2, figsize=(15, 5))

world[world.name == "United Kingdom"].plot(ax=ax_map[0], color="black")
gdf.plot(
    ax=ax_map[0],
    marker="o",
    color=nesta_orange,
    markersize=gdf["job id"] / 10,
    alpha=0.6,
)
ax_map[0].set_title("Number of job adverts in sample by region")
ax_map[0].set_axis_off()

sentence_data_with_meta["NUTs region"].value_counts().plot.barh(
    xlabel="", ylabel="", title="", color=nesta_orange, ax=ax_map[1]
)
plt.savefig(
    "outputs/skills_taxonomy_application/region_application/nuts_numbers_maps.pdf",
    bbox_inches="tight",
)


# %% [markdown]
# ## London vs the rest for level B

# %%
sum(sentence_data_with_meta["NUTs region"].notna())

# %%
sum(sentence_data_with_meta["subregion"].notna())

# %%
sentence_data_rest = sentence_data_with_meta[
    sentence_data_with_meta["NUTs region"] != "Greater London"
]
level_b_prop_rest = sentence_data_rest["Hierarchy level B name"].value_counts() / len(
    sentence_data_rest
)

sentence_data_with_meta_filter = sentence_data_with_meta[
    sentence_data_with_meta["subregion"] == "Greater London"
]
level_b_prop_london = sentence_data_with_meta_filter[
    "Hierarchy level B name"
].value_counts() / len(sentence_data_with_meta_filter)

london_quotient = level_b_prop_london / level_b_prop_rest

london_quotient = london_quotient[pd.notnull(london_quotient)].sort_values(
    ascending=True
)

london_quotient.plot.barh(
    figsize=(8, 10),
    ylabel="London quotient",
    xlabel="Level B hierarchy",
    title="Greater London quotient",
    color=nesta_orange,
)
plt.axvline(1, color="black")

plt.savefig(
    "outputs/skills_taxonomy_application/region_application/london_quotient_levb.pdf",
    bbox_inches="tight",
)

sentence_data_rest = sentence_data_with_meta[
    sentence_data_with_meta["NUTs region"] != "Greater London"
]
level_a_prop_rest = sentence_data_rest["Hierarchy level A name"].value_counts() / len(
    sentence_data_rest
)

sentence_data_with_meta_filter = sentence_data_with_meta[
    sentence_data_with_meta["subregion"] == "Greater London"
]
level_a_prop_london = sentence_data_with_meta_filter[
    "Hierarchy level A name"
].value_counts() / len(sentence_data_with_meta_filter)

london_quotient = level_a_prop_london / level_a_prop_rest
london_quotient = london_quotient[pd.notnull(london_quotient)].sort_values(
    ascending=True
)

london_quotient.plot.barh(
    figsize=(8, 4),
    ylabel="London quotient",
    xlabel="Level A hierarchy",
    title="Greater London quotient",
    color=[levela_cols[name2num_dict[i]] for i in london_quotient.keys()],
)
plt.axvline(1, color="black")

plt.savefig(
    "outputs/skills_taxonomy_application/region_application/london_quotient_leva.pdf",
    bbox_inches="tight",
)

# %% [markdown]
# ## Other outliers

# %%
# The North East has a much higher demand for “Teaching and care”.

region = "North East (England)"

sentence_data_region = sentence_data_with_meta[
    sentence_data_with_meta["NUTs region"] == region
]
level_b_prop_region = sentence_data_region[
    "Hierarchy level B name"
].value_counts() / len(sentence_data_region)

sentence_data_rest = sentence_data_with_meta[
    sentence_data_with_meta["NUTs region"] != region
]
level_b_prop_rest = sentence_data_rest["Hierarchy level B name"].value_counts() / len(
    sentence_data_rest
)

region_quotient = level_b_prop_region / level_b_prop_rest
region_quotient = region_quotient[pd.notnull(region_quotient)].sort_values(
    ascending=True
)

region_quotient

# %%
sentence_data[sentence_data["Hierarchy level B name"] == "clinical-patients-nursing"][
    "Hierarchy level C name"
].value_counts()


# %%
# Wales has a particular low demand for “Customer service and marketing” skills.
region = "Wales"

sentence_data_region = sentence_data_with_meta[
    sentence_data_with_meta["NUTs region"] == region
]
level_b_prop_region = sentence_data_region[
    "Hierarchy level B name"
].value_counts() / len(sentence_data_region)

sentence_data_rest = sentence_data_with_meta[
    sentence_data_with_meta["NUTs region"] != region
]
level_b_prop_rest = sentence_data_rest["Hierarchy level B name"].value_counts() / len(
    sentence_data_rest
)

region_quotient = level_b_prop_region / level_b_prop_rest
region_quotient = region_quotient[pd.notnull(region_quotient)].sort_values(
    ascending=True
)

region_quotient

# %%

region = "Northern Ireland"

sentence_data_region = sentence_data_with_meta[
    sentence_data_with_meta["NUTs region"] == region
]
level_b_prop_region = sentence_data_region[
    "Hierarchy level B name"
].value_counts() / len(sentence_data_region)

sentence_data_rest = sentence_data_with_meta[
    sentence_data_with_meta["NUTs region"] != region
]
level_b_prop_rest = sentence_data_rest["Hierarchy level B name"].value_counts() / len(
    sentence_data_rest
)


region_quotient = level_b_prop_region / level_b_prop_rest
region_quotient = region_quotient[pd.notnull(region_quotient)].sort_values(
    ascending=True
)

region_quotient

# %%
region = "East Midlands (England)"

sentence_data_region = sentence_data_with_meta[
    sentence_data_with_meta["NUTs region"] == region
]
level_b_prop_region = sentence_data_region[
    "Hierarchy level B name"
].value_counts() / len(sentence_data_region)

sentence_data_rest = sentence_data_with_meta[
    sentence_data_with_meta["NUTs region"] != region
]
level_b_prop_rest = sentence_data_rest["Hierarchy level B name"].value_counts() / len(
    sentence_data_rest
)


region_quotient = level_b_prop_region / level_b_prop_rest
region_quotient = region_quotient[pd.notnull(region_quotient)].sort_values(
    ascending=True
)

region_quotient

# %%
sentence_data[sentence_data["Hierarchy level B name"] == "driving-licence-vehicle"][
    "Hierarchy level C name"
].value_counts()

# %%

sentence_data[sentence_data["Hierarchy level B name"] == "stock-contractors-warehouse"][
    "Hierarchy level C name"
].value_counts()


# %%
region = "London"

sentence_data_region = sentence_data_with_meta[
    sentence_data_with_meta["NUTs region"] == region
]
level_b_prop_region = sentence_data_region[
    "Hierarchy level B name"
].value_counts() / len(sentence_data_region)

sentence_data_rest = sentence_data_with_meta[
    sentence_data_with_meta["NUTs region"] != region
]
level_b_prop_rest = sentence_data_rest["Hierarchy level B name"].value_counts() / len(
    sentence_data_rest
)


region_quotient = level_b_prop_region / level_b_prop_rest
region_quotient = region_quotient[pd.notnull(region_quotient)].sort_values(
    ascending=True
)

region_quotient

# %%
