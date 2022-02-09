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

# %%
## The polygon data for each nuts2
import os
from skills_taxonomy_v2 import PROJECT_DIR
from urllib.request import urlretrieve
from zipfile import ZipFile
from shapely.geometry import Point

shape_url = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/download/ref-nuts-2021-20m.geojson.zip"
nuts_file = "NUTS_RG_20M_2021_4326_LEVL_1.geojson"
shapefile_path = "/inputs/shapefiles/"
    
full_shapefile_path = str(PROJECT_DIR) + shapefile_path
if not os.path.isdir(full_shapefile_path):
    os.mkdir(full_shapefile_path)

zip_path, _ = urlretrieve(shape_url)
with ZipFile(zip_path, "r") as zip_files:
    for zip_names in zip_files.namelist():
        if zip_names == nuts_file:
            zip_files.extract(zip_names, path=full_shapefile_path)
            nuts_geo = gpd.read_file(full_shapefile_path + nuts_file)
            nuts_geo = nuts_geo[nuts_geo["CNTR_CODE"] == "UK"].reset_index(
                drop=True
            )
nuts2polygons = nuts_geo[['NUTS_NAME', 'geometry']]
nuts2polygons.index = nuts2polygons['NUTS_NAME']
nuts2polygons_dict = {k:v['geometry'] for k,v in pd.DataFrame(nuts2polygons).to_dict(orient='index').items()}
nuts2polygons_dict

# %% [markdown]
# ### Load data

# %%
hier_date = '2022.01.21'
skills_date = '2022.01.14'

# %%
skill_hierarchy_file = f"outputs/skills_taxonomy/{hier_date}_skills_hierarchy.json"
skill_hierarchy = load_s3_data(s3, bucket_name, skill_hierarchy_file)

# %%
sentence_data = load_s3_data(
    s3,
    bucket_name,
    f"outputs/skills_extraction/extracted_skills/{skills_date}_sentences_skills_data_lightweight.json",
)


# %%
sentence_data = pd.DataFrame(
            sentence_data,
            columns=['job id', 'sentence id',  'Cluster number']
            )
sentence_data = sentence_data[sentence_data["Cluster number"] >= 0]


# %%
sentence_data["Cluster number"].nunique()

# %% [markdown]
# ### Add the duplicated sentences

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
del dupe_job_ids, skill_job_ids_with_dupes_list
# Get the words id for the existing deduplicated sentence data
sentence_data_ehcd = sentence_data.merge(skill_job_ids_with_dupes_df, how='left', on=['job id', 'sentence id'])
skill_sent_word_ids = set(sentence_data_ehcd['words id'].unique())
len(skill_sent_word_ids)
del skill_job_ids_with_dupes_df

# %%
# Get all the job id+sent id for the duplicates with these word ids
dupe_sentence_data = []
for job_id, s_w_list in tqdm(dupe_words_id.items()):
    for (word_id, sent_id) in s_w_list:
        if word_id in skill_sent_word_ids:
            cluster_num = sentence_data_ehcd[sentence_data_ehcd['words id']==word_id].iloc[0]['Cluster number']
            dupe_sentence_data.append([job_id, sent_id, cluster_num])
dupe_sentence_data_df = pd.DataFrame(dupe_sentence_data, columns = ['job id', 'sentence id', 'Cluster number'])           
del sentence_data_ehcd, dupe_sentence_data

# %%
# Add new duplicates to sentence data
print(len(sentence_data))
sentence_data = pd.concat([sentence_data, dupe_sentence_data_df])
sentence_data.drop_duplicates(inplace=True)
sentence_data.reset_index(inplace=True)
print(len(sentence_data))

# %% [markdown]
# ### Add hierarchy information to this df

# %%

sentence_data["Hierarchy level A name"] = sentence_data["Cluster number"].apply(
    lambda x: skill_hierarchy[str(x)]["Hierarchy level A name"]
)
sentence_data["Hierarchy level B name"] = sentence_data["Cluster number"].apply(
    lambda x: skill_hierarchy[str(x)]["Hierarchy level B name"]
)
sentence_data["Hierarchy level C name"] = sentence_data["Cluster number"].apply(
    lambda x: skill_hierarchy[str(x)]["Hierarchy level C name"]
)


# %%
sentence_data["Hierarchy level A name"] = sentence_data["Hierarchy level A name"].apply(lambda x: x.replace("Cognitative","Cognitive"))

# %%
sentence_data.head(1)

# %%
job_ids = set(sentence_data["job id"].tolist())

# %%
len(job_ids)

# %%
sentence_data["Hierarchy level A name"].nunique()

# %%
sentence_data["Hierarchy level B name"].nunique()

# %%
sentence_data["Hierarchy level C name"].nunique()

# %% [markdown]
# ## Job title data

# %%
job_titles = load_s3_data(s3, bucket_name, "outputs/tk_data_analysis_new_method/metadata_job/14.01.22/sample_filtered.json")

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
    "outputs/tk_data_analysis_new_method/metadata_location/14.01.22/jobs_with_nuts_codes.json",
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
sentence_data_with_meta["Hierarchy level A name"] = sentence_data_with_meta["Hierarchy level A name"].apply(lambda x: x.replace("Cognitative","Cognitive"))

# %%

sentence_data_with_meta.head(2)

# %%
nesta_orange = [255 / 255, 90 / 255, 0 / 255]

# %%
from bokeh.models import LinearColorMapper

num_lev_as = sentence_data['Hierarchy level A name'].nunique()
levela_cols = []
for i in range(0, num_lev_as):
    levela_cols.append(Turbo256[i * round(len(Turbo256) / num_lev_as)])
random.seed(15)
random.shuffle(levela_cols)
levela_cols[10]="grey"
levela_cols

# %%
level_a_mapper = {}
for skill in skill_hierarchy.values():
    level_a_mapper[skill['Hierarchy level A name'].replace("Cognitative", "Cognitive")] = skill['Hierarchy level A']
level_a_mapper

# %%
levela_cols_dict = {level_a_name:levela_cols[level_a_num] for level_a_name, level_a_num in level_a_mapper.items()}
levela_cols_dict

# %%
level_a_mapper_names = list(level_a_mapper.keys())
level_a_mapper_names.sort()
levela_cols = [levela_cols[level_a_mapper[level_a_name]] for level_a_name in level_a_mapper_names]

# %%
levela_cols

# %%
# When you plot level a details do them from most common to least, with misc at the end
plot_level_a_order = [k[0] for k in Counter(sentence_data_with_meta["Hierarchy level A name"]).most_common() if k[0]!='Misc']
plot_level_a_order.append('Misc')
plot_level_a_order

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
len(sentence_data_with_meta)

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
    f"outputs/skills_taxonomy_application/region_application/{hier_date}/region_levela_props.pdf",
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
    f"outputs/skills_taxonomy_application/region_application/{hier_date}/nuts_levela_props.pdf",
    bbox_inches="tight",
)


# %%
(prop_level_a_nuts.unstack().round(3)*100).to_csv(
f"outputs/skills_taxonomy_application/region_application/{hier_date}/nuts_levela_props.csv")

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
    f"outputs/skills_taxonomy_application/region_application/{hier_date}/region_levela_props_t.pdf",
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
    f"outputs/skills_taxonomy_application/region_application/{hier_date}/nuts_levela_props_t.pdf",
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
    f"outputs/skills_taxonomy_application/region_application/{hier_date}/nuts_levela_props_box.pdf",
    bbox_inches="tight",
)


# %%
name2num_dict = level_a_mapper

# %%
# Only the ones with the biggest spread:
level_a_spread = [
    'Manufacturing, engineering and physical skills',
    'Food, cleaning and safety',
    'Attitudes, communication and social skills',
    'Digital and technology',
    'Management, business processes and administration'
]
fig, axs = plt.subplots(1, 5, figsize=(20, 3))

coli = 0
rowi = 0
for i, level_a_name in enumerate(level_a_spread):
    color = levela_cols_dict[level_a_name]
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
    f"outputs/skills_taxonomy_application/region_application/{hier_date}/nuts_levela_props_separate_top.pdf",
    bbox_inches="tight",
)


# %%
fig, axs = plt.subplots(3, 4, figsize=(20, 15))
axs[-1, -1].axis('off')

coli = 0
rowi = 0
for i, level_a_name in enumerate(
    plot_level_a_order
):
    color = levela_cols_dict[level_a_name]
    if i != 0 and i % 4 == 0:
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
    f"outputs/skills_taxonomy_application/region_application/{hier_date}/nuts_levela_props_separate_all.pdf",
    bbox_inches="tight",
)


# %% [markdown]
# ## Plot circles on map for
# - How many job avderts
# - How mnay from the 4 most diverging level A code

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

fig, ax_map = plt.subplots(3, 4, figsize=(20, 15))

coli = 0
rowi = 0
for i, col_name in enumerate(
    sentence_data_with_meta["Hierarchy level A name"].unique()
):
    #     color = levela_cols[name2num_dict[col_name]]
    #     if name2num_dict[col_name] == 0:
    #         color = "white"
    color = [
        [1, 1 - c, 0]
        for c in gdf[f"{col_name} - normalised"] / max(gdf[f"{col_name} - normalised"])
    ]
    if i != 0 and i % 4 == 0:
        rowi += 1
        coli = 0

    world[world.name == "United Kingdom"].plot(ax=ax_map[rowi, coli], color="black")
    gdf.plot(
        ax=ax_map[rowi, coli],
        marker="o",
        color=color,
        markersize=200,  # gdf[f"{col_name} - normalised"] * 500,
        alpha=1,
    )

    ax_map[rowi, coli].set_title(f"{col_name}")
    ax_map[rowi, coli].set_axis_off()

    coli += 1

plt.savefig(
    f"outputs/skills_taxonomy_application/region_application/{hier_date}/nuts_levela_props_maps.pdf",
    bbox_inches="tight",
)


# %%
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

fig, ax_map = plt.subplots(1, 2, figsize=(15, 5))

world[world.name == "United Kingdom"].plot(ax=ax_map[0], color="black")
gdf.plot(
    ax=ax_map[0],
    marker="o",
    c=[[1, 1 - c, 0] for c in gdf["job id"] / max(gdf["job id"])],
    markersize=200,  # gdf["job id"] / 10,
    alpha=1,
)
ax_map[0].set_title("Number of job adverts in sample by region")
ax_map[0].set_axis_off()

# sentence_data_with_meta["NUTs region"].value_counts().plot.barh(
#     xlabel="", ylabel="", title="",
#     color=nesta_orange,
#     ax=ax_map[1]
# )
nuts_num_jobs = (
    sentence_data_with_meta.groupby(["NUTs region"])["job id"].nunique().sort_values()
)

nuts_num_jobs.plot.barh(
    xlabel="",
    ylabel="",
    title="",
    color=[[1, 1 - c, 0] for c in nuts_num_jobs / max(nuts_num_jobs)],
    ax=ax_map[1],
)
plt.savefig(
    f"outputs/skills_taxonomy_application/region_application/{hier_date}/nuts_numbers_maps.pdf",
    bbox_inches="tight",
)


# %% [markdown]
# ## Chloropleth versions

# %%
gdf_polygon = gdf.copy()
gdf_polygon["Proportion of job adverts"] = gdf_polygon["job id"] / sum(gdf_polygon["job id"])
gdf_polygon['point'] = gdf_polygon['geometry']
gdf_polygon['geometry'] = gdf_polygon.index.map(nuts2polygons_dict)

# %%
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

fig, ax_map = plt.subplots(4, 3, figsize=(10, 15))
ax_map[-1, -1].axis('off')

coli = 0
rowi = 0
for i, col_name in enumerate(plot_level_a_order):
    
    gdf_polygon["temp_col"] = gdf_polygon[f"{col_name} - normalised"]/ max(gdf_polygon[f"{col_name} - normalised"])
    
    if i != 0 and i % 3 == 0:
        rowi += 1
        coli = 0

    gdf_polygon.plot(
        ax=ax_map[rowi, coli],
        cmap='autumn_r',
        column='temp_col',
    )

    ax_map[rowi, coli].set_title(col_name.replace('and','\n and'))
    ax_map[rowi, coli].set_axis_off()

    coli += 1

plt.savefig(
    f"outputs/skills_taxonomy_application/region_application/{hier_date}/nuts_levela_props_chloropleth.pdf",
    bbox_inches="tight",
)

# %%
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

fig, ax_map = plt.subplots(1, 2, figsize=(15, 5))

gdf_polygon.plot(
    ax=ax_map[0],
    column='Proportion of job adverts',
    cmap='autumn_r'
)
ax_map[0].set_title("Number of job adverts in sample by region")
ax_map[0].set_axis_off()

# sentence_data_with_meta["NUTs region"].value_counts().plot.barh(
#     xlabel="", ylabel="", title="",
#     color=nesta_orange,
#     ax=ax_map[1]
# )
nuts_num_jobs = (
    sentence_data_with_meta.groupby(["NUTs region"])["job id"].nunique().sort_values()
)

nuts_num_jobs.plot.barh(
    xlabel="",
    ylabel="",
    title="",
    color=[[1, 1 - c, 0] for c in nuts_num_jobs / max(nuts_num_jobs)],
    ax=ax_map[1],
)
plt.savefig(
    f"outputs/skills_taxonomy_application/region_application/{hier_date}/nuts_numbers_chloropleth.pdf",
    bbox_inches="tight",
)

# %%
nuts_num_jobs

# %% [markdown]
# ## London vs the rest for level B

# %%
sum(sentence_data_with_meta["NUTs region"].notna())

# %%
sum(sentence_data_with_meta["subregion"].notna())

# %%
levela_cols_dict

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

# Get the level A names for each of these (in same order)
level_a_names_mapped = [sentence_data[sentence_data['Hierarchy level B name']==i]['Hierarchy level A name'].unique()[0] for i in london_quotient.index]
level_a_cols_mapped = [levela_cols_dict[level_a_name] for level_a_name in level_a_names_mapped]

london_quotient.plot.barh(
    figsize=(8, 15),
    ylabel="London quotient",
    xlabel="Level B hierarchy",
    title="London quotient",
    color=level_a_cols_mapped,
)
plt.axvline(1, color="black")

markers = [
    plt.Line2D([0, 0], [0, 0], color=color, marker="o", linestyle="")
    for color in levela_cols_dict.values()
]
plt.legend(
    markers,
    levela_cols_dict.keys(),
    numpoints=1,
    title="Level A skill group",
    loc="lower right",
)


plt.savefig(
    f"outputs/skills_taxonomy_application/region_application/{hier_date}/london_quotient_levb.pdf",
    bbox_inches="tight",
)

# %%
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
    color=[levela_cols_dict[i] for i in london_quotient.keys()],
)
plt.axvline(1, color="black")

plt.savefig(
    f"outputs/skills_taxonomy_application/region_application/{hier_date}/london_quotient_leva.pdf",
    bbox_inches="tight",
)

# %% [markdown]
# ## Other outliers
# - "North East (England)"-  one of the lowest demands for “Digital and technology” skills, but the highest demand for “Childcare and Education” skills. 
# - Northern Ireland had the least demand for “Attitudes, communication and social skills” and “Management, business processes and administration”, but the highest demands for “Food, cleaning and safety” skills. 
# - Wales had the lowest demand for “Digital and technology” skills, 
# - Scotland had the lowest demands for “Childcare and Education” and “Cognitive skills and languages” skills.
# - West Midlands (England) and East Midlands (England) highest demand for "Manufacturing, engineering and physical skills"
#

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
sentence_data[sentence_data["Hierarchy level B name"] == "compassionate-caring-nature"][
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

region_quotient[-20:]

# %%
region = "West Midlands (England)"

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

region_quotient[-20:]

# %%
sentence_data[sentence_data["Hierarchy level B name"] == "driving-vehicle-electrical"][
    "Hierarchy level C name"
].value_counts()

# %%
sentence_data[sentence_data["Hierarchy level B name"] == "equipment-drawings-manufacturing"][
    "Hierarchy level C name"
].value_counts()


# %%
sentence_data[sentence_data["Hierarchy level B name"] == "store-responsible-footfall"][
    "Hierarchy level C name"
].value_counts()

# %%
sentence_data[sentence_data["Hierarchy level C name"].str.contains('warehouse')]["Hierarchy level B name"].unique()

# %%
sentence_data[sentence_data["Hierarchy level B name"] == "stock-orders-management"][
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
print(region_quotient[0:5])
region_quotient[-20:]

# %%
sentence_data[sentence_data["Hierarchy level C name"].str.contains('english')]["Hierarchy level B name"].unique()

# %%
sentence_data[sentence_data["Hierarchy level B name"] == "store-responsible-footfall"][
    "Hierarchy level C name"
].value_counts()

# %%
sentence_data[sentence_data["Hierarchy level B name"] == "english-command-written"][
    "Hierarchy level C name"
].value_counts()

# %%
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

region_quotient[-20:]

# %%
