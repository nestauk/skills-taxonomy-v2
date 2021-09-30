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
# ## Plotting hierarchy
# There are two plots produced from this script:
# 1. Basic plot - plot skills and colour by one hierarchy level at a time
# 2. Interactive plot - plot skills and use interactive filter to colour by skills groups

# %%
# cd ../../../..

# %%
from skills_taxonomy_v2.getters.s3_data import load_s3_data
from skills_taxonomy_v2 import PROJECT_DIR

# %%
from collections import Counter, defaultdict
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import boto3
from ipywidgets import interact
import streamlit as st
import bokeh.plotting as bpl
from bokeh.plotting import (
    ColumnDataSource,
    figure,
    output_file,
    show,
    from_networkx,
    gridplot,
)
from bokeh.models import (
    ResetTool,
    BoxZoomTool,
    WheelZoomTool,
    HoverTool,
    SaveTool,
    Label,
    CategoricalColorMapper,
    ColorBar,
    ColumnDataSource,
    LinearColorMapper,
    Circle,
    MultiLine,
    Plot,
    Range1d,
    Title,
)

from bokeh.io import output_file, reset_output, save, export_png, show, push_notebook
from bokeh.resources import CDN
from bokeh.embed import file_html
from bokeh.palettes import (
    Plasma,
    magma,
    cividis,
    inferno,
    plasma,
    viridis,
    Spectral6,
    Turbo256,
    Spectral,
    Spectral4,
    inferno,
)
from bokeh.transform import linear_cmap

bpl.output_notebook()

# %%
bucket_name = "skills-taxonomy-v2"
s3 = boto3.resource("s3")

# %% [markdown]
# ## Load hierarchy data

# %%
sentence_data = load_s3_data(
    s3,
    bucket_name,
    "outputs/skills_extraction/extracted_skills/2021.08.31_sentences_data.json",
)


# %%
skill_hierarchy_file = "outputs/skills_hierarchy/2021.09.06_skills_hierarchy.json"
skill_hierarchy = load_s3_data(s3, bucket_name, skill_hierarchy_file)

# %% [markdown]
# ### Manual names for level A

# %%
with open(
    str(PROJECT_DIR) + "/skills_taxonomy_v2/utils/2021.09.06_level_a_rename_dict.json",
    "r",
) as f:
    level_a_rename_dict = json.load(f)

# %% [markdown]
# ## Filter sentence data

# %%
sentence_data = pd.DataFrame(sentence_data)
sentence_data = sentence_data[sentence_data["Cluster number"] != -1]

# %%
sentence_data.head(2)

# %% [markdown]
# ## Create skills hierarchy dataframe with average sentence coordinates

# %%
skill_hierarchy_df = pd.DataFrame(skill_hierarchy).T
skill_hierarchy_df["Skill number"] = skill_hierarchy_df.index
skill_hierarchy_df["Hierarchy level A name"] = skill_hierarchy_df[
    "Hierarchy level A"
].apply(lambda x: level_a_rename_dict[str(x)])

# %%
# Average reduced embedding per skill
average_emb_clust = (
    sentence_data.groupby("Cluster number")[["reduced_points x", "reduced_points y"]]
    .mean()
    .to_dict(orient="index")
)
average_emb_clust = {
    str(k): [v["reduced_points x"], v["reduced_points y"]]
    for k, v in average_emb_clust.items()
}


skill_hierarchy_df["Average reduced_points x"] = skill_hierarchy_df[
    "Skill number"
].apply(lambda x: average_emb_clust.get(str(x))[0])
skill_hierarchy_df["Average reduced_points y"] = skill_hierarchy_df[
    "Skill number"
].apply(lambda x: average_emb_clust.get(str(x))[1])


# %% [markdown]
# ## Basic plot - plot skills and colour by one hierarchy level at a time

# %%
def plot_skills_by_level(skill_hierarchy_df, col_by="A"):
    hier_levela = skill_hierarchy_df["Hierarchy level A"].astype(str).tolist()
    hier_levelb = skill_hierarchy_df["Hierarchy level B"].astype(str).tolist()
    hier_levelc = skill_hierarchy_df["Hierarchy level C"].astype(str).tolist()

    hier_levela_names = skill_hierarchy_df["Hierarchy level A name"].tolist()
    hier_levelb_names = skill_hierarchy_df["Hierarchy level B name"].tolist()
    hier_levelc_names = skill_hierarchy_df["Hierarchy level C name"].tolist()

    skill_names = skill_hierarchy_df["Skill name"].tolist()
    reduced_x = skill_hierarchy_df["Average reduced_points x"].tolist()
    reduced_y = skill_hierarchy_df["Average reduced_points y"].tolist()
    color_palette = viridis

    if col_by == "A":
        colors_by_labels = hier_levela
    elif col_by == "B":
        colors_by_labels = hier_levelb
    elif col_by == "C":
        colors_by_labels = hier_levelc

    ds_dict = dict(
        x=reduced_x,
        y=reduced_y,
        skill_names=skill_names,
        label=colors_by_labels,
        hier_levela=hier_levela_names,
        hier_levelb=hier_levelb_names,
        hier_levelc=hier_levelc_names,
    )
    hover = HoverTool(
        tooltips=[
            ("Skill name", "@skill_names"),
            ("Hierarchy level A", "@hier_levela"),
            ("Hierarchy level B", "@hier_levelb"),
            ("Hierarchy level C", "@hier_levelc"),
        ]
    )
    source = ColumnDataSource(ds_dict)
    unique_colors = list(set(colors_by_labels))
    num_unique_colors = len(unique_colors)

    color_mapper = LinearColorMapper(
        palette="Turbo256", low=0, high=len(unique_colors) + 1
    )

    p = figure(
        plot_width=500,
        plot_height=500,
        tools=[hover, WheelZoomTool(), BoxZoomTool(), SaveTool()],
        title=f"Skills coloured by hierarchy level {col_by}",
        toolbar_location="below",
    )
    p.circle(
        x="x",
        y="y",
        radius=0.1,
        alpha=0.5,
        source=source,
        color={"field": "label", "transform": color_mapper},
    )
    return p


# %%
# %% [markdown]
# ## Interactive plot - plot skills and use interactive filter to colour by skills groups

# %%
def filter_skills(skill_hierarchy_df, level_A, level_B, level_C):
    skill_hierarchy_filt = skill_hierarchy_df.copy()
    if level_A != "All":
        skill_hierarchy_filt = skill_hierarchy_filt[
            skill_hierarchy_filt["Hierarchy level A"] == level_A
        ]
        if level_B != "All":
            skill_hierarchy_filt = skill_hierarchy_filt[
                skill_hierarchy_filt["Hierarchy level B"] == level_B
            ]
            if level_C != "All":
                skill_hierarchy_filt = skill_hierarchy_filt[
                    skill_hierarchy_filt["Hierarchy level C"] == level_C
                ]

    return skill_hierarchy_filt


# %%
def update_colours(skill_hierarchy_df, left_gp, col_by, radius_size, alpha):
    color_mapper = LinearColorMapper(
        palette="Turbo256",
        low=0,
        high=skill_hierarchy_df[f"Hierarchy level {col_by}"].nunique() + 1,
    )
    left = left_gp.circle(
        "Average reduced_points x",
        "Average reduced_points y",
        source=source,
        radius=radius_size,
        alpha=alpha,
        color={"field": f"Hierarchy level {col_by}", "transform": color_mapper},
    )
    return left


# %%
def update(col_by, level_A, level_B, level_C, radius_size):

    if level_A == "All":
        alpha = 0.5
        radius_size = 0.1
    elif level_B == "All":
        alpha = 0.8
        radius_size = 0.08
    else:
        alpha = 0.9
        radius_size = 0.05

    left = update_colours(
        skill_hierarchy_df, left_gp, col_by=col_by, radius_size=radius_size, alpha=alpha
    )

    left.data_source.data = filter_skills(skill_hierarchy_df, level_A, level_B, level_C)
    push_notebook()


# %%
source = ColumnDataSource(data=skill_hierarchy_df)

hover = HoverTool(
    tooltips=[
        ("Skill name", "@{Skill name}(@{Skill number})"),
        (
            "Number of sentences that created skill",
            "@{Number of sentences that created skill}",
        ),
        ("Hierarchy level A", "@{Hierarchy level A name} (@{Hierarchy level A})"),
        ("Hierarchy level B", "@{Hierarchy level B name} (@{Hierarchy level B})"),
        ("Hierarchy level C", "@{Hierarchy level C name} (@{Hierarchy level C})"),
    ]
)

left_gp = figure(
    plot_width=500,
    plot_height=500,
    tools=[hover, WheelZoomTool(), BoxZoomTool(), SaveTool()],
    title=f"Skills coloured by hierarchy level",
    toolbar_location="below",
)
left = update_colours(
    skill_hierarchy_df, left_gp, col_by="A", radius_size=0.1, alpha=0.5
)

p = gridplot([[left_gp]])
show(p, notebook_handle=True)
interact(
    update,
    col_by=["A", "B", "C"],
    level_A=["All"] + sorted(skill_hierarchy_df["Hierarchy level A"].unique().tolist()),
    level_B=["All"] + sorted(skill_hierarchy_df["Hierarchy level B"].unique().tolist()),
    level_C=["All"] + sorted(skill_hierarchy_df["Hierarchy level C"].unique().tolist()),
    radius_size=(0.001, 0.1, 0.001),
)

# %%
# streamlit code
title = '<span style="color:blue; font-family:Courier New; text-align:center; font-size:40px;">Explore the Skills Taxonomy.</span>'
st.markdown(title, unsafe_allow_html=True)
instruction = '<span style="color:black; font-family:Courier New; text-align:center; font-size:20px;">Select a skill level in the dropdown box:</span>'
st.markdown(instruction, unsafe_allow_html=True)

levels = st.selectbox("", ("A", "B", "C"))
p = plot_skills_by_level(skill_hierarchy_df, col_by=levels)
st.bokeh_chart(p)

links = '<span style="color:black; font-family:Courier New; text-align:center; font-size:15px;">Read the full technical report [here](https://docs.google.com/document/d/1ZHE6Z6evxyUsSiojdNatSa_yMDJ8_UlB1K4YD1AhGG8/edit) and the extended article [here](https://docs.google.com/document/d/14lY7niHD0lyYpBj8TtlFGMSA2q4p4u5hl6LnXE4HYLs/edit).</span>'
st.markdown(links, unsafe_allow_html=True)
