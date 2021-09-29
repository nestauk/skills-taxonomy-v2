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
# ## Identifying transversal skills
# Get skills co-occurences and find the most and least transversal ones.
# Might make sense to do this at a level C or D level rather than individual skills (as there is perhaps a lot of overlap).

# %%
# cd ../../../..

# %%
from skills_taxonomy_v2.getters.s3_data import load_s3_data

# %%
from collections import Counter, defaultdict
import json
from itertools import chain, combinations
from tqdm import tqdm
import random

from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import plotly.graph_objects as go
import plotly
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import boto3
from ipywidgets import interact
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
# ## Load data

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
# ### Join the hierarchy data to the sentences

# %%
sentence_data = pd.DataFrame(sentence_data)
sentence_data = sentence_data[sentence_data["Cluster number"] != -1]

# %%
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

# %%
sentence_data.head(2)

# %% [markdown]
# ## Easy way to get names of level skill groups

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


# %% [markdown]
# ## Get skill co-occurence

# %%
def get_cooccurence_network(sentence_data, level_skill_group):
    all_skill_combinations = []
    num_one_skills = 0
    for _, job_skills in tqdm(sentence_data.groupby("job id")):
        unique_skills = job_skills[level_skill_group].unique().tolist()
        if len(unique_skills) != 1:
            all_skill_combinations += list(combinations(sorted(unique_skills), 2))
        else:
            num_one_skills += 1

    print(f"{num_one_skills} job adverts had only one skill")
    print(f"{len(all_skill_combinations)} skill pairs in job adverts")

    edge_list = pd.DataFrame(all_skill_combinations, columns=["source", "target"])
    edge_list["weight"] = 1
    edge_list_weighted = (
        edge_list.groupby(["source", "target"])["weight"].sum().reset_index(drop=False)
    )

    print(f"{len(edge_list_weighted)} unique skill pairs in job adverts")

    # Build network
    net = nx.from_pandas_edgelist(edge_list_weighted, edge_attr=True)

    return net


# %% [markdown]
# ## Level D analysis

# %%
level_skill_group = "Hierarchy level D"  # Can be Cluster number
net_lev_d = get_cooccurence_network(sentence_data, level_skill_group)

# %%
edges = net_lev_d.edges(data=True)
all_weights = [w["weight"] for s, e, w in edges]
plt.hist(all_weights, bins=100)

# %%
print(
    f"{round(len([v for v in all_weights if v==1])*100/len(all_weights),2)}% of the edges have a weighting of 1"
)
max_edge_weight = 10
print(
    f"{round(len([v for v in all_weights if v>max_edge_weight])*100/len(all_weights),2)}% of the edges have a weighting of more than {max_edge_weight}"
)

# %% [markdown]
# ## Level C analysis

# %%
level_skill_group = "Hierarchy level C"  # Can be Cluster number
net_lev_c = get_cooccurence_network(sentence_data, level_skill_group)

# %%
edges = net_lev_c.edges(data=True)
all_weights = [w["weight"] for s, e, w in edges]
plt.hist(all_weights, bins=100, color="gray")

# %%
print(
    f"{round(len([v for v in all_weights if v==1])*100/len(all_weights),2)}% of the edges have a weighting of 1"
)
max_edge_weight = 10
print(
    f"{round(len([v for v in all_weights if v>max_edge_weight])*100/len(all_weights),2)}% of the edges have a weighting of more than {max_edge_weight}"
)

# %% [markdown]
# ### Level B

# %%
level_skill_group = "Hierarchy level B"  # Can be Cluster number
net_lev_b = get_cooccurence_network(sentence_data, level_skill_group)

# %%
edges = net_lev_b.edges(data=True)
all_weights = [w["weight"] for s, e, w in edges]
plt.hist(all_weights, bins=100)

# %%
print(
    f"{round(len([v for v in all_weights if v==1])*100/len(all_weights),2)}% of the edges have a weighting of 1"
)
max_edge_weight = 10
print(
    f"{round(len([v for v in all_weights if v>max_edge_weight])*100/len(all_weights),2)}% of the edges have a weighting of more than {max_edge_weight}"
)

# %% [markdown]
# ## Plot networks

# %%
# All the edges

max_weight = max(
    [net_lev_c.get_edge_data(a, b)["weight"] for a, b in net_lev_c.edges()]
)
min_weight = min(
    [net_lev_c.get_edge_data(a, b)["weight"] for a, b in net_lev_c.edges()]
)
linewidth_max = 0.5
linewidth_min = 0.01

grad = (linewidth_max - linewidth_min) / (max_weight - min_weight)


# Show with Bokeh
plot = Plot(plot_width=500, plot_height=500)
plot.title.text = "Level C skill groups co-occurence network"

node_hover_tool = HoverTool(tooltips=[("Skill", "@skill")])
plot.add_tools(node_hover_tool, BoxZoomTool(), ResetTool(), WheelZoomTool(), SaveTool())

graph_renderer = from_networkx(net_lev_c, nx.spring_layout, scale=1, center=(0, 0))

graph_renderer.node_renderer.glyph = Circle(
    size=3, fill_color="orange", line_color=None
)  # Spectral4[0])

graph_renderer.edge_renderer.data_source.data["line_width"] = [
    net_lev_c.get_edge_data(a, b)["weight"] * grad + linewidth_min
    for a, b in net_lev_c.edges()
]
graph_renderer.edge_renderer.glyph.line_width = {"field": "line_width"}

plot.renderers.append(graph_renderer)

show(plot, notebook_handle=True)

# %% [markdown]
# ### Only plot edges where weight>1

# %%
keepedges = ((s, e) for s, e, w in net_lev_c.edges(data=True) if w["weight"] > 40)
net_lev_c_filt = net_lev_c.edge_subgraph(keepedges)

# %%
max_weight = max(
    [net_lev_c_filt.get_edge_data(a, b)["weight"] for a, b in net_lev_c_filt.edges()]
)
min_weight = min(
    [net_lev_c_filt.get_edge_data(a, b)["weight"] for a, b in net_lev_c_filt.edges()]
)
linewidth_max = 0.5
linewidth_min = 0.01

grad = (linewidth_max - linewidth_min) / (max_weight - min_weight)


# Show with Bokeh
plot = Plot(plot_width=500, plot_height=500)
plot.title.text = (
    "Level C skill groups co-occurence network - only high weights kept in"
)

node_hover_tool = HoverTool(tooltips=[("Skill", "@skill")])
plot.add_tools(node_hover_tool, BoxZoomTool(), ResetTool(), WheelZoomTool(), SaveTool())

graph_renderer = from_networkx(net_lev_c_filt, nx.spring_layout, scale=1, center=(0, 0))

graph_renderer.node_renderer.glyph = Circle(
    size=3, fill_color="orange", line_color=None
)  # Spectral4[0])

graph_renderer.edge_renderer.data_source.data["line_width"] = [
    net_lev_c_filt.get_edge_data(a, b)["weight"] * grad + linewidth_min
    for a, b in net_lev_c_filt.edges()
]
graph_renderer.edge_renderer.glyph.line_width = {"field": "line_width"}

plot.renderers.append(graph_renderer)

show(plot, notebook_handle=True)

# %% [markdown]
# ### Level B network

# %%
max_weight = max(
    [net_lev_b.get_edge_data(a, b)["weight"] for a, b in net_lev_b.edges()]
)
min_weight = min(
    [net_lev_b.get_edge_data(a, b)["weight"] for a, b in net_lev_b.edges()]
)
linewidth_max = 0.5
linewidth_min = 0.01

grad = (linewidth_max - linewidth_min) / (max_weight - min_weight)


# Show with Bokeh
plot = Plot(plot_width=400, plot_height=400)
plot.title.text = "Level B skill groups co-occurence network"

node_hover_tool = HoverTool(tooltips=[("Skill", "@skill")])
plot.add_tools(node_hover_tool, BoxZoomTool(), ResetTool(), WheelZoomTool(), SaveTool())

graph_renderer = from_networkx(net_lev_b, nx.spring_layout, scale=1, center=(0, 0))

graph_renderer.node_renderer.glyph = Circle(
    size=3, fill_color="orange", line_color=None
)  # Spectral4[0])

graph_renderer.edge_renderer.data_source.data["line_width"] = [
    net_lev_b.get_edge_data(a, b)["weight"] * grad + linewidth_min
    for a, b in net_lev_b.edges()
]
graph_renderer.edge_renderer.glyph.line_width = {"field": "line_width"}


plot.renderers.append(graph_renderer)

show(plot, notebook_handle=True)

# %%
# Show with Bokeh
plot = Plot(plot_width=300, plot_height=300)
plot.title.text = "Level A skill groups co-occurence network"

node_hover_tool = HoverTool(tooltips=[("Skill", "@skill")])
plot.add_tools(node_hover_tool, BoxZoomTool(), ResetTool(), WheelZoomTool(), SaveTool())

graph_renderer = from_networkx(net_lev_a, nx.spring_layout, scale=1, center=(0, 0))

graph_renderer.node_renderer.glyph = Circle(
    size=3, fill_color="red", line_color=None
)  # Spectral4[0])
graph_renderer.edge_renderer.glyph = MultiLine(
    line_color="black", line_alpha=0.1, line_width=0.2
)
plot.renderers.append(graph_renderer)

show(plot, notebook_handle=True)


# %% [markdown]
# ## Extract tranversal skills
#
# From original report:
# - Eigenvector centrality is a measure of vertex influence, so if a vertex is connected to vertices with a high number of degrees, its centrality will be high (Austin, 2006).
# - The local clustering coefficient reflects how embedded the vertex is in its neighbourhood. If one vertex has a lower local clustering coefficient than another, it implies that fewer of that vertex’s neighbours are connected to each other (Watts and Strogatz, 1998).
# - We argue that a highly transversal skill is likely to have a high eigenvector centrality and a low local clustering coefficient, since the vertices they connect have relatively few other connections in common

# %%
def get_transversal_skills(net, level_skill_group):
    centrality = nx.eigenvector_centrality(net)
    clustering_coeff = nx.clustering(net)

    skill_group_scores = pd.concat(
        [
            pd.DataFrame.from_dict(centrality, orient="index", columns=["centrality"]),
            pd.DataFrame.from_dict(
                clustering_coeff, orient="index", columns=["clustering_coeff"]
            ),
        ],
        axis=1,
    )

    skill_group_names = []
    for skill_group_num in skill_group_scores.index:
        if level_skill_group == "Hierarchy level A":
            skill_group_names.append(lev_a_name_dict[str(skill_group_num)])
        elif level_skill_group == "Hierarchy level B":
            skill_group_names.append(lev_b_name_dict[str(skill_group_num)])
        elif level_skill_group == "Hierarchy level C":
            skill_group_names.append(lev_c_name_dict[str(skill_group_num)])
        elif level_skill_group == "Hierarchy level D":
            skill_group_names.append(lev_d_name_dict[str(skill_group_num)])
        elif level_skill_group == "Cluster number":
            skill_group_names.append(skills_data[str(skill_group_num)]["Skills name"])

    skill_group_scores[f"{level_skill_group} name"] = skill_group_names
    return skill_group_scores


# %%
def print_trans_skills(skill_group_scores, level_skill_group, cent_min, clust_max):

    transversal_skills = skill_group_scores[
        (skill_group_scores["centrality"] > cent_min)
        & (skill_group_scores["clustering_coeff"] < clust_max)
    ].index.tolist()

    transversal_skills_names = {}
    for skill_group_num in transversal_skills:
        if level_skill_group == "Hierarchy level A":
            skill_group_name = lev_a_name_dict[str(skill_group_num)]
        elif level_skill_group == "Hierarchy level B":
            skill_group_name = lev_b_name_dict[str(skill_group_num)]
        elif level_skill_group == "Hierarchy level C":
            skill_group_name = lev_c_name_dict[str(skill_group_num)]
        elif level_skill_group == "Hierarchy level D":
            skill_group_name = lev_d_name_dict[str(skill_group_num)]
        elif level_skill_group == "Cluster number":
            skill_group_name = skills_data[str(skill_group_num)]["Skills name"]
        transversal_skills_names[skill_group_num] = skill_group_name

    return transversal_skills_names


# %%
def print_untrans_skills(skill_group_scores, level_skill_group, cent_max, clust_min):

    untransversal_skills = skill_group_scores[
        (skill_group_scores["centrality"] < cent_max)
        & (skill_group_scores["clustering_coeff"] > clust_min)
    ].index.tolist()

    untransversal_skills_names = {}
    for skill_group_num in untransversal_skills:
        if level_skill_group == "Hierarchy level A":
            skill_group_name = lev_a_name_dict[str(skill_group_num)]
        elif level_skill_group == "Hierarchy level B":
            skill_group_name = lev_b_name_dict[str(skill_group_num)]
        elif level_skill_group == "Hierarchy level C":
            skill_group_name = lev_c_name_dict[str(skill_group_num)]
        elif level_skill_group == "Hierarchy level D":
            skill_group_name = lev_d_name_dict[str(skill_group_num)]
        elif level_skill_group == "Cluster number":
            skill_group_name = skills_data[str(skill_group_num)]["Skills name"]
        untransversal_skills_names[skill_group_num] = skill_group_name

    return untransversal_skills_names


# %% [markdown]
# ## Level D

# %%
level_d_skill_group_scores = get_transversal_skills(net_lev_d, "Hierarchy level D")

# %%
# highly transversal skill = high centrality and a low local clustering coefficient
ax = level_d_skill_group_scores.plot.scatter(
    "centrality",
    "clustering_coeff",
    figsize=(5, 5),
    color="black",
    alpha=0.6,
    title=f"Centrality score and local clustering coefficient\nfor Hierarchy level D skill groups",
    xlabel="Eigenvector centrality",
    ylabel="Local clustering coefficient",
)
ax.axvline(0.14, color="orange", linestyle="--")
ax.axhline(0.7, color="orange", linestyle="--")


# %%
cent_min = 0.14
clust_max = 0.7
level_d_skill_group_scores[
    (level_d_skill_group_scores["centrality"] > cent_min)
    & (level_d_skill_group_scores["clustering_coeff"] < clust_max)
]

# %%
cent_max = 0.06
clust_min = 0.8
level_d_skill_group_scores[
    (level_d_skill_group_scores["centrality"] < cent_max)
    & (level_d_skill_group_scores["clustering_coeff"] > clust_min)
]

# %% [markdown]
# ## Level C

# %%
level_c_skill_group_scores = get_transversal_skills(net_lev_c, "Hierarchy level C")

# %%
# highly transversal skill = high centrality and a low local clustering coefficient
ax = level_c_skill_group_scores.plot.scatter(
    "centrality",
    "clustering_coeff",
    figsize=(5, 5),
    color="black",
    alpha=0.6,
    title=f"Centrality score and local clustering coefficient\nfor Hierarchy level C skill groups",
    xlabel="Eigenvector centrality",
    ylabel="Local clustering coefficient",
)
ax.axvline(0.065, color="orange", linestyle="--")
ax.axhline(0.93, color="orange", linestyle="--")


# %%
cent_min = 0.065
clust_max = 0.93
level_c_skill_group_scores[
    (level_c_skill_group_scores["centrality"] > cent_min)
    & (level_c_skill_group_scores["clustering_coeff"] < clust_max)
]

# %%
cent_max = 0.05
clust_min = 0.95
level_c_skill_group_scores[
    (level_c_skill_group_scores["centrality"] < cent_max)
    & (level_c_skill_group_scores["clustering_coeff"] > clust_min)
]

# %% [markdown]
# ## Level B

# %%
level_b_skill_group_scores = get_transversal_skills(net_lev_b, "Hierarchy level B")

# %%
# highly transversal skill = high centrality and a low local clustering coefficient
ax = level_b_skill_group_scores.plot.scatter(
    "centrality",
    "clustering_coeff",
    figsize=(5, 5),
    color="black",
    alpha=0.6,
    title=f"Centrality score and local clustering coefficient\nfor Hierarchy level B skill groups",
    xlabel="Eigenvector centrality",
    ylabel="Local clustering coefficient",
)
ax.axvline(0.12, color="orange", linestyle="--")
ax.axhline(0.96, color="orange", linestyle="--")


# %%
cent_min = 0.12
clust_max = 0.96
level_b_skill_group_scores[
    (level_b_skill_group_scores["centrality"] > cent_min)
    & (level_b_skill_group_scores["clustering_coeff"] < clust_max)
]

# %%
cent_max = 0.1
clust_min = 0.98
level_b_skill_group_scores[
    (level_b_skill_group_scores["centrality"] < cent_max)
    & (level_b_skill_group_scores["clustering_coeff"] > clust_min)
]

# %% [markdown]
# ## Plots together

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

level_b_skill_group_scores.plot.scatter(
    "centrality",
    "clustering_coeff",
    color="black",
    alpha=0.6,
    title=f"Centrality score and local clustering coefficient\nfor Hierarchy level B skill groups",
    xlabel="Eigenvector centrality",
    ylabel="Local clustering coefficient",
    ax=axes[0],
)
axes[0].axvline(0.12, color="orange", linestyle="--")
axes[0].axhline(0.96, color="orange", linestyle="--")


level_c_skill_group_scores.plot.scatter(
    "centrality",
    "clustering_coeff",
    color="black",
    alpha=0.6,
    title=f"Centrality score and local clustering coefficient\nfor Hierarchy level C skill groups",
    xlabel="Eigenvector centrality",
    ylabel="Local clustering coefficient",
    ax=axes[1],
)
axes[1].axvline(0.065, color="orange", linestyle="--")
axes[1].axhline(0.93, color="orange", linestyle="--")


plt.tight_layout()

plt.savefig(
    "outputs/skills_taxonomy/transversal/transversal_skills_scatter.pdf",
    bbox_inches="tight",
)


# %%
fig, axes = plt.subplots(1, 1, figsize=(5, 5))

level_c_skill_group_scores.plot.scatter(
    "centrality",
    "clustering_coeff",
    color="black",
    alpha=0.6,
    title=f"Centrality score and local clustering coefficient\nfor Hierarchy level C skill groups",
    xlabel="Eigenvector centrality",
    ylabel="Local clustering coefficient",
    ax=axes,
)
axes.axvline(0.065, color="orange", linestyle="--")
axes.axhline(0.93, color="orange", linestyle="--")

plt.tight_layout()

plt.savefig(
    "outputs/skills_taxonomy/transversal/transversal_skills_scatter_levc.pdf",
    bbox_inches="tight",
)


# %%
cent_min = 0.065
clust_max = 0.93
trans_skills_levc = level_c_skill_group_scores[
    (level_c_skill_group_scores["centrality"] > cent_min)
    & (level_c_skill_group_scores["clustering_coeff"] < clust_max)
]

# %%
cent_max = 0.04
clust_min = 0.96
level_c_skill_group_scores[
    (level_c_skill_group_scores["centrality"] < cent_max)
    & (level_c_skill_group_scores["clustering_coeff"] > clust_min)
]

# %%
trans_skills = trans_skills_levc.index.tolist()
trans_skills_hier = {}
for lev_a_id, lev_a in hier_structure.items():
    for lev_b_id, lev_b in lev_a["Level B"].items():
        for lev_c_id, lev_c in lev_b["Level C"].items():
            if int(lev_c_id) in trans_skills:
                trans_skills_hier[int(lev_c_id)] = {
                    "Level A": lev_a_id,
                    "Level B": lev_b_id,
                    "Level C": lev_c_id,
                    "Level A name": level_a_rename_dict[lev_a_id],
                    "Level B name": lev_b_name_dict[lev_b_id],
                }

# %%
num_all_job_ads = sentence_data["job id"].nunique()
trans_skills_levc = pd.concat(
    [trans_skills_levc, pd.DataFrame(trans_skills_hier).T], axis=1
)
trans_skills_levc["Percentage of job adverts with this skill"] = trans_skills_levc[
    "Level C"
].apply(
    lambda x: round(
        sentence_data[sentence_data["Hierarchy level C"] == int(x)]["job id"].nunique()
        * 100
        / num_all_job_ads,
        2,
    )
)
trans_skills_levc.to_csv("outputs/skills_taxonomy/transversal/lev_c_trans_skills.csv")
trans_skills_levc

# %%
