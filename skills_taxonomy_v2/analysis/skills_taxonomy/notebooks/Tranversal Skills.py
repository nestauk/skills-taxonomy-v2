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
hier_date = '2022.01.21'
skills_date = '2022.01.14'

# %%
hier_structure_file = f"outputs/skills_taxonomy/{hier_date}_hierarchy_structure.json"
hier_structure = load_s3_data(s3, bucket_name, hier_structure_file)

# %%
skill_hierarchy_file = f"outputs/skills_taxonomy/{hier_date}_skills_hierarchy_named.json"
skill_hierarchy = load_s3_data(s3, bucket_name, skill_hierarchy_file)

# %%
sentence_data = load_s3_data(
    s3,
    bucket_name,
    f"outputs/skills_extraction/extracted_skills/{skills_date}_sentences_skills_data_lightweight.json",
)

# %%
skills_data = load_s3_data(
    s3,
    bucket_name,
    f"outputs/skills_extraction/extracted_skills/{skills_date}_skills_data.json",
)

# %%
sentence_data_orig = sentence_data.copy()

# %%
sentence_data = pd.DataFrame(sentence_data, columns=['job id', 'sentence id',  'Cluster number predicted'])
sentence_data = sentence_data[sentence_data["Cluster number predicted"] >=0]

# %% [markdown]
# ### Add the deduplicated sentences

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
# Get the words id for the existing deduplicated sentence data
sentence_data_ehcd = sentence_data.merge(skill_job_ids_with_dupes_df, how='left', on=['job id', 'sentence id'])
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


# %%
# Add new duplicates to sentence data
sentence_data_all = pd.concat([sentence_data, dupe_sentence_data_df])
sentence_data_all.drop_duplicates(inplace=True)
sentence_data_all.reset_index(inplace=True)

# %%
print(len(sentence_data))
print(len(dupe_sentence_data_df))
print(len(sentence_data_all))

# %%
sentence_data = sentence_data_all.copy()

# %% [markdown]
# ### Add hierarchy info

# %%
sentence_data["Hierarchy level A"] = (
    sentence_data["Cluster number predicted"]
    .astype(str)
    .apply(lambda x: skill_hierarchy[x]["Hierarchy level A"])
)
sentence_data["Hierarchy level A name"] = (
    sentence_data["Cluster number predicted"]
    .astype(str)
    .apply(lambda x: skill_hierarchy[x]["Hierarchy level A name"])
)
sentence_data["Hierarchy level B"] = (
    sentence_data["Cluster number predicted"]
    .astype(str)
    .apply(lambda x: skill_hierarchy[x]["Hierarchy level B"])
)
sentence_data["Hierarchy level C"] = (
    sentence_data["Cluster number predicted"]
    .astype(str)
    .apply(lambda x: skill_hierarchy[x]["Hierarchy level C"])
)
sentence_data["Hierarchy ID"] = (
    sentence_data["Cluster number predicted"]
    .astype(str)
    .apply(lambda x: skill_hierarchy[x]["Hierarchy ID"])
)

# %%
sentence_data.head(2)

# %%
len(sentence_data)

# %% [markdown]
# ## Easy way to get names of level skill groups

# %%
lev_a_name_dict = {}
lev_b_name_dict = {}
lev_c_name_dict = {}
for lev_a_id, lev_a in hier_structure.items():
    lev_a_name_dict[lev_a_id] = lev_a["Name"]
    for lev_b_id, lev_b in lev_a["Level B"].items():
        lev_b_name_dict[lev_b_id] = lev_b["Name"]
        for lev_c_id, lev_c in lev_b["Level C"].items():
            lev_c_name_dict[lev_c_id] = lev_c["Name"]


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
# ## Level C analysis
# October 2021:
# - 58029 job adverts had only one skill
# - 279780 skill pairs in job adverts
# - 28623 unique skill pairs in job adverts
# - 9.2% of the edges have a weighting of 1
# - 30.32% of the edges have a weighting of more than 10
#
# January 2022:
#
# - 646897 job adverts had only one skill
# - 7993790 skill pairs in job adverts
# - 30422 unique skill pairs in job adverts
# - 1.6% of the edges have a weighting of 1
# - 92.57% of the edges have a weighting of more than 10
# - 51.3% of the edges have a weighting of more than 200
#
# January 2022 (add duplicate sentences):
#
# - 654087 job adverts had only one skill
# - 8009860 skill pairs in job adverts
# - 30422 unique skill pairs in job adverts
# - 1.6% of the edges have a weighting of 1
# - 92.6% of the edges have a weighting of more than 10
# - 51.38% of the edges have a weighting of more than 200
#

# %%
level_skill_group = "Hierarchy level C"  # Can be Cluster number
net_lev_c = get_cooccurence_network(sentence_data, level_skill_group)

# %%
edges = net_lev_c.edges(data=True)
all_weights = [w["weight"] for s, e, w in edges]
plt.hist(all_weights, bins=100, color="gray");

# %%
print(
    f"{round(len([v for v in all_weights if v==1])*100/len(all_weights),2)}% of the edges have a weighting of 1"
)
max_edge_weight = 10
print(
    f"{round(len([v for v in all_weights if v>max_edge_weight])*100/len(all_weights),2)}% of the edges have a weighting of more than {max_edge_weight}"
)
max_edge_weight = 200
print(
    f"{round(len([v for v in all_weights if v>max_edge_weight])*100/len(all_weights),2)}% of the edges have a weighting of more than {max_edge_weight}"
)

# %% [markdown]
# ### Level B
# October 2021:
# - 59821 job adverts had only one skill
# - 225599 skill pairs in job adverts
# - 2013 unique skill pairs in job adverts
# - 3.28% of the edges have a weighting of 1
# - 83.06% of the edges have a weighting of more than 10
#
# January 2022:
# - 662488 job adverts had only one skill
# - 5991782 skill pairs in job adverts
# - 3849 unique skill pairs in job adverts
# - 2.08% of the edges have a weighting of 1
# - 88.8% of the edges have a weighting of more than 10

# %%
level_skill_group = "Hierarchy level B"  # Can be Cluster number
net_lev_b = get_cooccurence_network(sentence_data, level_skill_group)

# %%
edges = net_lev_b.edges(data=True)
all_weights = [w["weight"] for s, e, w in edges]
plt.hist(all_weights, bins=100);

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
keepedges = ((s, e) for s, e, w in net_lev_c.edges(data=True) if w["weight"] < 100)
net_lev_c_filt = net_lev_c.edge_subgraph(keepedges)

# %%
max_weight = max(
    [net_lev_c_filt.get_edge_data(a, b)["weight"] for a, b in net_lev_c_filt.edges()]
)
min_weight = min(
    [net_lev_c_filt.get_edge_data(a, b)["weight"] for a, b in net_lev_c_filt.edges()]
)
linewidth_max = 0.1
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
        elif level_skill_group == "Cluster number predicted":
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
        elif level_skill_group == "Cluster number predicted":
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
        elif level_skill_group == "Cluster number predicted":
            skill_group_name = skills_data[str(skill_group_num)]["Skills name"]
        untransversal_skills_names[skill_group_num] = skill_group_name

    return untransversal_skills_names


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
ax.axvline(0.063, color="orange", linestyle="--")
ax.axhline(0.978, color="orange", linestyle="--")


# %%
cent_min = 0.063
clust_max = 0.978
level_c_skill_group_scores[
    (level_c_skill_group_scores["centrality"] > cent_min)
    & (level_c_skill_group_scores["clustering_coeff"] < clust_max)
]

# %% [markdown]
# #### Get examples for the connections to a transversal skill

# %%
lev_c_name ="team-part-teams"
lev_c_name_num = {v:k for k,v in lev_c_name_dict.items()}[lev_c_name]
lev_c_name_num

# %%
all_skill_combinations = []
num_one_skills = 0
for _, job_skills in tqdm(sentence_data.groupby("job id")):
    unique_skills = job_skills['Hierarchy level C'].unique().tolist()
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

# %%
# Get random pairs of close neighbours and see how connected these are
print(f"The highly transversal node {lev_c_name} has the quite unconnected neighbours..")

high_connected = edge_list_weighted[(edge_list_weighted['source']==int(lev_c_name_num)) & (edge_list_weighted["weight"]>100)]['target'].tolist()
for source, target in [random.sample(high_connected, 2) for i in range(100)]:
    neighbours_weights = edge_list_weighted[(edge_list_weighted['source']==source) & (edge_list_weighted["target"]==target)]["weight"]
    if len(neighbours_weights)!=0:
        neighbours_weights = neighbours_weights.iloc[0]
        if neighbours_weights < 100:
            print(f"{lev_c_name_dict[str(source)]} to {lev_c_name_dict[str(target)]} has weight {neighbours_weights}")


# %%
random.sample(high_connected,2)

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
ax.axvline(0.1, color="orange", linestyle="--")
ax.axhline(0.963, color="orange", linestyle="--")


# %%
cent_min = 0.10
clust_max = 0.963
level_b_skill_group_scores[
    (level_b_skill_group_scores["centrality"] > cent_min)
    & (level_b_skill_group_scores["clustering_coeff"] < clust_max)
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

axes[0].axvline(0.1, color="orange", linestyle="--")
axes[0].axhline(0.963, color="orange", linestyle="--")
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

axes[1].axvline(0.063, color="orange", linestyle="--")
axes[1].axhline(0.978, color="orange", linestyle="--")
plt.tight_layout()

plt.savefig(
    f"outputs/skills_taxonomy/transversal/{hier_date}/transversal_skills_scatter.pdf",
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
axes.axvline(0.063, color="orange", linestyle="--")
axes.axhline(0.978, color="orange", linestyle="--")

plt.tight_layout()

plt.savefig(
    f"outputs/skills_taxonomy/transversal/{hier_date}/transversal_skills_scatter_levc.pdf",
    bbox_inches="tight",
)


# %%
cent_min = 0.063
clust_max = 0.978
trans_skills_levc = level_c_skill_group_scores[
    (level_c_skill_group_scores["centrality"] > cent_min)
    & (level_c_skill_group_scores["clustering_coeff"] < clust_max)
]

# %%
trans_skills = trans_skills_levc.index.tolist()
trans_skills_hier = {}
for lev_a_id, lev_a in hier_structure.items():
    for lev_b_id, lev_b in lev_a["Level B"].items():
        for lev_c_id, lev_c in lev_b["Level C"].items():
            if int(lev_c_id) in trans_skills:
                transkills_info = []
                for skill_name, skill_info in lev_c['Skills'].items():
                    transkills_info.append(
                        (skill_hierarchy[skill_name]['Skill name'], skill_info['Number of sentences that created skill']))
                transkills_info.sort(key=lambda x:x[1], reverse=True)
                trans_skills_hier[int(lev_c_id)] = {
                    "Level A": lev_a_id,
                    "Level B": lev_b_id,
                    "Level C": lev_c_id,
                    "Level A name": lev_a['Name'],
                    "Level B name": lev_b_name_dict[lev_b_id],
                    "Skills": [s[0] for s in transkills_info[0:10]]
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
trans_skills_levc.to_csv(f"outputs/skills_taxonomy/transversal/{hier_date}/lev_c_trans_skills.csv")
trans_skills_levc

# %%
