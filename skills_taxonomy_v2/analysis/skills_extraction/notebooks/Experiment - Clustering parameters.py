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
# ## Find the best clustering parameters for the reduced embeddings.
#
# First, we decide to remove sentences if the length is over 100 characters. This is reduce noisy centre of embedding space where the sentences dont cluster well and also mess up the rest of the clustering.
#
# There are two parameters to tweak:
# - eps float, default=0.5 The maximum distance between two samples for one to be considered as in the neighborhood of the other. This is not a maximum bound on the distances of points within a cluster. This is the most important DBSCAN parameter to choose appropriately for your data set and distance function.
#
# - min_samples int, default=5 The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.
#
# We did 1231 parameter combinations to predict clusters in a random sample of 300k sentences.
#
# We decided on the following metrics to find good parameters:
# - Shouldn't be too many clusters (last time 20k was a lot) - between 7000 and 15000
# - Shouldn't be any massive clusters - 0 clusters of size >10,000 sentences
# - Shouldn't be too many small clusters - average size of clusters is >10
# - Number of sentences not clustered (the "-1 cluster") < 200,000
#
# |dbscan_eps	| dbscan_min_samples | Number of clusters	| Number of really large clusters (>10000 sentences)|	Number of really small clusters (<10 sentences)	| Number of sentences not clustered	|Average size of clusters|
# |---|---|---|---|---|---|---|
# |	0.008|	4.0|	12205|	0|	9987|	164817|	11|
# |	0.008|	5.0|    7422|	0|	5512|	198439|	14|
# |	0.009|	4.0|	12190|	0|	9664|	139739|	13|
# |	0.009|	5.0|	7892|	0|	5650|	173253|	16|
# |	**0.010**|	**4.0**|	11551|	0|	8892|	117923|	16|
# |	0.010|	5.0|	8058|	0|	5628|	149301|	19|
#
# We went for `dbscan_eps = 0.01` and `dbscan_min_samples = 4` which produces 11551 clusters in 300000 sentences.
#

# %%
import yaml
import random
from tqdm import tqdm
import json
from collections import Counter

import pandas as pd
import numpy as np
import boto3
from sklearn import metrics
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

from skills_taxonomy_v2.getters.s3_data import get_s3_data_paths, save_to_s3, load_s3_data
from skills_taxonomy_v2.pipeline.skills_extraction.extract_skills_utils import (
    load_sentences_embeddings,ExtractSkills
    )
from skills_taxonomy_v2 import BUCKET_NAME

# %%
import bokeh.plotting as bpl
from bokeh.plotting import ColumnDataSource, figure, output_file, show
from bokeh.models import (
    BoxZoomTool,
    WheelZoomTool,
    HoverTool,
    ResetTool,
    SaveTool,
    Label,
    CategoricalColorMapper,
    ColorBar,
    ColumnDataSource,
)
from bokeh.io import output_file, reset_output, save, export_png, show
from bokeh.resources import CDN
from bokeh.embed import file_html
from bokeh.palettes import Plasma, magma, cividis, inferno, plasma, viridis, Spectral6
from bokeh.transform import linear_cmap

bpl.output_notebook()

# %%
s3 = boto3.resource("s3")

# %% [markdown]
# ## Load the reduced sentences
# - can do the vocab increase with number of sentences
# - does the reduction look good?

# %%
sentences_data = pd.DataFrame()
for i in tqdm(range(0, 8)):
    sentences_data_i = load_s3_data(
        s3, BUCKET_NAME,
        f"outputs/skills_extraction/reduced_embeddings/sentences_data_{i}.json"
    )
    sentences_data = pd.concat([sentences_data, pd.DataFrame(sentences_data_i)])

# %%
print(len(sentences_data))
sentences_data.head(2)

# %%
sentences_data["job id"].nunique()

# %%
sentences_data["reduced_points x"] = sentences_data["embedding"].apply(lambda x: x[0])
sentences_data["reduced_points y"] = sentences_data["embedding"].apply(lambda x: x[1])

# %%
sentences_data["original sentence length"] = sentences_data["original sentence"].apply(lambda x:len(x))
sentences_data["number words length"] = sentences_data["description"].apply(lambda x:len(x))


# %%
def run_plot(sentences_data, clustering_number):
    
    sentences_data["cluster_number"]=clustering_number
    
    sentences_data_samp = sentences_data.sample(n=500000, random_state=42)

    colour_by_list = sentences_data_samp["cluster_number"].tolist()

    ds_dict = dict(
        x=sentences_data_samp["reduced_points x"].tolist(),
        y=sentences_data_samp["reduced_points y"].tolist(),
        texts=sentences_data_samp["original sentence"].tolist(),
        cols=colour_by_list,
    )
    mapper = linear_cmap(
        field_name="cols",
        palette=Spectral6,
        low=min(colour_by_list),
        high=max(colour_by_list),
    )
    hover = HoverTool(tooltips=[("node", "@texts"),("colour by", "@cols")])  # ,("colour by", "@cols"),
    source = ColumnDataSource(ds_dict)
    p = figure(
        plot_width=500,
        plot_height=500,
        tools=[hover, WheelZoomTool(), BoxZoomTool(), SaveTool(), ResetTool()],
        title=f"Skills",
        toolbar_location="below",
    )
    p.circle(x="x", y="y", radius=0.01, alpha=0.05, source=source, color=mapper)
    show(p)


# %% [markdown]
# ## Get rid of mega cluster by length?

# %%
sentences_data["original sentence length"].plot.hist()

# %%
sentences_data["original sentence length grouped"] = sentences_data[
    "original sentence length"].apply(
    lambda x: (1 if x>100 else 0))




# %%
sentences_data_samp = sentences_data.sample(n=200000, random_state=42)

colour_by_list = sentences_data_samp["original sentence length grouped"].tolist()

ds_dict = dict(
    x=sentences_data_samp["reduced_points x"].tolist(),
    y=sentences_data_samp["reduced_points y"].tolist(),
    texts=sentences_data_samp["original sentence"].tolist(),
    cols=colour_by_list,
)
mapper = linear_cmap(
    field_name="cols",
    palette=Spectral6,
    low=min(colour_by_list),
    high=max(colour_by_list),
)
hover = HoverTool(tooltips=[("node", "@texts"),("colour by", "@cols")])  # ,("colour by", "@cols"),
source = ColumnDataSource(ds_dict)
p = figure(
    plot_width=500,
    plot_height=500,
    tools=[hover, WheelZoomTool(), BoxZoomTool(), SaveTool(), ResetTool()],
    title=f"Sentences coloured by if they are over (red) or under (blue) 100 characters",
    toolbar_location="below",
)
p.circle(x="x", y="y", radius=0.01, alpha=0.1, source=source, color=mapper)
show(p)

# %% [markdown]
# ## Remove larger sentences and then cluster

# %%
sentences_data_short = sentences_data[sentences_data["original sentence length"]<=100]
len(sentences_data_short)

# %%
sentences_data_short_samp = sentences_data_short.sample(n=500000, random_state=42)

ds_dict = dict(
    x=sentences_data_short_samp["reduced_points x"].tolist(),
    y=sentences_data_short_samp["reduced_points y"].tolist(),
    texts=sentences_data_short_samp["original sentence"].tolist()
)

hover = HoverTool(tooltips=[("node", "@texts")])
source = ColumnDataSource(ds_dict)
p = figure(
    plot_width=500,
    plot_height=500,
    tools=[hover, WheelZoomTool(), BoxZoomTool(), SaveTool(), ResetTool()],
    title=f"Sentences under 100 characters in 2D reduced space",
    toolbar_location="below",
)
p.circle(x="x", y="y", radius=0.01, alpha=0.05, source=source, color="black")
show(p)


# %% [markdown]
# ## Cluster and plot

# %%
def run_plot_by_col(sentences_data_samp, clustering_col_name, radius=0.01):

    colour_by_list = sentences_data_samp[clustering_col_name].tolist()

    ds_dict = dict(
        x=sentences_data_samp["reduced_points x"].tolist(),
        y=sentences_data_samp["reduced_points y"].tolist(),
        texts=sentences_data_samp["original sentence"].tolist(),
        cols=colour_by_list,
    )
    mapper = linear_cmap(
        field_name="cols",
        palette=Spectral6,
        low=min(colour_by_list),
        high=max(colour_by_list),
    )
    hover = HoverTool(tooltips=[("node", "@texts"),("colour by", "@cols")])  # ,("colour by", "@cols"),
    source = ColumnDataSource(ds_dict)
    p = figure(
        plot_width=500,
        plot_height=500,
        tools=[hover, WheelZoomTool(), BoxZoomTool(), SaveTool(), ResetTool()],
        title=f"Skills",
        toolbar_location="below",
    )
    p.circle(x="x", y="y", radius=radius, alpha=0.05, source=source, color=mapper)
    show(p)

# %%
# dbscan_eps = 0.01
# dbscan_min_samples = 10
            
# clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
# clustering_number = clustering.fit_predict(sentences_data_short["embedding"].tolist()).tolist()

# sentences_data_short[f"cluster_number_{dbscan_eps}_{dbscan_min_samples}"] = clustering_number


# %%
# count_clust = Counter(clustering_number)
# print(f"There are {len(set(clustering_number))} clusters")
# print(f"{count_clust.most_common(1)[0][1]/len(clustering_number)} of the points are in one cluster")
# print(f"{len([c for c in clustering_number if c!=-1])/len(clustering_number)} of the points are in clusters")

# %% [markdown]
# ## Cluster on sample to quickly get dbscan_eps

# %%
sentences_data_samp_experiment = sentences_data_short.sample(n=300000, random_state=42)

# %%
# dbscan_eps = 0.01
# dbscan_min_samples = 4
            
# clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
# clustering_number = clustering.fit_predict(sentences_data_samp_experiment["embedding"].tolist()).tolist()

# sentences_data_samp_experiment[f"cluster_number_{dbscan_eps}_{dbscan_min_samples}"] = clustering_number


# %% [markdown]
# ### Uncluster any clusters with <n sentences in?

# %%
# cluster_col = 'cluster_number_0.01_3'

# size_threshold = 5
# small_clusters = set([i for i,v in sentences_data_samp_experiment[cluster_col].value_counts().items() if v<size_threshold])
# print(f"There are {len(small_clusters)} out of {sentences_data_samp_experiment[cluster_col].nunique()} clusters which are of size < {size_threshold}")

# sentences_data_samp_experiment[f"{cluster_col}_no_small"] = sentences_data_samp_experiment[cluster_col].apply(lambda x: -1 if x in small_clusters else x)

# %% [markdown]
# ### Try some different options

# %%
# dbscan_eps_range = np.arange(0.015,0.025,0.0005)
# dbscan_min_samples_range = np.arange(1,10,1)
# for dbscan_eps in tqdm(dbscan_eps_range):
#     for dbscan_min_samples in dbscan_min_samples_range:
            
#         clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
#         clustering_number = clustering.fit_predict(sentences_data_samp_experiment["embedding"].tolist()).tolist()
#         del clustering
        
#         sentences_data_samp_experiment[f"cluster_number_{dbscan_eps}_{dbscan_min_samples}"] = clustering_number


# %%
# sentences_data_samp_experiment.to_csv("sentences_data_samp_experiment.csv")

# %%
sentences_data_samp_experiment.columns

# %%
# Load experiments from another time you ran these
sentences_data_samp_experiment = pd.read_csv("sentences_data_samp_experiment.csv")

# %%
# exp_colnames = sorted([s for s in sentences_data_samp_experiment.columns if 'cluster_number_' in s])

# # How many clusters (not -1) are of size > max threshold (don't want big clusters)
# large_threshold = 10000
# small_threshold = 10
# #  len([k for k, v in sentences_data_samp_experiment['cluster_number_0.015_5'].value_counts().items() if k!=-1 and v>large_threshold])
    
# func_large_size = lambda x: len([k for k, v in x.value_counts().items() if k!=-1 and v>large_threshold])
# func_small_size = lambda x: len([k for k, v in x.value_counts().items() if k!=-1 and v<small_threshold])
# func_num_not = lambda x: sum(x==-1)

# d1 = sentences_data_samp_experiment[exp_colnames].agg(['nunique', func_large_size]).T
# d1 = d1.rename(columns={"nunique":"Number of clusters", "<lambda>": "Number of really large clusters"})
# d2 = sentences_data_samp_experiment[exp_colnames].agg([func_small_size]).T
# d2 = d2.rename(columns={"<lambda>": "Number of really small clusters"})
# d3 = sentences_data_samp_experiment[exp_colnames].agg([func_num_not]).T
# d3 = d3.rename(columns={"<lambda>": "Size of -1 cluster"})
# cluster_params_summary = pd.concat([d1, d2, d3], axis =1)

# cluster_params_summary.reset_index(inplace=True)
# cluster_params_summary["dbscan_eps"] = cluster_params_summary['index'].apply(
#     lambda x: float(x.split("cluster_number_")[1].split("_")[0]))
# cluster_params_summary["dbscan_min_samples"] = cluster_params_summary['index'].apply(
#     lambda x: float(x.split("cluster_number_")[1].split("_")[1]))

# # Don't include the no small ones
# cluster_params_summary = cluster_params_summary[~cluster_params_summary['index'].str.contains("_no_small")]
# cluster_params_summary = cluster_params_summary[~cluster_params_summary['dbscan_eps'].isin([0.15,0.1])] 


# %%
# cluster_params_summary["Number of clusters log"] = np.log10(cluster_params_summary['Number of clusters'])
# cluster_params_summary["Number of really small clusters log"] = np.log10(cluster_params_summary['Number of really small clusters'])


# %%
# cluster_params_summary.to_csv("cluster_params_summary.csv")

# %%
cluster_params_summary = pd.read_csv("cluster_params_summary.csv")

# %%
fig, ((ax1, ax2,ax3), (ax4,ax5, ax6)) = plt.subplots(2,3, figsize=(20,12))

cluster_params_summary.plot.scatter(
    x="dbscan_eps", y= "dbscan_min_samples", c= "Number of really large clusters",
    colormap='viridis',  s=30,sharex=False, ax= ax1);

cluster_params_summary.plot.scatter(
    x="dbscan_eps", y= "dbscan_min_samples", c= "Size of -1 cluster",
    colormap='viridis', s=50,sharex=False, ax = ax4);

cluster_params_summary.plot.scatter(
    x="dbscan_eps", y= "dbscan_min_samples", c= "Number of really small clusters",
    colormap='viridis', s=50, sharex=False, ax = ax2);

cluster_params_summary.plot.scatter(
    x="dbscan_eps", y= "dbscan_min_samples", c= "Number of really small clusters log",
    colormap='viridis',  s=50, sharex=False, ax = ax5);

cluster_params_summary.plot.scatter(
    x="dbscan_eps", y= "dbscan_min_samples", c= 'Number of clusters',
    colormap='viridis', s=50,sharex=False, ax = ax3);

cluster_params_summary.plot.scatter(
    x="dbscan_eps", y= "dbscan_min_samples", c= 'Number of clusters log',
    colormap='viridis',  s=50,sharex=False, ax = ax6);

fig.savefig('../figures/nov_2021/clustering_params_exploration.pdf')
fig.savefig('../figures/nov_2021/clustering_params_exploration.png')

# %%
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,4))

param = "dbscan_eps"
y1 = cluster_params_summary[param].tolist()
y2 = cluster_params_summary[cluster_params_summary["Number of really large clusters"]==0][param]
ax1.hist([y1,y2],color=["black","red"],  alpha =0.6, density=True)
ax1.set_ylabel("Frequency")
ax1.set_xlabel(param)
plt.tight_layout()

param = "dbscan_min_samples"
y1 = cluster_params_summary[param].tolist()
y2 = cluster_params_summary[cluster_params_summary["Number of really large clusters"]==0][param]
ax2.hist([y1,y2],color=["black","red"], alpha =0.6, density=True, label=['All',"When number of large clusters is 0"])
ax2.set_ylabel("Frequency")
ax2.set_xlabel(param)
plt.tight_layout()
plt.legend();

fig.savefig('../figures/nov_2021/clustering_params_exploration_2.pdf')
fig.savefig('../figures/nov_2021/clustering_params_exploration_2.png')


# %% [markdown]
# - Shouldn't be too many cluster (last time 20k was a lot)
# - Shouldn't be any massive clusters
# - Shouldn't be too many small clusters

# %%
len(sentences_data_samp_experiment)

# %%
cluster_params_summary['Average size non -1 cluster'] = (len(sentences_data_samp_experiment)- cluster_params_summary['Size of -1 cluster'])/(cluster_params_summary['Number of clusters']-1)

# %%
len(cluster_params_summary)

# %%
cluster_params_summary.head(3)

# %%
cluster_params_summary[(
    cluster_params_summary["Number of really large clusters"]==0) & (
cluster_params_summary["Number of clusters"]>7000)& (
cluster_params_summary["Number of clusters"]<15000)& (
cluster_params_summary["Average size non -1 cluster"]>10)&(
cluster_params_summary["Size of -1 cluster"]<200000)]

# %% [markdown]
# ## Best parameters plot
# The parameters were optimised for a subsection of the data where:
# 1. Only short sentences (`sentences_data_short = sentences_data[sentences_data["original sentence length"]<=100]`)
# 2. Random 300000 sentences of the short ones (`sentences_data_samp_experiment = sentences_data_short.sample(n=300000, random_state=42)`)
#

# %%
dbscan_eps = 0.01
dbscan_min_samples = 4

# %%
cluster_params_summary[(
    cluster_params_summary["dbscan_eps"]==0.01) & (
cluster_params_summary["dbscan_min_samples"]==dbscan_min_samples)]

# %%
sentences_data_short_sample_used = sentences_data_short.sample(n=300000, random_state=42)
len(sentences_data_short_sample_used)

# %%
clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
clustering_number = clustering.fit_predict(sentences_data_short_sample_used["embedding"].tolist()).tolist()

# %%
print(f"{len(set(clustering_number))} clusters in {len(sentences_data_short_sample_used)} sentences")

# %%
df = sentences_data_short_sample_used

df["cluster_number"]=clustering_number

df_sample = df.sample(n=300000, random_state=42)
    
colour_by_list = df_sample["cluster_number"].tolist()

ds_dict = dict(
    x=df_sample["reduced_points x"].tolist(),
    y=df_sample["reduced_points y"].tolist(),
    texts=df_sample["original sentence"].tolist(),
    cols=colour_by_list,
)
mapper = linear_cmap(
    field_name="cols",
    palette=Spectral6,
    low=min(colour_by_list),
    high=max(colour_by_list),
)
hover = HoverTool(tooltips=[("node", "@texts"),("colour by", "@cols")])  # ,("colour by", "@cols"),
source = ColumnDataSource(ds_dict)
p = figure(
    plot_width=500,
    plot_height=500,
    tools=[hover, WheelZoomTool(), BoxZoomTool(), SaveTool(), ResetTool()],
    title=f"Skills",
    toolbar_location="below",
)
p.circle(x="x", y="y", radius=0.01, alpha=0.05, source=source, color=mapper)
show(p)

# %% [markdown]
# ## Remove the -1 clusters and plot

# %%
df_clusts = df[df["cluster_number"]!=-1]
len(df_clusts)

# %%
colour_by_list = df_clusts["cluster_number"].tolist()

ds_dict = dict(
    x=df_clusts["reduced_points x"].tolist(),
    y=df_clusts["reduced_points y"].tolist(),
    texts=df_clusts["original sentence"].tolist(),
    cols=colour_by_list,
)
mapper = linear_cmap(
    field_name="cols",
    palette=Spectral6,
    low=min(colour_by_list),
    high=max(colour_by_list),
)
hover = HoverTool(tooltips=[("node", "@texts"),("colour by", "@cols")])  # ,("colour by", "@cols"),
source = ColumnDataSource(ds_dict)
p = figure(
    plot_width=500,
    plot_height=500,
    tools=[hover, WheelZoomTool(), BoxZoomTool(), SaveTool(), ResetTool()],
    title=f"Skills",
    toolbar_location="below",
)
p.circle(x="x", y="y", radius=0.01, alpha=0.05, source=source, color=mapper)
show(p)

# %%
