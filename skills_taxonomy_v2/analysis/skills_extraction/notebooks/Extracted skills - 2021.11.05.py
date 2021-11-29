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
# # Have a look at the extracted skills

# %%
# cd ../../../..

# %%
from tqdm import tqdm
import json

import pandas as pd
import numpy as np
import boto3
import matplotlib.pyplot as plt

from skills_taxonomy_v2.getters.s3_data import load_s3_data
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
from bokeh.palettes import Plasma, magma, cividis, inferno, plasma, viridis, Spectral6, YlOrRd
from bokeh.transform import linear_cmap, log_cmap

bpl.output_notebook()

# %%
s3 = boto3.resource("s3")

# %% [markdown]
# ## Load the skill clusters and the sentence data

# %%
skill_data = load_s3_data(
        s3, BUCKET_NAME,
        'outputs/skills_extraction/extracted_skills/2021.11.05_sentences_skills_data.json'
    )
skill_data_df = pd.DataFrame(skill_data)
len(skill_data_df)

# %%
sentences_data = pd.DataFrame()
for i in tqdm(range(0, 8)):
    sentences_data_i = load_s3_data(
        s3, BUCKET_NAME,
        f"outputs/skills_extraction/reduced_embeddings/sentences_data_{i}.json"
    )
    sentences_data = pd.concat([sentences_data, pd.DataFrame(sentences_data_i)])

# %%
sentences_data["reduced_points x"] = sentences_data["embedding"].apply(lambda x: x[0])
sentences_data["reduced_points y"] = sentences_data["embedding"].apply(lambda x: x[1])
len(sentences_data)

# %% [markdown]
# ### Merge

# %%
clustered_sentences_data = pd.merge(
        sentences_data,
        skill_data_df,
        how='left', on=['job id', 'sentence id'])
len(clustered_sentences_data)

# %%
clustered_sentences_data["original sentence length"] = clustered_sentences_data["original sentence"].apply(lambda x:len(x))

# %%
max_length= 100
clustered_sentences_data = clustered_sentences_data[clustered_sentences_data["original sentence length"] <= max_length]
len(clustered_sentences_data)

# %% [markdown]
# ## Predictions
# - How often is fitted cluster number the same as the predicted (for the 300k in our sample)

# %%
clustered_sentences_data_train = clustered_sentences_data[(clustered_sentences_data["cluster_number"].notnull()) & (clustered_sentences_data["cluster_number"]!=-1)]
clustered_sentences_data_train["Merged clusters"] = clustered_sentences_data_train["Merged clusters"].astype(int)
clustered_sentences_data_train["cluster_number"] = clustered_sentences_data_train["cluster_number"].astype(int)
print(len(clustered_sentences_data_train))

# %%
# Predicted clusters don't include the -1 cluster

# %%
pred_diff = clustered_sentences_data_train[clustered_sentences_data_train["Cluster number predicted"]!=clustered_sentences_data_train["Merged clusters"]]
print(f"Out of 300000 sentences {len(clustered_sentences_data_train)} weren't in the -1 cluster")
print(f"{len(pred_diff)} out of {len(clustered_sentences_data_train)} sentences ({round(len(pred_diff)*100/len(clustered_sentences_data_train))}%) don't have the same prediction")


# %% [markdown]
# ## Plot

# %%
# Too many points to plot without crashing!
clustered_sentences_data_sample = clustered_sentences_data.sample(100000, random_state=42)

# %%
## plot all the points in grey
output_file(
    filename=f"outputs/skills_extraction/figures/2021_11_05/sents_col_clustfit.html"
)

ds_dict = dict(
    x=clustered_sentences_data_sample["reduced_points x"].tolist(),
    y=clustered_sentences_data_sample["reduced_points y"].tolist(),
    texts=clustered_sentences_data_sample["original sentence"].tolist(),
)
hover = HoverTool(tooltips=[("node", "@texts")])
source = ColumnDataSource(ds_dict)
p = figure(
    plot_width=500,
    plot_height=500,
    tools=[hover, WheelZoomTool(), BoxZoomTool(), SaveTool(), ResetTool()],
    title=f"Skills",
    toolbar_location="below",
)
p.circle(x="x", y="y", radius=0.01, alpha=0.1, source=source, color="red")

# Just the points which were used in cluster analysis
clustered_sentences_data_sample_x = clustered_sentences_data_sample[clustered_sentences_data_sample['cluster_number'].notnull()]
ds_dict2 = dict(
    x=clustered_sentences_data_sample_x["reduced_points x"].tolist(),
    y=clustered_sentences_data_sample_x["reduced_points y"].tolist(),
    texts=clustered_sentences_data_sample_x["original sentence"].tolist(),
)
source2 = ColumnDataSource(ds_dict2)
p.circle(x="x", y="y", radius=0.01, alpha=0.1, source=source2, color="green")

save(p)

# %%
output_file(
    filename=f"outputs/skills_extraction/figures/2021_11_05/sents_col_skill.html"
)

colour_by_list = clustered_sentences_data_sample["Cluster number predicted"].tolist()

ds_dict = dict(
    x=clustered_sentences_data_sample["reduced_points x"].tolist(),
    y=clustered_sentences_data_sample["reduced_points y"].tolist(),
    texts=clustered_sentences_data_sample["original sentence"].tolist(),
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
p.circle(x="x", y="y", radius=0.03, alpha=0.1, source=source, color=mapper)
save(p)

# %% [markdown]
# ## Plot single dot for each skill

# %%
skills_dict = load_s3_data(
        s3, BUCKET_NAME,
        'outputs/skills_extraction/extracted_skills/2021.11.05_skills_data.json'
    )
len(skills_dict)

# %%
# most frequent words per cluster

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
corpus = [' '.join(s['Sentences']) for s in skills_dict.values()]
skill_ids = [k for k in skills_dict.keys()]

X = tfidf.fit_transform(corpus)
feature_names = np.array(tfidf.get_feature_names())

def get_top_tf_idf_words(response, top_n=2):
    sorted_nzs = np.argsort(response.data)[:-(top_n+1):-1]
    return feature_names[response.indices[sorted_nzs]]
  
topwords_skill = [' '.join(get_top_tf_idf_words(response, 5).tolist()) for response in X]

for i, topwords in enumerate(topwords_skill):
    skill_num = skill_ids[i]
    skills_dict[skill_num]['Top words'] = topwords

# %%
xs = []
ys = []
sents = []
num_sents = []
labels = []
words = []
for k, v in skills_dict.items():
    xs.append(v['Centroid'][0])
    ys.append(v['Centroid'][1])
    sents.append(v['Sentences'][0:10])
    num_sents.append(len(v['Sentences']))
    labels.append(k)
    words.append(v["Top words"])
ds_dict = {'x': xs, 'y': ys, 'texts': sents, 'num_sents': num_sents, 'labels': labels, 'words':words}    

# %%
plt.hist(num_sents, bins=100);

# %%
output_file(
    filename=f"outputs/skills_extraction/figures/2021_11_05/skills.html"
)

hover = HoverTool(tooltips=[("node", "@texts"),("colour by", "@num_sents")])
source = ColumnDataSource(ds_dict)
p = figure(
    plot_width=500,
    plot_height=500,
    tools=[hover, WheelZoomTool(), BoxZoomTool(), SaveTool(), ResetTool()],
    title=f"Skills",
    toolbar_location="below",
)
p.circle(x="x", y="y", radius=0.05, alpha=0.1, source=source, color='black')

save(p)

# %%
output_file(
    filename=f"outputs/skills_extraction/figures/2021_11_05/skills_col_numsent.html"
)

mapper = log_cmap(
    field_name="num_sents",
    palette=Spectral6,
    low=min(num_sents),
    high=max(num_sents),
)
hover = HoverTool(tooltips=[("node", "@texts"),("colour by", "@num_sents")])
source = ColumnDataSource(ds_dict)
p = figure(
    plot_width=500,
    plot_height=500,
    tools=[hover, WheelZoomTool(), BoxZoomTool(), SaveTool(), ResetTool()],
    title=f"Skills coloured by number of sentences",
    toolbar_location="below",
)
p.circle(x="x", y="y", radius=0.05, alpha=0.8, source=source, color=mapper)
color_bar = ColorBar(color_mapper=mapper['transform'], width=8,  location=(0,0), title='Number of sentences in skill')

p.add_layout(color_bar, 'right')
save(p)

# %%
output_file(
    filename=f"outputs/skills_extraction/figures/2021_11_05/skills_col_numsent_tfidf.html"
)

mapper = log_cmap(
    field_name="num_sents",
    palette=Spectral6,
    low=min(num_sents),
    high=max(num_sents),
)
hover = HoverTool(tooltips=[("node", "@words"),("colour by", "@num_sents")])
source = ColumnDataSource(ds_dict)
p = figure(
    plot_width=500,
    plot_height=500,
    tools=[hover, WheelZoomTool(), BoxZoomTool(), SaveTool(), ResetTool()],
    title=f"Skills coloured by number of sentences (hover for 5 top TF-IDF)",
    toolbar_location="below",
)
p.circle(x="x", y="y", radius=0.05, alpha=0.8, source=source, color=mapper)
color_bar = ColorBar(color_mapper=mapper['transform'], width=8,  location=(0,0), title='Number of sentences in skill')

p.add_layout(color_bar, 'right')
save(p)

# %%
