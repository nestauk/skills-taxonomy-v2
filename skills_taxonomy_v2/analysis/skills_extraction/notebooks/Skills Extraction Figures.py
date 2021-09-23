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
# One place for all things related to skills extraction figures and high level analysis.
#
# 1. How many skills
#
# - There are 392625 sentences that went into creating skills
# - There are 24912 skills found
#
# 2. How many not in skills
#
# - 0.108378223495702 proportion of sentences arent in a cluster
#
# 3. Plot of skills
# 4. Examples of skills
# 5. Number of sentences in skills
#
# - The mean number of sentences for each skills is 14.390775529865126
# - The median number of sentences for each skills is 3.0
# - There are 235 skills with more than 200 sentences
#
# 6. Number of skills in job adverts
#
# - There are 78674 unique job adverts with skills in
# - The mean number of unique skills per job advert is 4.2201235478048655
# - The median number of unique skills per job advert is 3.0
# - There are 42 job adverts with more than 30 skills
#
# 7. Mapping to ESCO
#
# - 19796 out of 24912 (79%) TK skills were linked with ESCO skills
# - 0 of these were linked to multiple ESCO skills
# - 4301 out of 13958 (31%) ESCO skills were linked with TK skills
# - 2450 of these were linked to multiple TK skills
#

# %%
# cd ../../../..

# %%
from skills_taxonomy_v2.getters.s3_data import load_s3_data

from skills_taxonomy_v2.pipeline.skills_extraction.get_sentence_embeddings_utils import (
    process_sentence_mask,
)

from skills_taxonomy_v2.pipeline.sentence_classifier.sentence_classifier import (
    BertVectorizer,
)

# %%
import json
import pickle
import random

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn import metrics
import umap.umap_ as umap
import boto3
import nltk
from nltk.corpus import stopwords
import spacy

import bokeh.plotting as bpl
from bokeh.plotting import ColumnDataSource, figure, output_file, show
from bokeh.models import (
    BoxZoomTool,
    WheelZoomTool,
    HoverTool,
    SaveTool,
    Label,
    CategoricalColorMapper,
    ColorBar,
    ColumnDataSource,
    LinearColorMapper,
)
from bokeh.io import output_file, reset_output, save, export_png, show
from bokeh.resources import CDN
from bokeh.embed import file_html
from bokeh.palettes import Plasma, magma, cividis, inferno, plasma, viridis, Spectral6
from bokeh.transform import linear_cmap

bpl.output_notebook()

# %%
nlp = spacy.load("en_core_web_sm")

bert_vectorizer = BertVectorizer(
        bert_model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
        multi_process=True,
    )
bert_vectorizer.fit()

# %% [markdown]
# ## Load data

# %% [markdown]
# ### 1. The sentences clustered into skills

# %%
bucket_name = "skills-taxonomy-v2"
s3 = boto3.resource("s3")

# %%
sentence_clusters = load_s3_data(s3, bucket_name, 'outputs/skills_extraction/data/2021.08.02_sentences_data.json')

# %%
sentence_clusters = pd.DataFrame(sentence_clusters)
sentence_clusters.head(3)

# %% [markdown]
# ### 2. Skills data

# %%
skills_data = load_s3_data(s3, bucket_name, 'outputs/skills_extraction/data/2021.08.02_skills_data.json')
len(skills_data)

# %% [markdown]
# ### 3. The ESCO mapping

# %%
esco2tk_mapper = load_s3_data(s3, bucket_name,
                              'outputs/skills_extraction/data/2021.08.02_esco2tk_mapper.json')
print(len(esco2tk_mapper))
tk2esco_mapper = load_s3_data(s3, bucket_name,
                              'outputs/skills_extraction/data/2021.08.02_tk2esco_mapper.json')
len(tk2esco_mapper)

# %%
esco_ID2skill = load_s3_data(s3, bucket_name,
                              'outputs/skills_extraction/data/2021.08.02_esco_ID2skill.json')
len(esco_ID2skill)

# %% [markdown]
# ## Some random sentences

# %%
random.seed(42)
sentences = random.sample(sentence_clusters['original sentence'].tolist(), 10)
sentences

# %%
masked_sentences = []
for sentence in sentences:
    masked_sentences.append(process_sentence_mask(
                        sentence,
                        nlp,
                        bert_vectorizer,
                        token_len_threshold=20,
                        stopwords=stopwords.words(),
                    ))

# %%
masked_sentences

# %% [markdown]
# ### Analysis
# 1. How many skills
# 2. How many not in skills
# 3. Siloutte score for clustering

# %%
print(f'There are {len(sentence_clusters)} sentences that went into creating skills')
print(f'There are {sentence_clusters["Cluster number"].nunique()} skills found')
print(f'{sum(sentence_clusters["Cluster number"]==-1)/len(sentence_clusters)} proportion of sentences arent in a cluster')

# %% [markdown]
# ## Plot

# %%
output_file(filename="outputs/skills_extraction/figures/skill_clusters.html")

colors_by_labels = sentence_clusters["Cluster number"].astype(str).tolist()
reduced_x = sentence_clusters['reduced_points x'].tolist()
reduced_y = sentence_clusters['reduced_points y'].tolist()
color_palette = viridis

ds_dict = dict(
    x=reduced_x,
    y=reduced_y,
    texts=sentence_clusters["description"].tolist(),
    label=colors_by_labels,
)
hover = HoverTool(
    tooltips=[
        ("Sentence", "@texts"),
        ("Skill cluster", "@label"),
    ]
)
source = ColumnDataSource(ds_dict)
unique_colors = list(set(colors_by_labels))
num_unique_colors = len(unique_colors)

# color_palette_cols = color_palette(len(unique_colors))
# color_mapper = CategoricalColorMapper(factors=unique_colors, palette=color_palette_cols)

color_mapper = LinearColorMapper(palette="Turbo256", low=0, high=len(unique_colors) + 1)

p = figure(
    plot_width=500,
    plot_height=500,
    tools=[hover, WheelZoomTool(), BoxZoomTool(), SaveTool()],
    title=f"Sentences clustered into skills",
    toolbar_location="below",
)
p.circle(
    x="x",
    y="y",
    radius=0.01,
    alpha=0.5,
    source=source,
    color={"field": "label", "transform": color_mapper},
)

save(p)

# %%
output_file(filename="outputs/skills_extraction/figures/skill_clusters_not_clustered.html")

colors_by_labels = sentence_clusters["Cluster number"].astype(str).tolist()
ds_dict_1 = dict(
    x=reduced_x,
    y=reduced_y,
    texts=sentence_clusters["description"].tolist(),
    label=colors_by_labels,
)
source1 = ColumnDataSource(ds_dict_1)

not_clust_ix = sentence_clusters[
    sentence_clusters["Cluster number"] == -1
].index.tolist()
ds_dict_2 = dict(
    x=sentence_clusters.iloc[not_clust_ix]["reduced_points x"].tolist(),
    y=sentence_clusters.iloc[not_clust_ix]["reduced_points y"].tolist(),
    texts=sentence_clusters.iloc[not_clust_ix]["description"].tolist(),
    label=[colors_by_labels[i] for i in not_clust_ix],
)
source2 = ColumnDataSource(ds_dict_2)

hover = HoverTool(
    tooltips=[
        ("Sentence", "@texts"),
        ("Clustered", "@label"),
    ]
)

p = figure(
    plot_width=500,
    plot_height=500,
    tools=[hover, WheelZoomTool(), BoxZoomTool(), SaveTool()],
    title=f"Sentences not clustered into skills",
    toolbar_location="below",
)
p.circle(
    x="x",
    y="y",
    radius=0.01,
    alpha=0.5,
    source=source1,
    color="grey",
)
p.circle(
    x="x",
    y="y",
    radius=0.01,
    alpha=0.5,
    source=source2,
    color="red",
)
save(p)

# %% [markdown]
# ## Skill examples

# %%
skill_id = '10'
print(skills_data[skill_id])
print(sentence_clusters[sentence_clusters["Cluster number"]==int(skill_id)]['original sentence'].tolist())


# %%
skill_id = '20'
print(skills_data[skill_id])
print(sentence_clusters[sentence_clusters["Cluster number"]==int(skill_id)]['original sentence'].tolist())


# %%
skill_id = '5'
print(skills_data[skill_id])
print(sentence_clusters[sentence_clusters["Cluster number"]==int(skill_id)]['original sentence'].tolist())


# %%
data_skills = []
for k,v in skills_data.items():
    if ('machine' in v['Skill name']) and ('learning' in v['Skill name']):
        data_skills.append(k)

# %%
skill_id = '9487'
print(skills_data[skill_id])
print(sentence_clusters[sentence_clusters["Cluster number"]==int(skill_id)]['original sentence'].tolist())

# %% [markdown]
# ## How many sentences in each skill?
# - Are those with loads of skills junk?

# %%
skill_lengths = {}
for k, v in skills_data.items():
    skill_lengths[k] = len(v['text'])


# %%
print(f"The mean number of sentences for each skills is {np.mean(list(skill_lengths.values()))}")
print(f"The median number of sentences for each skills is {np.median(list(skill_lengths.values()))}")
n_max=200
print(f"There are {len([s for s in skill_lengths.values() if s>n_max])} skills with more than {n_max} sentences")


# %%
plt.hist(skill_lengths.values(), bins =1000);

# %%
plt.hist([s for s in skill_lengths.values() if s<200], bins =100);

# %%
[i for i, s in skill_lengths.items() if s>2000]

# %%
skills_data['123']

# %% [markdown]
# ## How many skills in each job advert?

# %%
sentence_clusters_notnone = sentence_clusters[sentence_clusters['Cluster number']!=-1]

# %%
num_skills = list(sentence_clusters_notnone.groupby('job id')['Cluster number'].nunique())

# %%
print(f"There are {len(num_skills)} unique job adverts with skills in")
print(f"The mean number of unique skills per job advert is {np.mean(num_skills)}")
print(f"The median number of unique skills per job advert is {np.median(num_skills)}")
n_max = 30
print(f"There are {len([s for s in num_skills if s>n_max])} job adverts with more than {n_max} skills")


# %% [markdown]
# ## ESCO mappings

# %%
print(f"{len(tk2esco_mapper)} out of {len(skills_data)} ({round(len(tk2esco_mapper)*100/len(skills_data))}%) TK skills were linked with ESCO skills")
print(f"{len([k for k,v in tk2esco_mapper.items() if len(v)>1])} of these were linked to multiple ESCO skills")
print(f"{len(esco2tk_mapper)} out of {len(esco_ID2skill)} ({round(len(esco2tk_mapper)*100/len(esco_ID2skill))}%) ESCO skills were linked with TK skills")
print(f"{len([k for k,v in esco2tk_mapper.items() if len(v)>1])} of these were linked to multiple TK skills")

# %%
not_in_tk = list(set(esco_ID2skill.keys()).difference(set(esco2tk_mapper.keys())))
for esco_id in not_in_tk[0:10]:
    print(esco_ID2skill[esco_id])

# %%
not_in_esco = list(set(skills_data.keys()).difference(set(tk2esco_mapper.keys())))
for tk_id in not_in_esco[0:10]:
    print(skills_data[tk_id]['Skill name'])

# %%
tk_mapper_to_esco = sentence_clusters['Cluster number'].apply(lambda x: 1 if str(x) in tk2esco_mapper else 0).tolist()
sentence_clusters['Cluster in ESCO'] = tk_mapper_to_esco

# %%
sentence_clusters.head(2)

# %%
## Where are the mapped TK skills to ESCO?
output_file(filename="outputs/skills_extraction/figures/skill_clusters_in_esco_map.html")

colors_by_labels = sentence_clusters["Cluster in ESCO"].astype(str).tolist()
reduced_x = sentence_clusters['reduced_points x'].tolist()
reduced_y = sentence_clusters['reduced_points y'].tolist()
color_palette = viridis

ds_dict = dict(
    x=reduced_x,
    y=reduced_y,
    texts=sentence_clusters["description"].tolist(),
    label=colors_by_labels,
)
hover = HoverTool(
    tooltips=[
        ("Sentence", "@texts"),
        ("Skill cluster mapped to ESCO", "@label"),
    ]
)
source = ColumnDataSource(ds_dict)
unique_colors = list(set(colors_by_labels))
num_unique_colors = len(unique_colors)

# color_palette_cols = color_palette(len(unique_colors))
# color_mapper = CategoricalColorMapper(factors=unique_colors, palette=color_palette_cols)

color_mapper = LinearColorMapper(palette="Turbo256", low=0, high=len(unique_colors) + 1)

p = figure(
    plot_width=500,
    plot_height=500,
    tools=[hover, WheelZoomTool(), BoxZoomTool(), SaveTool()],
    title=f"Whether the sentence are mapped to ESCO skills",
    toolbar_location="below",
)
p.circle(
    x="x",
    y="y",
    radius=0.01,
    alpha=0.5,
    source=source,
    color={"field": "label", "transform": color_mapper},
)

save(p)

# %%
