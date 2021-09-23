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
# Still need to update once I have skill names and ESCO names.
#
# One place for all things related to skills extraction figures and high level analysis.
#
# 1. How many skills
#
# - There are x sentences that went into creating skills
# - There are x skills found
#
# 2. How many not in skills
#
# - x proportion of sentences arent in a cluster
#
# 3. Plot of skills
# 4. Examples of skills
# 5. Number of sentences in skills
#
# - The mean number of sentences for each skills is x
# - The median number of sentences for each skills is x
# - There are x skills with more than x sentences
#
# 6. Number of skills in job adverts
#
# - There are x unique job adverts with skills in
# - The mean number of unique skills per job advert is x
# - The median number of unique skills per job advert is x
# - There are x job adverts with more than x skills
#
# 7. Mapping to ESCO
#
# - x out of x (x%) TK skills were linked with ESCO skills
# - 0 of these were linked to multiple ESCO skills
# - x out of x (x%) ESCO skills were linked with TK skills
# - x of these were linked to multiple TK skills
#

# %%
# cd ../../../..

# %%
from skills_taxonomy_v2 import BUCKET_NAME, custom_stopwords_dir

# %%
from skills_taxonomy_v2.getters.s3_data import load_s3_data,get_s3_data_paths

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

# %%
file_name_date = '2021.08.31'

# %% [markdown]
# ### 1. The sentences clustered into skills

# %%
bucket_name = "skills-taxonomy-v2"
s3 = boto3.resource("s3")

# %%
sentence_clusters = load_s3_data(s3, bucket_name, f'outputs/skills_extraction/extracted_skills/{file_name_date}_sentences_data.json')


# %%
sentence_clusters = pd.DataFrame(sentence_clusters)
sentence_clusters.head(3)

# %%
sentence_clusters['Cluster number'].nunique()

# %% [markdown]
# ### 2. Skills data

# %%
skills_data = load_s3_data(s3, bucket_name, f'outputs/skills_extraction/extracted_skills/{file_name_date}_skills_data.json')
len(skills_data)

# %% [markdown]
# ### 3. An original file before filtering out sentences with low length

# %%
sentence_embeddings_dirs = get_s3_data_paths(
        s3, bucket_name, "outputs/skills_extraction/word_embeddings/data/2021.08.31/", file_types=["*.json"]
    )

# %%
sentence_embeddings_dirs = sentence_embeddings_dirs[0:2]

# %%
original_sentences = {}
for embedding_dir in sentence_embeddings_dirs:
    if "original_sentences.json" in embedding_dir:
        original_sentences.update(load_s3_data(s3, bucket_name, embedding_dir))

# %%
original_sentences['546245933490949713']

# %%
mask_seq="[MASK]"
prop_not_masked_threshold=0.5

bad_sentences = []
for embedding_dir in tqdm(sentence_embeddings_dirs):
    if "embeddings.json" in embedding_dir:
        sentence_embeddings = load_s3_data(s3, bucket_name, embedding_dir)
        # Only output data for this sentence if it matches various conditions
        print('here')
        count_keep = 0
        for job_id, sent_id, words, embedding in sentence_embeddings:
            words_without_mask = words.replace(mask_seq, "")
            prop_not_masked = len(words_without_mask) / len(words)
            if prop_not_masked < prop_not_masked_threshold:
                original_sentence = original_sentences[str(sent_id)]
                if len(original_sentence) > 300:
                    bad_sentences.append([original_sentence, words])

# %%
len(embedding)

# %%
bad_sentences[0]

# %% [markdown]
# ## Some random sentences

# %%
with open(custom_stopwords_dir) as file:
    custom_stopwords = file.read().splitlines()

# %%
random.seed(42)
sentences = random.sample(sentence_clusters['original sentence'].tolist(), 10)

# %%
for sentence in sentences:
    mask_seq = process_sentence_mask(
                        sentence,
                        nlp,
                        bert_vectorizer,
                        token_len_threshold=20,
                        stopwords=stopwords.words(),
                        custom_stopwords=custom_stopwords
                    )
    print('---')
    print(sentence)
    print(mask_seq)

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
output_file(filename=f"outputs/skills_extraction/figures/{file_name_date}_reduced_sentences.html")

reduced_x = sentence_clusters['reduced_points x'].tolist()
reduced_y = sentence_clusters['reduced_points y'].tolist()
color_palette = viridis

ds_dict = dict(
    x=reduced_x,
    y=reduced_y,
    texts=sentence_clusters["description"].tolist(),
)
hover = HoverTool(
    tooltips=[
        ("Sentence", "@texts"),
    ]
)
source = ColumnDataSource(ds_dict)

p = figure(
    plot_width=500,
    plot_height=500,
    tools=[hover, ResetTool(), WheelZoomTool(), BoxZoomTool(), SaveTool()],
    title=f"Sentences embeddings reduced to 2D space",
    toolbar_location="below",
)
p.circle(
    x="x",
    y="y",
    radius=0.01,
    alpha=0.1,
    source=source,
    color='black',
)
p.xaxis.visible = False
p.xgrid.visible = False
p.yaxis.visible = False
p.ygrid.visible = False

save(p)

# %%
output_file(filename=f"outputs/skills_extraction/figures/{file_name_date}_skill_clusters_all_sentences.html")

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
    tools=[hover, ResetTool(), WheelZoomTool(), BoxZoomTool(), SaveTool()],
    title=f"Sentences in 2D space, coloured by skill (including those not in a skill)",
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
p.xaxis.visible = False
p.xgrid.visible = False
p.yaxis.visible = False
p.ygrid.visible = False

save(p)

# %%
output_file(filename=f"outputs/skills_extraction/figures/{file_name_date}_skill_clusters_not_clustered.html")

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
    tools=[hover, ResetTool(), WheelZoomTool(), BoxZoomTool(), SaveTool()],
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
p.xaxis.visible = False
p.xgrid.visible = False
p.yaxis.visible = False
p.ygrid.visible = False
save(p)

# %%
sentence_clusters_notnone = sentence_clusters[sentence_clusters["Cluster number"]!=-1]

# %%
output_file(filename=f"outputs/skills_extraction/figures/{file_name_date}_skill_clusters.html")

colors_by_labels = sentence_clusters_notnone["Cluster number"].astype(str).tolist()
reduced_x = sentence_clusters_notnone['reduced_points x'].tolist()
reduced_y = sentence_clusters_notnone['reduced_points y'].tolist()
color_palette = viridis

ds_dict = dict(
    x=reduced_x,
    y=reduced_y,
    texts=sentence_clusters_notnone["description"].tolist(),
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
    tools=[hover, ResetTool(), WheelZoomTool(), BoxZoomTool(), SaveTool()],
    title=f"Sentences in 2D space, coloured by skill",
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
p.xaxis.visible = False
p.xgrid.visible = False
p.yaxis.visible = False
p.ygrid.visible = False

save(p)

# %%
skills_clusters = sentence_clusters_notnone.groupby("Cluster number")[
    ['reduced_points x', 'reduced_points y']].mean().reset_index()
skills_clusters['Skills name'] = skills_clusters['Cluster number'].apply(lambda x: skills_data[str(x)]['Skills name'])
skills_clusters['Examples'] = skills_clusters['Cluster number'].apply(lambda x: skills_data[str(x)]['Examples'])
skills_clusters.head(2)


# %%
output_file(filename=f"outputs/skills_extraction/figures/{file_name_date}_skill_clusters_average.html")

colors_by_labels = skills_clusters["Cluster number"].astype(str).tolist()
reduced_x = skills_clusters['reduced_points x'].tolist()
reduced_y = skills_clusters['reduced_points y'].tolist()
color_palette = viridis

ds_dict = dict(
    x=reduced_x,
    y=reduced_y,
    texts=skills_clusters["Skills name"].tolist(),
    examples = skills_clusters["Examples"].tolist(),
    label=colors_by_labels,
)
hover = HoverTool(
    tooltips=[
        ("Skill name", "@texts"),
        ("Examples", "@examples"),
        ("Skill number", "@label"),
    ]
)
source = ColumnDataSource(ds_dict)

p = figure(
    plot_width=500,
    plot_height=500,
    tools=[hover, ResetTool(), WheelZoomTool(), BoxZoomTool(), SaveTool()],
    title=f"Skills",
    toolbar_location="below",
)
p.circle(
    x="x",
    y="y",
    radius=0.04,
    alpha=0.2,
    source=source,
    color="black"
)
p.xaxis.visible = False
p.xgrid.visible = False
p.yaxis.visible = False
p.ygrid.visible = False

save(p)

# %%
from bokeh.models import ColumnDataSource, Grid, LinearAxis, Plot, Text


# %%
output_file(filename=f"outputs/skills_extraction/figures/{file_name_date}_skill_clusters_average_labelled.html")


colors_by_labels = skills_clusters["Cluster number"].astype(str).tolist()
reduced_x = skills_clusters['reduced_points x'].tolist()
reduced_y = skills_clusters['reduced_points y'].tolist()
color_palette = viridis

ds_dict = dict(
    x=reduced_x,
    y=reduced_y,
    texts=skills_clusters["Skills name"].tolist(),
    examples = skills_clusters["Examples"].tolist(),
    label=colors_by_labels,
)
hover = HoverTool(
    tooltips=[
        ("Skill name", "@texts"),
        ("Examples", "@examples"),
        ("Skill number", "@label"),
    ]
)
source = ColumnDataSource(ds_dict)

p = figure(
    plot_width=500,
    plot_height=500,
    tools=[hover, ResetTool(), WheelZoomTool(), BoxZoomTool(), SaveTool()],
    title=f"Skills",
    toolbar_location="below",
)

p.circle(
    x="x",
    y="y",
    radius=0.04,
    alpha=0.15,
    source=source,
    color="grey"
)

skills_clusters_sample_n = [13189,9866,1525, 5077, 6418, 192, 4235, 4566, 13536, 5183, 1556, 7613,
                            2744, 1768, 18415, 12386, 6760, 2970, 9588, 6213, 11451,
                            13347, 15905, 8898, 1674, 3876, 9296, 18040, 8253, 16692, 8584, 8556,8888, 8497,
                           242, 11136]
skills_clusters_sample= skills_clusters.copy()[skills_clusters['Cluster number'].isin(skills_clusters_sample_n)]
ds_dict_text = dict(
    x=skills_clusters_sample['reduced_points x'].tolist(),
    y=skills_clusters_sample['reduced_points y'].tolist(),
    texts=skills_clusters_sample["Skills name"].tolist(),
    label=skills_clusters_sample['Cluster number'].tolist(),
)
source_text = ColumnDataSource(ds_dict_text)

glyph = Text(x="x", y="y", text="texts", angle=0, text_color="black", text_font_size="7pt")
p.add_glyph(source_text, glyph)



p.xaxis.visible = False
p.xgrid.visible = False
p.yaxis.visible = False
p.ygrid.visible = False

save(p)

# %% [markdown]
# ## Skill examples

# %%
skill_id = '-1'
sentence_clusters[sentence_clusters["Cluster number"]==int(skill_id)]['original sentence'].tolist()[100:110]

# %%
data_skills = []
for k,v in skills_data.items():
    if ('excel' in v['Skills name']) and ('microsoft' in v['Skills name']):
        data_skills.append(k)
data_skills[0:10]

# %%
skill_id = '107'
# print(skills_data[skill_id])
print(sentence_clusters[sentence_clusters["Cluster number"]==int(skill_id)]['original sentence'].tolist()[0:10])


# %%
data_skills = []
for k,v in skills_data.items():
    if ('unit' in v['Skills name']) and ('test' in v['Skills name']):
        data_skills.append(k)
data_skills[0:10]

# %%
skill_id = '18625'
print(skills_data[skill_id])
print(sentence_clusters[sentence_clusters["Cluster number"]==int(skill_id)]['original sentence'].tolist())

# %%
data_skills = []
for k,v in skills_data.items():
    if ('machine' in v['Skills name']) and ('learning' in v['Skills name']):
        data_skills.append(k)

# %%
data_skills

# %%
skill_id = '1228'
# print(skills_data[skill_id])
print(sentence_clusters[sentence_clusters["Cluster number"]==int(skill_id)]['original sentence'].tolist()[0:10])

# %%
skill_id = '13347'
print(skills_data[skill_id]['Skills name'])
print(sentence_clusters[sentence_clusters["Cluster number"]==int(skill_id)]['original sentence'].tolist()[0:100])

# %%
skill_id = '275'
print(skills_data[skill_id]['Skills name'])
print(sentence_clusters[sentence_clusters["Cluster number"]==int(skill_id)]['original sentence'].tolist()[0:10])

# %% [markdown]
# ## How many sentences in each skill?
# - Are those with loads of skills junk?

# %%
skill_lengths = {}
for k, v in skills_data.items():
    skill_lengths[k] = len(v['Texts'])


# %%
print(f"The mean number of sentences for each skills is {np.mean(list(skill_lengths.values()))}")
print(f"The median number of sentences for each skills is {np.median(list(skill_lengths.values()))}")
n_max=200
print(f"There are {len([s for s in skill_lengths.values() if s>n_max])} skills with more than {n_max} sentences")


# %%
plt.hist(skill_lengths.values(), bins =10);

# %%
plt.hist([s for s in skill_lengths.values() if s<100],
         bins =100, color=[255/255,90/255,0]);
plt.xlabel("Number of sentences in a skill cluster");
plt.ylabel("Number of skills");
plt.savefig('outputs/skills_extraction/figures/num_sent_in_skill.pdf',bbox_inches='tight')

# %%
len([i for i, s in skill_lengths.items() if s<20])/len(skill_lengths)

# %%
[i for i, s in skill_lengths.items() if s>2000]

# %%
n='21'
skills_data[n]['Skills name']
skills_data[n]['Texts'][0:10]

# %% [markdown]
# ## How many skills in each job advert?

# %%
num_skills = list(sentence_clusters_notnone.groupby('job id')['Cluster number'].nunique())

# %%
print(f"There are {len(num_skills)} unique job adverts with skills in")
print(f"The mean number of unique skills per job advert is {np.mean(num_skills)}")
print(f"The median number of unique skills per job advert is {np.median(num_skills)}")
n_max = 30
print(f"There are {len([s for s in num_skills if s>n_max])} job adverts with more than {n_max} skills")

