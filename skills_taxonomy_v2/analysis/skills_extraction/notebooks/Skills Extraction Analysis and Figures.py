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
# - There are 6685 skills found
# - There are 10378654 sentences with embeddings given
# - 3633001 of the sentences (35.0%) were assigned a skill
# - 6745653 of the sentences (65.0%) were skill number -2
# - There were 2624694 unique job adverts (4.17% of all) in all sentences used
# - There were 1287661 unique job adverts (2.05% of all, 49.06% of those with sentences) with a skill given
# - There were 2371657 unique job adverts (3.77% of all, 90.36% of those with sentences) with skill number -2
#
# 1. How many skills
#
# - There are 10378654 sentences were predicted on
# - There are 6685 skills found (inc the -2 cluster)
#
# 2. How many not in skills
#
# - 65% of sentences arent in a cluster
#
# 3. Plot of skills
# 4. Examples of skills
# 5. Number of sentences in skills
#
# - The mean number of sentences for each skills is 543.4556469708302
# - The median number of sentences for each skills is 366.0
# - There are 6153 skills with more than 200 sentences
#
# 6. Number of skills in job adverts
#
# - There are 2624694 unique job adverts with skills in (inc -2 skill)
# - There are 10378654 unique sentences with skills in (inc -2 skill)
# - There are 1287661 unique job adverts with skills in (not inc -2 skill)
# - There are 3633001 unique sentences with skills in (not inc -2 skill)
# - The mean number of unique skills per job advert is 2.801606168083059
# - The median number of unique skills per job advert is 2.0
# - There are 378 job adverts with more than 30 skills
#

# %%
# cd ../../../..

# %%
from bokeh.models import ColumnDataSource, Grid, LinearAxis, Plot, Text
from skills_taxonomy_v2 import BUCKET_NAME, custom_stopwords_dir

# %%
from skills_taxonomy_v2.getters.s3_data import load_s3_data, get_s3_data_paths

from skills_taxonomy_v2.pipeline.skills_extraction.get_sentence_embeddings_utils import (
    process_sentence_mask,
)

from skills_taxonomy_v2.pipeline.sentence_classifier.sentence_classifier import (
    BertVectorizer,
)
from skills_taxonomy_v2.pipeline.skills_taxonomy.build_taxonomy_utils import get_level_names


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
    bert_model_name="sentence-transformers/all-MiniLM-L6-v2",
    multi_process=True,
)
bert_vectorizer.fit()

# %% [markdown]
# ## Load data

# %%
file_name_date = "2022.01.14"

# %% [markdown]
# ### 1. The sentences clustered into skills

# %%
bucket_name = "skills-taxonomy-v2"
s3 = boto3.resource("s3")

# %%
reduced_embeddings_dir=f"outputs/skills_extraction/reduced_embeddings/{file_name_date}"
clustered_sentences_path=f"outputs/skills_extraction/extracted_skills/{file_name_date}_sentences_skills_data_lightweight.json"

# %%
# The sentences ID + cluster num
sentence_embs = load_s3_data(s3, bucket_name, clustered_sentences_path)
sentence_embs = pd.DataFrame(sentence_embs, columns=['job id', 'sentence id',  'Cluster number predicted'])

# %%
sentence_embs[sentence_embs['Cluster number predicted']==0]

# %%
print(len(sentence_embs))
# A sample of the sentences
# The data is so big, so need to do most plots on a sample
sentence_embs_sample = sentence_embs.sample(n=2000000, random_state=1)
print(len(sentence_embs_sample)) 

# %%
sample_sent_ids = set(sentence_embs_sample['sentence id'].unique())

# %%
len(sample_sent_ids)

# %%
# Get the reduced embeddings + sentence texts and the sentence IDs

reduced_embeddings_paths = get_s3_data_paths(
    s3,
    bucket_name,
    reduced_embeddings_dir,
    file_types=["*sentences_data_*.json"]
    )

sample_sentences_data = pd.DataFrame()
for reduced_embeddings_path in tqdm(reduced_embeddings_paths):
    sentences_data_i = load_s3_data(
        s3, bucket_name,
        reduced_embeddings_path
    )
    sentences_data_i_df = pd.DataFrame(sentences_data_i)
    sample_sentences_data = pd.concat([sample_sentences_data, sentences_data_i_df[sentences_data_i_df['sentence id'].isin(sample_sent_ids)]])
sample_sentences_data.reset_index(drop=True, inplace=True)

# %%
# Merge the reduced embeddings + texts with the sentence ID+cluster number
sample_sentence_clusters = pd.merge(
        sample_sentences_data,
        sentence_embs_sample,
        how='left', on=['job id', 'sentence id'])
sample_sentence_clusters["description"] = sample_sentence_clusters["description"].apply(lambda x: " ".join(x) if isinstance(x, list) else x)
sample_sentence_clusters.head(3)

# %%
print(len(sample_sentence_clusters))
sample_sentence_clusters["reduced_points x"] = sample_sentence_clusters["embedding"].apply(lambda x: x[0])
sample_sentence_clusters["reduced_points y"] = sample_sentence_clusters["embedding"].apply(lambda x: x[1])
sample_sentence_clusters["Cluster number"] = sample_sentence_clusters["Cluster number predicted"]
sample_sentence_clusters.head(3)

# %%
sample_sentence_clusters["Cluster number"].nunique()

# %%
sample_sentence_clusters[sample_sentence_clusters["Cluster number"] >= 0]["Cluster number"].nunique()

# %% [markdown]
# ### 2. Skills data

# %%
skills_data = load_s3_data(
    s3,
    bucket_name,
    f"outputs/skills_extraction/extracted_skills/{file_name_date}_skills_data.json",
)
len(skills_data)

# %%
skill_tfidf = get_level_names(
                sample_sentence_clusters[sample_sentence_clusters["Cluster number"]>=0], "Cluster number", top_n=3
            )

skills_data_names = {}
for k, v in skills_data.items():
    val = v
    val["Skills name"] = skill_tfidf[int(k)]
    skills_data_names[k] = val

skills_data = skills_data_names

# %% [markdown]
# ## High level numbers

# %%
skill_sentences_df = sample_sentence_clusters[sample_sentence_clusters['Cluster number'] >= 0]
not_skill_sentences_df = sample_sentence_clusters[sample_sentence_clusters['Cluster number'] < 0]

# %%
sent_id_skill = [sentence_embs['sentence id'][i] for i,v in enumerate(sentence_embs['Cluster number predicted']) if v>=0]
job_id_skill = [sentence_embs['job id'][i] for i,v in enumerate(sentence_embs['Cluster number predicted']) if v>=0]

sent_id_not_skill = [sentence_embs['sentence id'][i] for i,v in enumerate(sentence_embs['Cluster number predicted']) if v<0]
job_id_not_skill = [sentence_embs['job id'][i] for i,v in enumerate(sentence_embs['Cluster number predicted']) if v<0]

# %%
all_job_ads = 62892486

print(f"There are {len(skills_data)} skills found")
print(f"There are {len(sentence_embs)} sentences with embeddings given")
print(f"{len(sent_id_skill)} of the sentences ({round(len(sent_id_skill)*100/len(sentence_embs),2)}%) were assigned a skill")
print(f"{len(sent_id_not_skill)} of the sentences ({round(len(sent_id_not_skill)*100/len(sentence_embs),2)}%) were skill number -2")

num_jobid_sents = len(set(sentence_embs['job id']))
print(f"There were {num_jobid_sents} unique job adverts ({round(num_jobid_sents*100/all_job_ads,2)}% of all) in all sentences used")

num_jobid_skills = len(set(job_id_skill))
print(f"There were {num_jobid_skills} unique job adverts ({round(num_jobid_skills*100/all_job_ads,2)}% of all, {round(num_jobid_skills*100/num_jobid_sents,2)}% of those with sentences) with a skill given")

num_jobid_noskills = len(set(job_id_not_skill))
print(f"There were {num_jobid_noskills} unique job adverts ({round(num_jobid_noskills*100/all_job_ads,2)}% of all, {round(num_jobid_noskills*100/num_jobid_sents,2)}% of those with sentences) with skill number -2")


# %% [markdown]
# ### 3. An original file before filtering out sentences with low length

# %%
sentence_embeddings_dirs = get_s3_data_paths(
    s3,
    bucket_name,
    f"outputs/skills_extraction/word_embeddings/data/{file_name_date}/",
    file_types=["*.json"],
)
sentence_embeddings_dirs = sentence_embeddings_dirs[0:2]

# %%
original_sentences = {}
for embedding_dir in sentence_embeddings_dirs:
    if "original_sentences.json" in embedding_dir:
        original_sentences.update(load_s3_data(s3, bucket_name, embedding_dir))

# %%
original_sentences["3844972346455279895"]

# %%
mask_seq = "[MASK]"
prop_not_masked_threshold = 0.5

bad_sentences = []
for embedding_dir in tqdm(sentence_embeddings_dirs):
    if "embeddings.json" in embedding_dir:
        sentence_embeddings = load_s3_data(s3, bucket_name, embedding_dir)
        # Only output data for this sentence if it matches various conditions
        print("here")
        for job_id, sent_id, words, embedding in sentence_embeddings:
            original_sentence = original_sentences[str(sent_id)]
            if len(original_sentence) > 250:
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
sentences = random.sample(sample_sentence_clusters["original sentence"].tolist(), 10)

# %%
for sentence in sentences:
    mask_seq = process_sentence_mask(
        sentence,
        nlp,
        bert_vectorizer,
        token_len_threshold=20,
        stopwords=stopwords.words(),
        custom_stopwords=custom_stopwords,
    )
    print("---")
    print(sentence)
    print(mask_seq)

# %% [markdown]
# ## Plot
# You have an odd data point far to the right, make sure this isn't in the sample so our plots are nicer

# %%
# sentence_clusters is really big, and we don't need to plot it all, so use a sample
print(len(sample_sentence_clusters))
sentence_clusters_sample = sample_sentence_clusters.sample(100000, random_state=42).reset_index()
sentence_clusters_sample  = sentence_clusters_sample[sentence_clusters_sample["reduced_points x"]<10]
sentence_clusters_sample.reset_index(inplace=True)

reduced_x = sentence_clusters_sample["reduced_points x"].tolist()
reduced_y = sentence_clusters_sample["reduced_points y"].tolist()

# %%
output_file(
    filename=f"outputs/skills_extraction/figures/{file_name_date.replace('.','_')}/{file_name_date}_reduced_sentences.html"
)

color_palette = viridis

ds_dict = dict(
    x=reduced_x,
    y=reduced_y,
    texts=sentence_clusters_sample["description"].tolist(),
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
    radius=0.02,
    alpha=0.1,
    source=source,
    color="black",
)
p.xaxis.visible = False
p.xgrid.visible = False
p.yaxis.visible = False
p.ygrid.visible = False

save(p)

# %%
output_file(
    filename=f"outputs/skills_extraction/figures/{file_name_date.replace('.','_')}/{file_name_date}_skill_clusters_all_sentences.html"
)

colors_by_labels = sentence_clusters_sample["Cluster number"].astype(str).tolist()
reduced_x = sentence_clusters_sample["reduced_points x"].tolist()
reduced_y = sentence_clusters_sample["reduced_points y"].tolist()
color_palette = viridis

ds_dict = dict(
    x=reduced_x,
    y=reduced_y,
    texts=sentence_clusters_sample["description"].tolist(),
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
output_file(
    filename=f"outputs/skills_extraction/figures/{file_name_date.replace('.','_')}/{file_name_date}_skill_clusters_not_clustered.html"
)

colors_by_labels = sentence_clusters_sample["Cluster number"].astype(str).tolist()
ds_dict_1 = dict(
    x=reduced_x,
    y=reduced_y,
    texts=sentence_clusters_sample["description"].tolist(),
    label=colors_by_labels,
)
source1 = ColumnDataSource(ds_dict_1)

not_clust_ix = sentence_clusters_sample[
    sentence_clusters_sample["Cluster number"] == -2
].index.tolist()
ds_dict_2 = dict(
    x=sentence_clusters_sample.iloc[not_clust_ix]["reduced_points x"].tolist(),
    y=sentence_clusters_sample.iloc[not_clust_ix]["reduced_points y"].tolist(),
    texts=sentence_clusters_sample.iloc[not_clust_ix]["description"].tolist(),
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
sentence_clusters_notnone = sample_sentence_clusters[sample_sentence_clusters["Cluster number"] >= 0]
print(len(sentence_clusters_notnone))

sentence_clusters_notnone_sample = sentence_clusters_notnone.sample(100000, random_state=42)
sentence_clusters_notnone_sample  = sentence_clusters_notnone_sample[sentence_clusters_notnone_sample["reduced_points x"]<10]
sentence_clusters_notnone_sample.reset_index(inplace=True)

# %%
output_file(
    filename=f"outputs/skills_extraction/figures/{file_name_date.replace('.','_')}/{file_name_date}_skill_clusters.html"
)

colors_by_labels = sentence_clusters_notnone_sample["Cluster number"].astype(str).tolist()
reduced_x = sentence_clusters_notnone_sample["reduced_points x"].tolist()
reduced_y = sentence_clusters_notnone_sample["reduced_points y"].tolist()
color_palette = viridis

ds_dict = dict(
    x=reduced_x,
    y=reduced_y,
    texts=sentence_clusters_notnone_sample["description"].tolist(),
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
    radius=0.02,
    alpha=0.1,
    source=source,
    color={"field": "label", "transform": color_mapper},
)
p.xaxis.visible = False
p.xgrid.visible = False
p.yaxis.visible = False
p.ygrid.visible = False

save(p)

# %%
skills_data['0']['Sentences'][0:3]

# %%
output_file(
    filename=f"outputs/skills_extraction/figures/{file_name_date.replace('.','_')}/{file_name_date}_skill_clusters_single.html"
)

reduced_x = [s['Centroid'][0] for s in skills_data.values()]
reduced_y = [s['Centroid'][1] for s in skills_data.values()]
skill_nums = list(skills_data.keys())
sents = [s['Sentences'][0:3] for s in skills_data.values()]

ds_dict = dict(
    x=reduced_x,
    y=reduced_y,
    texts=skill_nums,
    label=sents
)
hover = HoverTool(
    tooltips=[
        ("Skill cluster number", "@texts"),
        ("Sentences", "@label"),
    ]
)
source = ColumnDataSource(ds_dict)

p = figure(
    plot_width=500,
    plot_height=500,
    tools=[hover, ResetTool(), WheelZoomTool(), BoxZoomTool(), SaveTool()],
    title=f"Skills in 2D space",
    toolbar_location="below",
)
p.circle(
    x="x",
    y="y",
    radius=0.05,
    alpha=0.2,
    source=source,
    color='black',
)
p.xaxis.visible = False
p.xgrid.visible = False
p.yaxis.visible = False
p.ygrid.visible = False

save(p)

# %% [markdown]
# ## Examples of skill names post hierarchy naming

# %%
skill_hierarchy_file_dup_name = "outputs/skills_taxonomy/2022.01.21_skills_hierarchy.json"
skill_hierarchy_dup_name = load_s3_data(s3, bucket_name, skill_hierarchy_file_dup_name)

# %%
orig_skill_names = [s['Skill name'] for s in skill_hierarchy_dup_name.values()]
len(set(orig_skill_names))

# %%
from collections import Counter

# %%
print([d[0] for d in Counter(orig_skill_names).most_common(100)])

# %% [markdown]
# # By skill
# - Average embeddings for skills

# %%
# The new skill names are in here (since they are dependent on the hierarchy)
skill_hierarchy_file = "outputs/skills_taxonomy/2022.01.21_skills_hierarchy_named.json"
skill_hierarchy = load_s3_data(s3, bucket_name, skill_hierarchy_file)

# %%
skill_hierarchy['0']

# %%
for k, v in skills_data.items():
    skills_data[k]["Skills name"] = skill_hierarchy[k]['Skill name']

# %%
skills_clusters = (
    sentence_clusters_notnone.groupby("Cluster number")[
        ["reduced_points x", "reduced_points y"]
    ]
    .mean()
    .reset_index()
)
skills_clusters["Skills name"] = skills_clusters["Cluster number"].apply(
    lambda x: skills_data[str(x)]["Skills name"]
)
skills_clusters["Examples"] = skills_clusters["Cluster number"].apply(
    lambda x: skills_data[str(x)]["Sentences"][0:10]
)
skills_clusters.head(2)


# %%
output_file(
    filename=f"outputs/skills_extraction/figures/{file_name_date.replace('.','_')}/{file_name_date}_skill_clusters_average.html"
)

colors_by_labels = skills_clusters["Cluster number"].astype(str).tolist()
reduced_x = skills_clusters["reduced_points x"].tolist()
reduced_y = skills_clusters["reduced_points y"].tolist()
color_palette = viridis

ds_dict = dict(
    x=reduced_x,
    y=reduced_y,
    texts=skills_clusters["Skills name"].tolist(),
    examples=skills_clusters["Examples"].tolist(),
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
p.circle(x="x", y="y", radius=0.04, alpha=0.2, source=source, color="black")
p.xaxis.visible = False
p.xgrid.visible = False
p.yaxis.visible = False
p.ygrid.visible = False

save(p)

# %% [markdown]
# ## Skills by level A/B/C

# %%
sentence_clusters_notnone['Level A'] = sentence_clusters_notnone['Cluster number predicted'].apply(lambda x: skill_hierarchy[str(x)]['Hierarchy level A name'])
sentence_clusters_notnone['Level B'] = sentence_clusters_notnone['Cluster number predicted'].apply(lambda x: skill_hierarchy[str(x)]['Hierarchy level B name'])
sentence_clusters_notnone['Level C'] = sentence_clusters_notnone['Cluster number predicted'].apply(lambda x: skill_hierarchy[str(x)]['Hierarchy level C name'])
sentence_clusters_notnone.head(2)

# %%
small_skills = []
for skill_num, skill_info in skill_hierarchy.items():
    if skill_info['Number of sentences that created skill']<100:
        small_skills.append(skill_num)
small_skills

# %%
skill_num = '6684'# 4404
print(skill_hierarchy[skill_num]['Skill name'])
print(skill_hierarchy[skill_num]['Number of sentences that created skill'])
sentence_clusters_notnone[sentence_clusters_notnone['Cluster number predicted']==int(skill_num)]['original sentence'].tolist()

# %%
empathy_skills = []
for skill_num, skill_info in skill_hierarchy.items():
    if "empathy" in skill_info['Skill name']:
        empathy_skills.append(skill_num)
empathy_skills

# %%
skill_num = '2807'
print(skill_hierarchy[skill_num])
sentence_clusters_notnone[sentence_clusters_notnone['Cluster number predicted']==int(skill_num)]['original sentence'].tolist()[0:10]

# %%
skill_num = '4240'
print(skill_hierarchy[skill_num])
sentence_clusters_notnone[sentence_clusters_notnone['Cluster number predicted']==int(skill_num)]['original sentence'].tolist()[0:10]

# %% [markdown]
# ## Skill examples - don't forget this is from a sample!
#

# %%
skill_id = "-2"
sample_sentence_clusters[sample_sentence_clusters["Cluster number"] == int(skill_id)][
    "original sentence"
].tolist()[100:110]

# %%
data_skills = []
for k, v in skills_data.items():
    if ("excel" in v["Skills name"]) and ("microsoft" in v["Skills name"]):
        if skill_hierarchy[k]['Number of sentences that created skill']>1000:
            print('---')
            print(k)
            print(v["Skills name"])
            print(skill_hierarchy[k]['Number of sentences that created skill'])

# %%
skill_id = "1107"
print(skill_hierarchy[skill_id]['Skill name'])
print(skill_hierarchy[skill_id]['Number of sentences that created skill'])
print(
    sample_sentence_clusters[sample_sentence_clusters["Cluster number"] == int(skill_id)][
        "original sentence"
    ].tolist()[0:10]
)


# %%
data_skills = []
for k, v in skills_data.items():
    if ("python" in v["Skills name"]) and ("python" in v["Skills name"]):
        if skill_hierarchy[k]['Number of sentences that created skill']>1000:
            print('---')
            print(k)
            print(v["Skills name"])
            print(skill_hierarchy[k]['Number of sentences that created skill'])

# %%
skill_id = "1697"
print(skill_hierarchy[skill_id]['Skill name'])
print(skill_hierarchy[skill_id]['Number of sentences that created skill'])
print(
    sample_sentence_clusters[sample_sentence_clusters["Cluster number"] == int(skill_id)][
        "original sentence"
    ].tolist()[0:10]
)

# %%
data_skills = []
for k, v in skills_data.items():
    if ("food" in v["Skills name"]) and ("food" in v["Skills name"]):
        if skill_hierarchy[k]['Number of sentences that created skill']>1000:
            print('---')
            print(k)
            print(v["Skills name"])
            print(skill_hierarchy[k]['Number of sentences that created skill'])

# %%
skill_id = "4326"
print(skill_hierarchy[skill_id]['Skill name'])
print(skill_hierarchy[skill_id]['Number of sentences that created skill'])
print(
    sample_sentence_clusters[sample_sentence_clusters["Cluster number"] == int(skill_id)][
        "original sentence"
    ].tolist()[0:10]
)

# %%
data_skills = []
for k, v in skills_data.items():
    if ("motivated" in v["Skills name"]) and ("motivated" in v["Skills name"]):
        if skill_hierarchy[k]['Number of sentences that created skill']>1000:
            print('---')
            print(k)
            print(v["Skills name"])
            print(skill_hierarchy[k]['Number of sentences that created skill'])

# %%
skill_id = "2696"
print(skill_hierarchy[skill_id]['Skill name'])
print(skill_hierarchy[skill_id]['Number of sentences that created skill'])
print(
    sample_sentence_clusters[sample_sentence_clusters["Cluster number"] == int(skill_id)][
        "original sentence"
    ].tolist()[0:10]
)

# %% [markdown]
# ## Sentence examples per level A

# %%
sentence_clusters_notnone.head(2)

# %%
for level_a_name, level_a_sents in sentence_clusters_notnone.groupby('Level A'):
    print('---')
    print(level_a_name)
    print(level_a_sents["original sentence"].sample(5, random_state=0).tolist())

# %% [markdown]
# ## How many sentences in each skill?
# - Are those with loads of skills junk?

# %%
skill_lengths = {}
for k, v in skills_data.items():
    skill_lengths[k] = len(v["Sentences"])


# %%
print(
    f"The mean number of sentences for each skills is {np.mean(list(skill_lengths.values()))}"
)
print(
    f"The median number of sentences for each skills is {np.median(list(skill_lengths.values()))}"
)
n_max = 200
print(
    f"There are {len([s for s in skill_lengths.values() if s>n_max])} skills with more than {n_max} sentences"
)



# %%
plt.hist(skill_lengths.values(), bins=10);

# %%
len([s for s in skill_lengths.values() if s >=1000])/len(skill_lengths)

# %%
len([s for s in skill_lengths.values() if s <1000])/len(skill_lengths)

# %%
plt.hist(
    [s for s in skill_lengths.values() if s < 1000],
    bins=100,
    color=[255 / 255, 90 / 255, 0],
)
plt.xlabel("Number of sentences in a skill cluster")
plt.ylabel("Number of skills")
filename=f"outputs/skills_extraction/figures/{file_name_date.replace('.','_')}/{file_name_date}_num_sent_in_skill.pdf"
plt.savefig(
    filename, bbox_inches="tight"
)


# %%
len([i for i, s in skill_lengths.items() if s < 100]) / len(skill_lengths)

# %%
[i for i, s in skill_lengths.items() if s > 6000]

# %%
n = "604"
print(skills_data[n]["Skills name"])
skills_data[n]["Sentences"][0:10]

# %% [markdown]
# ## How many skills in each job advert?

# %%
sentence_clusters_notnone_all = sentence_embs[sentence_embs["Cluster number predicted"] >= 0]


# %%
num_skills = list(
    sentence_clusters_notnone_all.groupby("job id")["Cluster number predicted"].nunique()
)

# %%
print(f"There are {sentence_embs['job id'].nunique()} unique job adverts with skills in (inc -2 skill)")
print(f"There are {sentence_embs['sentence id'].nunique()} unique sentences with skills in (inc -2 skill)")

print(f"There are {len(num_skills)} unique job adverts with skills in (not inc -2 skill)")
print(f"There are {sentence_clusters_notnone_all['job id'].nunique()} unique job adverts with skills in (not inc -2 skill)")
print(f"There are {sentence_clusters_notnone_all['sentence id'].nunique()} unique sentences with skills in (not inc -2 skill)")

print(f"The mean number of unique skills per job advert is {np.mean(num_skills)}")
print(f"The median number of unique skills per job advert is {np.median(num_skills)}")
n_max = 30
print(
    f"There are {len([s for s in num_skills if s>n_max])} job adverts with more than {n_max} skills"
)

# %% [markdown]
# ## Number of skills as number sentences increases

# %%
sentence_clusters_notnone_all_shuffle = sentence_clusters_notnone_all.copy()
sentence_clusters_notnone_all_shuffle = sentence_clusters_notnone_all_shuffle.sample(frac=1) 

# %%
sentence_clusters_notnone_all_shuffle.head(2)

# %%
unique_skills = {}
for k in tqdm(np.linspace(0, 400000, num=1000)):
    k = int(k)
    unique_skills[k] = sentence_clusters_notnone_all_shuffle.iloc[0:k]['Cluster number predicted'].nunique()

# %%
# This is vocab size as number of sentences increases, it's from older data, but I think it's still relevant
num_sentences_and_vocab_size = load_s3_data(
    s3,
    BUCKET_NAME,
    "outputs/skills_extraction/extracted_skills_sample_50k/2021.08.31_num_sentences_and_vocab_size.json",
)

# %%
x_vocab = [v[0] for v in num_sentences_and_vocab_size]
y_vocab = [v[1] for v in num_sentences_and_vocab_size]

x_skills = list(unique_skills.keys())
y_skills = list(unique_skills.values())

# %%
fig, axs = plt.subplots(1,2, figsize=(12,3))

axs[0].plot(x_vocab, y_vocab, color='black');
axs[0].axvline(300000, color="orange", ls='--')
axs[0].set_xlabel('Number of sentences')
axs[0].set_ylabel('Number of unique words in vocab')

axs[1].plot(x_skills, y_skills, color='black');
axs[1].set_xlabel('Number of sentences')
axs[1].set_ylabel('Number of unique skills')

plt.tight_layout()
plt.savefig('outputs/skills_extraction/figures/2022_01_14/num_sent_num_skills_vocab_size.pdf',bbox_inches='tight')

# %%
