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
# - There are 6783 skills found
# - There are 4097008 sentences with embeddings given
# - There are 4097008 sentences with clusters given
# - 1465639 of the sentences (35.77%) were assigned a skill
# - 2631369 of the sentences (64.23%) were skill number -2
# - There were 1012869 unique job adverts (1.61% of all) in all sentences used
# - There were 510039 unique job adverts (0.81% of all, 50.36% of those with sentences) with a - skill given
# - There were 916270 unique job adverts (1.46% of all, 90.46% of those with sentences) with skill number -2
#
# 1. How many skills
#
# - There are 4097008 sentences were predicted on
# - 2631369 sentences were too long (over 100 characters)
# - There are 1465639 sentences that went into creating skills
# - There are 6784 skills found (inc the -2 cluster)
#
# 2. How many not in skills
#
# - 0.6422660146135912 proportion of sentences arent in a cluster
#
# 3. Plot of skills
# 4. Examples of skills
# 5. Number of sentences in skills
#
# - The mean number of sentences for each skills is 216.07533539731682
# - The median number of sentences for each skills is 148.0
# - There are 1995 skills with more than 200 sentences
#
# 6. Number of skills in job adverts
#
# - There are 510039 unique job adverts with skills in
# - The mean number of unique skills per job advert is 2.8537053048884498
# - The median number of unique skills per job advert is 2.0
# - There are 168 job adverts with more than 30 skills
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
file_name_date = "2021.11.05"

# %% [markdown]
# ### 1. The sentences clustered into skills

# %%
bucket_name = "skills-taxonomy-v2"
s3 = boto3.resource("s3")

# %%
reduced_embeddings_dir="outputs/skills_extraction/reduced_embeddings/"
clustered_sentences_path=f"outputs/skills_extraction/extracted_skills/{file_name_date}_sentences_skills_data.json"

# %%
# The sentences ID + cluster num
sentence_embs = load_s3_data(s3, bucket_name, clustered_sentences_path)
sentence_embs = pd.DataFrame(sentence_embs)

# %%
# Get the reduced embeddings + sentence texts and the sentence IDs

reduced_embeddings_paths = get_s3_data_paths(
    s3,
    bucket_name,
    reduced_embeddings_dir,
    file_types=["*sentences_data_*.json"]
    )

sentences_data = pd.DataFrame()
for reduced_embeddings_path in tqdm(reduced_embeddings_paths):
    sentences_data_i = load_s3_data(
        s3, bucket_name,
        reduced_embeddings_path
    )
    sentences_data = pd.concat([sentences_data, pd.DataFrame(sentences_data_i)])
sentences_data.reset_index(drop=True, inplace=True)

# %%
# Merge the reduced embeddings + texts with the sentence ID+cluster number
sentence_clusters = pd.merge(
        sentences_data,
        sentence_embs,
        how='left', on=['job id', 'sentence id'])
sentence_clusters["description"] = sentence_clusters["description"].apply(lambda x: " ".join(x) if isinstance(x, list) else x)
sentence_clusters.head(3)

# %%
print(len(sentence_clusters))
sentence_clusters["reduced_points x"] = sentence_clusters["embedding"].apply(lambda x: x[0])
sentence_clusters["reduced_points y"] = sentence_clusters["embedding"].apply(lambda x: x[1])
sentence_clusters["Cluster number"] = sentence_clusters["Cluster number predicted"]
sentence_clusters.head(3)

# %%
sentence_clusters["Cluster number"].nunique()

# %%
sentence_clusters[sentence_clusters["Cluster number"] >= 0]["Cluster number"].nunique()

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
                sentence_clusters[sentence_clusters["Cluster number"]>=0], "Cluster number", top_n=3
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
skill_sentences_df = sentence_clusters[sentence_clusters['Cluster number'] >= 0]
not_skill_sentences_df = sentence_clusters[sentence_clusters['Cluster number'] < 0]

# %%
all_job_ads = 62892486

print(f"There are {len(skills_data)} skills found")
print(f"There are {len(sentence_embs)} sentences with embeddings given")
print(f"There are {len(sentence_clusters)} sentences with clusters given")
print(f"{len(skill_sentences_df)} of the sentences ({round(len(skill_sentences_df)*100/len(sentence_clusters),2)}%) were assigned a skill")
print(f"{len(not_skill_sentences_df)} of the sentences ({round(len(not_skill_sentences_df)*100/len(sentence_clusters),2)}%) were skill number -2")

num_jobid_sents = sentence_clusters['job id'].nunique()
print(f"There were {num_jobid_sents} unique job adverts ({round(num_jobid_sents*100/all_job_ads,2)}% of all) in all sentences used")

num_jobid_skills = skill_sentences_df['job id'].nunique()
print(f"There were {num_jobid_skills} unique job adverts ({round(num_jobid_skills*100/all_job_ads,2)}% of all, {round(num_jobid_skills*100/num_jobid_sents,2)}% of those with sentences) with a skill given")

num_jobid_noskills = not_skill_sentences_df['job id'].nunique()
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

# %%
sentence_embeddings_dirs = sentence_embeddings_dirs[0:2]

# %%
original_sentences = {}
for embedding_dir in sentence_embeddings_dirs:
    if "original_sentences.json" in embedding_dir:
        original_sentences.update(load_s3_data(s3, bucket_name, embedding_dir))

# %%
original_sentences["-6242736777306751508"]

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
sentences = random.sample(sentence_clusters["original sentence"].tolist(), 10)

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
# ### Analysis
# 1. How many skills
# 2. How many not in skills
# 3. Siloutte score for clustering
#
# You assigned "Cluster number" == to "Cluster number predicted" earlier up

# %%
print(f"There are {len(sentence_clusters)} sentences were predicted on")
print(f"{len(sentence_clusters[sentence_clusters['Cluster number']<0])} sentences were too long (over 100 characters)")
print(f"There are {len(sentence_clusters[sentence_clusters['Cluster number']>=0])} sentences that went into creating skills")
print(f'There are {sentence_clusters["Cluster number"].nunique()} skills found (inc the -2 cluster)')
print(
    f'{sum(sentence_clusters["Cluster number"]==-2)/len(sentence_clusters)} proportion of sentences arent in a cluster'
)

# %% [markdown]
# ## Plot

# %%
sentence_clusters_notnone = sentence_clusters[sentence_clusters["Cluster number"] >= 0]
len(sentence_clusters_notnone)

# %%
# sentence_clusters is really big, and we don't need to plot it all, so use a sample
print(len(sentence_clusters))
sentence_clusters_sample = sentence_clusters.sample(100000, random_state=42).reset_index()

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
    radius=0.01,
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
sentence_clusters_notnone_sample = sentence_clusters_notnone.sample(100000, random_state=42)

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
    radius=0.005,
    alpha=0.1,
    source=source,
    color={"field": "label", "transform": color_mapper},
)
p.xaxis.visible = False
p.xgrid.visible = False
p.yaxis.visible = False
p.ygrid.visible = False

save(p)

# %% [markdown]
# ### By skill

# %%
# The new skill names are in here (since they are dependent on the hierarchy)
skill_hierarchy_file = "outputs/skills_taxonomy/2021.11.30_skills_hierarchy_named.json"
skill_hierarchy = load_s3_data(s3, bucket_name, skill_hierarchy_file)

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

# %%
output_file(
    filename=f"outputs/skills_extraction/figures/{file_name_date.replace('.','_')}/{file_name_date}_skill_clusters_average_labelled.html"
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

p.circle(x="x", y="y", radius=0.04, alpha=0.08, source=source, color="grey")

skills_clusters_sample_n = [
    13189,
    9866,
    1525,
    5077,
    6418,
    192,
    4235,
    4566,
    13536,
    5183,
    1556,
    7613,
    2744,
    1768,
    18415,
    12386,
    6760,
    2970,
    9588,
    6213,
    11451,
    13347,
    15905,
    8898,
    1674,
    3876,
    9296,
    18040,
    8253,
    16692,
    8584,
    8556,
    8888,
    8497,
    242,
    11136,
]
skills_clusters_sample = skills_clusters.copy()[
    skills_clusters["Cluster number"].isin(skills_clusters_sample_n)
]
ds_dict_text = dict(
    x=skills_clusters_sample["reduced_points x"].tolist(),
    y=skills_clusters_sample["reduced_points y"].tolist(),
    texts=skills_clusters_sample["Skills name"].tolist(),
    label=skills_clusters_sample["Cluster number"].tolist(),
)
source_text = ColumnDataSource(ds_dict_text)

glyph = Text(
    x="x", y="y", text="texts", angle=0, text_color="black", text_font_size="7pt"
)
p.add_glyph(source_text, glyph)


p.xaxis.visible = False
p.xgrid.visible = False
p.yaxis.visible = False
p.ygrid.visible = False

save(p)

# %% [markdown]
# ## Skill examples

# %%
skill_id = "-2"
sentence_clusters[sentence_clusters["Cluster number"] == int(skill_id)][
    "original sentence"
].tolist()[100:110]

# %%
data_skills = []
for k, v in skills_data.items():
    if ("excel" in v["Skills name"]) and ("microsoft" in v["Skills name"]):
        data_skills.append(k)
data_skills[0:10]

# %%
skill_id = "136"
# print(skills_data[skill_id])
print(
    sentence_clusters[sentence_clusters["Cluster number"] == int(skill_id)][
        "original sentence"
    ].tolist()[0:10]
)


# %%
data_skills = []
for k, v in skills_data.items():
    if ("unit" in v["Skills name"]):
        data_skills.append(k)
data_skills[0:10]

# %%
skill_id = "922"
# print(skills_data[skill_id])
print(
    sentence_clusters[sentence_clusters["Cluster number"] == int(skill_id)][
        "original sentence"
    ].tolist()[0:3]
)

# %%
data_skills = []
for k, v in skills_data.items():
    if ("python" in v["Skills name"]) and ("progra" in v["Skills name"]):
        data_skills.append(k)
data_skills

# %%
skill_id = "1590"
# print(skills_data[skill_id])
print(
    sentence_clusters[sentence_clusters["Cluster number"] == int(skill_id)][
        "original sentence"
    ].tolist()[0:10]
)

# %%
data_skills = []
for k, v in skills_data.items():
    if ("teaching" in v["Skills name"]):
        data_skills.append(k)
data_skills[0:10]

# %%
skill_id = "23"
print(skills_data[skill_id]["Skills name"])
print(
    sentence_clusters[sentence_clusters["Cluster number"] == int(skill_id)][
        "original sentence"
    ].tolist()[0:10]
)

# %%
data_skills = []
for k, v in skills_data.items():
    if ("food" in v["Skills name"]):
        data_skills.append(k)
data_skills[0:10]

# %%
skill_id = "288"
print(skills_data[skill_id]["Skills name"])
print(
    sentence_clusters[sentence_clusters["Cluster number"] == int(skill_id)][
        "original sentence"
    ].tolist()[0:10]
)

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
plt.hist(
    [s for s in skill_lengths.values() if s < 100],
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
len([i for i, s in skill_lengths.items() if s < 20]) / len(skill_lengths)

# %%
[i for i, s in skill_lengths.items() if s > 3000]

# %%
n = "251"
print(skills_data[n]["Skills name"])
skills_data[n]["Sentences"][0:10]

# %% [markdown]
# ## How many skills in each job advert?

# %%
num_skills = list(
    sentence_clusters_notnone.groupby("job id")["Cluster number"].nunique()
)

# %%
print(f"There are {len(num_skills)} unique job adverts with skills in")
print(f"The mean number of unique skills per job advert is {np.mean(num_skills)}")
print(f"The median number of unique skills per job advert is {np.median(num_skills)}")
n_max = 30
print(
    f"There are {len([s for s in num_skills if s>n_max])} job adverts with more than {n_max} skills"
)

# %%
