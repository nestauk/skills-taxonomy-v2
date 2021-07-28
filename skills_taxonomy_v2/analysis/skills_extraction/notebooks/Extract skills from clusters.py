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
# ## Get skills
#
# After reading, cleaning and reducing the skill sentences to 2D, in this notebook we can read in and start to experiment with getting out the skills.
#
# This is all about having a new skills list with: `skill number, skill description, skill name`

# %%
# cd ../../../..

# %%
import json
from collections import Counter

import boto3
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


# %%
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
)
from bokeh.io import output_file, reset_output, save, export_png, show
from bokeh.resources import CDN
from bokeh.embed import file_html
from bokeh.palettes import Plasma, magma, cividis, inferno, plasma, viridis, Spectral6
from bokeh.transform import linear_cmap

bpl.output_notebook()

# %%
from skills_taxonomy_v2.getters.s3_data import get_s3_data_paths, load_s3_data

# %%
from skills_taxonomy_v2.pipeline.skills_extraction.cleaning_sentences import (
    get_common_tuples,
    build_ngrams,
)

# %%
clustered_data = pd.read_csv("outputs/skills_extraction/data/clustered_data.csv")

# %%
clustered_data.head(2)

# %% [markdown]
# ## Load original sentence

# %%
bucket_name = "skills-taxonomy-v2"
s3 = boto3.resource("s3")
sentence_id_dir = "outputs/skills_extraction/word_embeddings/data/"

# %%
sentence_id_dirs = get_s3_data_paths(
    s3, bucket_name, sentence_id_dir, file_types=["*.json"]
)
sentence_id_dirs = [
    file_dir for file_dir in sentence_id_dirs if "original_sentences.json" in file_dir
]
len(sentence_id_dirs)

# %%
original_sentences = {}

for sentence_id_dir in sentence_id_dirs:
    original_sentences.update(load_s3_data(s3, bucket_name, sentence_id_dir))

# %%
len(original_sentences)

# %%
original_sentences["2664663441956474979"]

# %% [markdown]
# ## Plot

# %%
colour_by_list = clustered_data["Cluster number"].tolist()

# %%
ds_dict = dict(
    x=clustered_data["reduced_points x"].tolist(),
    y=clustered_data["reduced_points y"].tolist(),
    texts=clustered_data["description"].tolist(),
    cols=colour_by_list,
)
mapper = linear_cmap(
    field_name="cols",
    palette=Spectral6,
    low=min(colour_by_list),
    high=max(colour_by_list),
)
hover = HoverTool(tooltips=[("node", "@texts")])  # ,("colour by", "@cols"),
source = ColumnDataSource(ds_dict)
p = figure(
    plot_width=500,
    plot_height=500,
    tools=[hover, WheelZoomTool(), BoxZoomTool(), SaveTool()],
    title=f"Skills",
    toolbar_location="below",
)
p.circle(x="x", y="y", radius=0.05, alpha=0.5, source=source, color=mapper)
show(p)

# %% [markdown]
# ## Normalise the texts for getting descriptions from
# - lemmatize
# - lower case
# - remove duplicates
# - n-grams
#
# For each cluster

# %%
import re
import nltk
from nltk.util import ngrams  # function for making ngrams
from nltk.stem import WordNetLemmatizer


# %%
def replace_ngrams(sentence, ngram_words):
    for word_list in ngram_words:
        sentence = sentence.replace(" ".join(word_list), "-".join(word_list))
    return sentence


# %%
# Init the Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()

# %%
# How many times a n-gram has to occur in order to be converted
num_times_ngrams_thresh = 3

# %%
clustered_data["description"] = clustered_data["description"].apply(
    lambda x: re.sub("\s+", " ", x)
)

# %%
cluster_clean_details = {}
for cluster_num, cluster_group in clustered_data.groupby("Cluster number"):
    cluster_docs = cluster_group["description"].tolist()
    cluster_docs_cleaned = []
    for doc in cluster_docs:
        # Remove capitals, but not when it's an acronym
        acronyms = re.findall("[A-Z]{2,}", doc)
        # Lemmatize
        lemmatized_output = [
            lemmatizer.lemmatize(w)
            if w in acronyms
            else lemmatizer.lemmatize(w.lower())
            for w in doc.split(" ")
        ]
        cluster_docs_cleaned.append(" ".join(lemmatized_output).strip())
    # Remove duplicates
    cluster_docs_cleaned = list(set(cluster_docs_cleaned))

    # Find the ngrams for this cluster
    all_cluster_docs = " ".join(cluster_docs_cleaned).split(" ")

    esBigrams = ngrams(all_cluster_docs, 3)
    ngram_words = [
        words
        for words, count in Counter(esBigrams).most_common()
        if count >= num_times_ngrams_thresh
    ]

    cluster_docs_clean = [
        replace_ngrams(sentence, ngram_words) for sentence in cluster_docs_cleaned
    ]

    cluster_clean_details[cluster_num] = {
        "docs": cluster_docs_clean,
        "n_grams": ngram_words,
    }


# %% [markdown]
# ## Get skills from the clusters
# - top tfidf words to name
# - cleaned sentences
# - nearest original sentence to the cluster centre

# %%
def get_top_tf_idf_words(clusters_vect, feature_names, top_n=2):
    """
    From https://stackoverflow.com/questions/34232190/scikit-learn-tfidfvectorizer-how-to-get-top-n-terms-with-highest-tf-idf-score
    """
    sorted_nzs = np.argsort(clusters_vect.data)[: -(top_n + 1) : -1]
    return feature_names[clusters_vect.indices[sorted_nzs]]


# %%
# Get top words for each cluster
num_top_words = 5

cluster_texts = {
    cluster_num: ". ".join(c["docs"])
    for cluster_num, c in cluster_clean_details.items()
}

# Have a token pattern that keeps in words with dashes in betwee
cluster_vectorizer = TfidfVectorizer(
    stop_words="english", token_pattern=r"(?u)\b\w[\w-]*\w\b|\b\w+\b"
)
clusters_vects = cluster_vectorizer.fit_transform(list(cluster_texts.values()))

# %%
cluster_main_sentence = {}
for cluster_num in tqdm(clustered_data["Cluster number"].unique()):
    # There may be the same sentence repeated
    v_reps = clustered_data[clustered_data["Cluster number"] == cluster_num]
    v = v_reps.drop_duplicates(["sentence id"])
    cluster_text = v["sentence id"].apply(lambda x: original_sentences[str(x)]).tolist()
    cluster_coords = v[["reduced_points x", "reduced_points y"]].values

    # Get similarities to centre
    similarities = cosine_similarity(
        np.mean(cluster_coords, axis=0).reshape(1, -1), cluster_coords
    )

    # Join the 5 closest sentences to the centroid
    cluster_main_sentence[cluster_num] = ". ".join(
        [cluster_text[i] for i in similarities.argsort()[0][::-1].tolist()[0:5]]
    )

# %%
# Top n words for each cluster + other info
feature_names = np.array(cluster_vectorizer.get_feature_names())

cluster_info = {}
for (cluster_num, text), clusters_vect in zip(cluster_texts.items(), clusters_vects):
    cluster_info[cluster_num] = {
        "Skill name": " ".join(
            list(get_top_tf_idf_words(clusters_vect, feature_names, num_top_words))
        ),
        "Description": cluster_main_sentence[cluster_num],
        "text": text,
    }

# %%
cluster_info[100]

# %%
# Save skills
pd.DataFrame(cluster_info).T.to_csv(
    "outputs/skills_extraction/data/clustered_data_skillnames.csv"
)

# %% [markdown]
# ## Summarizing the clusters using full text experiments
#
# - keybert
# - transformers summarization
#
# #### Keywords
#
# ```
# %pip install keybert
#
# from keybert import KeyBERT
# kw_model = KeyBERT()
#
# cluster_num = 1
# all_cluster_text = ". ".join([original_sentences[str(c)] for c in clustered_data[clustered_data['Cluster number']==cluster_num]['sentence id'].tolist()])
#
# ```
# 'Server Sharepoint Services About the Role You will be working as part of a team to. To be a Financial Advisor to Members and Officers at all levels of seniority across the Council. As a Senior Assistant Merchandiser you will play a vital role within our Merchandising teams. We have a wide range of clients in need of Class NUMBER drivers to do a multitude of different jobs. You will provide leadership to the Infrastructure 4 and Service Desk NUMBER teams on a NUMBER basis. The role is within the Global Markets GM Costs team which is part of GM Finance Central. How would you like to work for a world leading organisation in the South West. Access to a wide range of vacancies as we are a preferred supplier to numerous NHS Trusts. We are looking for someone to take on the role of a Procurement and Contracts officer. We want the best people on our team and the best way to do that is to look after them. We are looking for a Practical Land Management Assistant to join our team on a sessional basis. The role itself is within the Customer Marketing team working as a CRM Manager'
#
# ```
# kw_model.extract_keywords(all_cluster_text)
# ```
# [('server', 0.5068),
#  ('clients', 0.4956),
#  ('sharepoint', 0.44),
#  ('management', 0.4383),
#  ('services', 0.4336)]
#
#
# #### Text summary
# Takes a long time!
# ```
# from transformers import pipeline
#
# # Initialize the HuggingFace summarization pipeline
# summarizer = pipeline("summarization")
# ```
#
# 'Server Sharepoint Services About the Role You will be working as part of a team to. To be a Financial Advisor to Members and Officers at all levels of seniority across the Council. As a Senior Assistant Merchandiser you will play a vital role within our Merchandising teams. We have a wide range of clients in need of Class NUMBER drivers to do a multitude of different jobs. You will provide leadership to the Infrastructure 4 and Service Desk NUMBER teams on a NUMBER basis. The role is within the Global Markets GM Costs team which is part of GM Finance Central. How would you like to work for a world leading organisation in the South West. Access to a wide range of vacancies as we are a preferred supplier to numerous NHS Trusts. We are looking for someone to take on the role of a Procurement and Contracts officer. We want the best people on our team and the best way to do that is to look after them. We are looking for a Practical Land Management Assistant to join our team on a sessional basis. The role itself is within the Customer Marketing team working as a CRM Manager'
#
# ```
# summarized = summarizer(all_cluster_text, min_length=10, max_length=50)
#
# # Print summarized text
# print(summarized)
# ```
# [{'summary_text': ' You will be working as part of a team to be a Financial Advisor to Members and Officers at all levels of seniority across the Council . The role is within the Global Markets GM Costs team which is part of GM Finance Central .'}]

# %%
