# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Plot sentence embeddings to view skills
# We calculated these embeddings in the sentence classifier step

# %%
# cd ../../../..

# %%
import json
import pickle

import numpy as np

# %%
import umap.umap_ as umap

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
)
from bokeh.io import output_file, reset_output, save, export_png, show
from bokeh.resources import CDN
from bokeh.embed import file_html

bpl.output_notebook()

# %%
config_name = "2021.07.09.small"
with open(
    f"outputs/sentence_classifier/data/skill_sentences/{config_name}_jobsnew1_embeddings.pkl",
    "rb",
) as file:
    sentences_vec = pickle.load(file)

with open(
    f"outputs/sentence_classifier/data/skill_sentences/{config_name}_jobsnew1_predictions.pkl",
    "rb",
) as file:
    sentences_pred = pickle.load(file)

with open(
    f"outputs/sentence_classifier/data/skill_sentences/{config_name}_jobsnew1_sentences.pkl",
    "rb",
) as file:
    sentences = pickle.load(file)


# %%
# Filter out the non-skill sentences
print(len(sentences))
sentences_vec = sentences_vec[sentences_pred.astype(bool)]
sentences = [sentences[i] for i, p in enumerate(sentences_pred.astype(bool)) if p == 1]
print(sentences_vec.shape)
print(len(sentences))

# %%
deduplicated_sentences[0]

# %%
deduplicated_sentences_vec[0][0:10]

# %%
sentences[0]

# %%
deduplicated_dict = {}
for sentence, vector in zip(sentences, sentences_vec):
    if sentence not in deduplicated_dict:
        deduplicated_dict[sentence] = vector
deduplicated_sentences_vec = list(deduplicated_dict.values())
deduplicated_sentences = list(deduplicated_dict.keys())


# %%
# Deduplicate sentences
print(len(deduplicated_sentences_vec))
print(len(deduplicated_sentences))

# %% [markdown]
# ## Reduce to 2D

# %%
reducer_class = umap.UMAP(n_neighbors=50, min_dist=0.2, random_state=42)
reduced_points_umap = reducer_class.fit_transform(deduplicated_sentences_vec)

reduced_points = reduced_points_umap
reduced_x = reduced_points[:, 0]
reduced_y = reduced_points[:, 1]


# %%
ds_dict = dict(x=reduced_x, y=reduced_y, texts=deduplicated_sentences)
hover = HoverTool(tooltips=[("node", "@texts"),])
source = ColumnDataSource(ds_dict)
p = figure(
    plot_width=500,
    plot_height=500,
    tools=[hover, WheelZoomTool(), BoxZoomTool(), SaveTool()],
    title=f"Skills",
    toolbar_location="below",
)
p.circle(
    x="x", y="y", radius=0.1, alpha=0.5, source=source,
)
show(p)

# %%
