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
# After finding the taxonomy in `build_taxonomy.py`, this notebook provides some visualisation and analysis of it.

# %%
# cd ../../../..

# %%
from skills_taxonomy_v2.getters.s3_data import load_s3_data

# %%
from collections import Counter, defaultdict
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import boto3
from ipywidgets import interact
import bokeh.plotting as bpl
from bokeh.plotting import ColumnDataSource, figure, output_file, show, from_networkx, gridplot
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
    Title
)

from bokeh.io import output_file, reset_output, save, export_png, show, push_notebook
from bokeh.resources import CDN
from bokeh.embed import file_html
from bokeh.palettes import Plasma, magma, cividis, inferno, plasma, viridis, Spectral6, Turbo256, Spectral, Spectral4, inferno 
from bokeh.transform import linear_cmap

bpl.output_notebook()

# %%
bucket_name = "skills-taxonomy-v2"
s3 = boto3.resource("s3")

# %% [markdown]
# ## Load the hierarchy and original sentence information

# %%
hier_structure_file = 'outputs/skills_hierarchy/2021.09.06_hierarchy_structure.json'
hier_structure = load_s3_data(s3, bucket_name, hier_structure_file)

# %%
skill_hierarchy_file = 'outputs/skills_hierarchy/2021.09.06_skills_hierarchy.json'
skill_hierarchy = load_s3_data(s3, bucket_name, skill_hierarchy_file)

# %%
sentence_data = load_s3_data(s3, bucket_name, 'outputs/skills_extraction/extracted_skills/2021.08.31_sentences_data.json')


# %%
skills_data = load_s3_data(s3, bucket_name, 'outputs/skills_extraction/extracted_skills/2021.08.31_skills_data.json')


# %% [markdown]
# ### Get the manual names

# %%
with open('skills_taxonomy_v2/utils/2021.09.06_level_a_rename_dict.json', 'r') as f:
    level_a_rename_dict = json.load(f)

# %% [markdown]
# #### Join some of the hierarchy data to the sentence data for ease of visualisations

# %%
sentence_data = pd.DataFrame(sentence_data)
sentence_data = sentence_data[sentence_data['Cluster number']!=-1]

# %%
sentence_data['Hierarchy level A'] = sentence_data["Cluster number"].astype(str).apply(
    lambda x: skill_hierarchy[x]['Hierarchy level A'])
sentence_data['Hierarchy level A name'] = sentence_data["Hierarchy level A"].astype(str).apply(
    lambda x: level_a_rename_dict[x])
sentence_data['Hierarchy level B'] = sentence_data["Cluster number"].astype(str).apply(
    lambda x: skill_hierarchy[x]['Hierarchy level B'])
sentence_data['Hierarchy level C'] = sentence_data["Cluster number"].astype(str).apply(
    lambda x: skill_hierarchy[x]['Hierarchy level C'])
# Hierarchy D isn't unique to where it is in the hierarchy, so needs to be merged with level C
sentence_data['Hierarchy level D'] = sentence_data["Cluster number"].astype(str).apply(
    lambda x: str(skill_hierarchy[x]['Hierarchy level C'])+'-'+str(skill_hierarchy[x]['Hierarchy level D']))
sentence_data['Hierarchy ID'] = sentence_data["Cluster number"].astype(str).apply(
    lambda x: skill_hierarchy[x]['Hierarchy ID'])



# %%
sentence_data['Hierarchy ID name'] = sentence_data["Cluster number"].astype(str).apply(
    lambda x: "/".join([
        str(skill_hierarchy[x]['Hierarchy level A name']),
        str(skill_hierarchy[x]['Hierarchy level B name']),
        str(skill_hierarchy[x]['Hierarchy level C name'])
    ]))

# %% [markdown]
# ## Size of levels and how many sentences in each level?

# %%
print(sentence_data['Hierarchy level A'].nunique())
print(sentence_data['Hierarchy level B'].nunique())
print(sentence_data['Hierarchy level C'].nunique())
print(sentence_data['Hierarchy level D'].nunique())
print(sentence_data['Cluster number'].nunique())

# %%
sentence_data.groupby(['Hierarchy level A'])['sentence id'].nunique()

# %%
sentence_data.groupby(['Hierarchy level A'])['Hierarchy level B'].nunique()

# %%
sentence_data.groupby(['Hierarchy level A', 'Hierarchy level B', 'Hierarchy level C'])['sentence id'].nunique()

# %% [markdown]
# ## Taxonomy examples

# %%
[k for k,v in skill_hierarchy.items() if 'machine learn' in v['Skill name']]

# %%
skill_hierarchy['1228']

# %%
[(kk,vv['Name']) for kk,vv in hier_structure['5']['Level B']['47']['Level C']['146']['Level D'].items()]

# %%
hier_structure['5']['Level B']['47']['Level C']['146']['Level D']['3']


# %% [markdown]
# ## Plot sentences coloured by hierarchy level

# %%
def save_plot_sentence_col_hier(sentence_data, col_by_level, filename):
    output_file(filename=filename)

    colors_by_labels = sentence_data[f'Hierarchy level {col_by_level}'].astype(str)
    reduced_x = sentence_data['reduced_points x'].tolist()
    reduced_y = sentence_data['reduced_points y'].tolist()
    color_palette = viridis

    ds_dict = dict(
        x=reduced_x,
        y=reduced_y,
        texts=sentence_data["description"].tolist(),
        hier_info=sentence_data["Hierarchy ID"],
        label=colors_by_labels,
    )
    hover = HoverTool(
        tooltips=[
            ("Sentence", "@texts"),
            (f"Hierarchy level {col_by_level}", "@label"),
            ("Hierarchy information", "@hier_info"),
        ]
    )
    source = ColumnDataSource(ds_dict)

    color_mapper = LinearColorMapper(palette="Turbo256", low=0, high=len(list(set(colors_by_labels))) + 1)

    p = figure(
        plot_width=500,
        plot_height=500,
        tools=[hover, ResetTool(), WheelZoomTool(), BoxZoomTool(), SaveTool()],
        title=f"Sentences coloured by hierarchy {col_by_level}",
        toolbar_location="below",
    )
    p.circle(
        x="x",
        y="y",
        radius=0.01,
        alpha=0.3,
        source=source,
        color={"field": "label", "transform": color_mapper},
    )
    
    p.xaxis.visible = False
    p.xgrid.visible = False
    p.yaxis.visible = False
    p.ygrid.visible = False

    save(p)


# %%
col_by_level = 'A'
filename = 'outputs/skills_taxonomy/figures/2021.09.06/sentences_col_hier_A.html'
save_plot_sentence_col_hier(sentence_data, col_by_level, filename)

# %%
col_by_level = 'B'
filename = 'outputs/skills_taxonomy/figures/2021.09.06/sentences_col_hier_B.html'
save_plot_sentence_col_hier(sentence_data, col_by_level, filename)

# %%
col_by_level = 'C'
filename = 'outputs/skills_taxonomy/figures/2021.09.06/sentences_col_hier_C.html'
save_plot_sentence_col_hier(sentence_data, col_by_level, filename)

# %% [markdown]
# ## Plot average embeddings for each level coloured by next level

# %%
level_b_names = {}
level_c_names = {}
for _, level_a_info in hier_structure.items():
    for level_b_num, level_b_info in level_a_info['Level B'].items():
        level_b_names[level_b_num] = level_b_info['Name']
        for level_c_num, level_c_info in level_b_info['Level C'].items():
            level_c_names[level_c_num] = level_c_info['Name']

# %%
a2o_dict = sentence_data.groupby('Hierarchy level A')['Hierarchy level A'].unique().apply(lambda x: 0) # This is just for viz reasons
b2a_dict = sentence_data.groupby('Hierarchy level B')['Hierarchy level A'].unique().apply(lambda x: x[0])
c2b_dict = sentence_data.groupby('Hierarchy level C')['Hierarchy level B'].unique().apply(lambda x: x[0])


# %%
def get_level_average_emb(hier_structure, sentence_data, level_name, a2o_dict=a2o_dict, b2a_dict=b2a_dict, c2b_dict=c2b_dict):
    level_embs = sentence_data.groupby(
        f'Hierarchy level {level_name}')[['reduced_points x', 'reduced_points y']].mean().to_dict(orient='index')
    for k in level_embs.keys():
        if level_name=='A':
            level_embs[k]['Name'] = hier_structure[str(k)]['Name']
            level_embs[k]['Next hierarchy'] = a2o_dict[k]
        elif level_name=='B':
            level_embs[k]['Name'] = level_b_names[str(k)]
            level_embs[k]['Next hierarchy'] = b2a_dict[k]
        elif level_name=='C':
            level_embs[k]['Name'] = level_c_names[str(k)]
            level_embs[k]['Next hierarchy'] = c2b_dict[k]
            
    level_embs = pd.DataFrame(level_embs).T
    level_embs[f'Level {level_name} number'] = level_embs.index
    return level_embs


# %%
level_a_embs = get_level_average_emb(hier_structure, sentence_data, 'A')
level_b_embs = get_level_average_emb(hier_structure, sentence_data, 'B')
level_c_embs = get_level_average_emb(hier_structure, sentence_data, 'C')


# %%
def plot_average_levels(sentence_data, level_embs_df, col_by_level, filename, title='', sub_title='',pnt_rad=0.4):
    output_file(filename=filename)

    # Data for the foreground (hierarchy level average embeddings)
    colors_by_labels = level_embs_df[f'Next hierarchy'].tolist()
    ds_dict_fore = dict(
        x=level_embs_df['reduced_points x'].tolist(),
        y=level_embs_df['reduced_points y'].tolist(),
        texts=level_embs_df["Name"].tolist(),
        label=colors_by_labels,
    )
    hover = HoverTool(
        tooltips=[
            ("Name", "@texts"),
        ]
    )
    source_fore = ColumnDataSource(ds_dict_fore)
    color_mapper_fore = LinearColorMapper(palette="Turbo256", low=0, high=len(list(set(colors_by_labels))) + 1)

    # Data for the background (sentence embeddings)
    colors_by_labels_back = level_embs_df[f'Level {col_by_level} number'].tolist()
    source_back = ColumnDataSource(dict(
        x=sentence_data['reduced_points x'].tolist(),
        y=sentence_data['reduced_points y'].tolist(),
        label=sentence_data[f'Hierarchy level {col_by_level}'].tolist(),
        texts=sentence_data["description"].tolist()
    ))
    color_mapper_back = LinearColorMapper(palette="Turbo256", low=0, high=len(list(set(colors_by_labels_back))) + 1)
    
    p = figure(
        plot_width=500,
        plot_height=500,
        tools=[hover, ResetTool(), WheelZoomTool(), BoxZoomTool(), SaveTool()],
        title=title,
        toolbar_location="below",
    )
    p.add_layout(Title(text=sub_title, text_font_style="italic"), 'above')

    p.circle(
        x="x",
        y="y",
        radius=0.02,
        alpha=0.05,
        source=source_back,
        color={"field": "label", "transform": color_mapper_back},
    )
    p.circle(
        x="x",
        y="y",
        radius=pnt_rad,
        alpha=1,
        source=source_fore,
        line_width=1,
        line_color='black',
        color={"field": "label", "transform": color_mapper_fore},
    )
    p.xaxis.visible = False
    p.xgrid.visible = False
    p.yaxis.visible = False
    p.ygrid.visible = False

    save(p)


# %%
plot_average_levels(sentence_data,
                    level_a_embs,
                    'A',
                    'outputs/skills_taxonomy/figures/2021.09.06/average_hier_A.html',
                   title='Sentences coloured by hierarchy A level',
                   sub_title='Average embeddings overlaid')


# %%
plot_average_levels(sentence_data, level_b_embs, 'B',
                    'outputs/skills_taxonomy/figures/2021.09.06/average_hier_B.html',
                    pnt_rad=0.2,
                   title='Sentences coloured by hierarchy B level',
                    sub_title='Average embeddings overlaid and coloured by hierarchy A')


# %%
plot_average_levels(sentence_data, level_c_embs, 'C',
                    'outputs/skills_taxonomy/figures/2021.09.06/average_hier_C.html',
                    pnt_rad=0.15,
                   title='Sentences coloured by hierarchy C level',
                    sub_title='Average embeddings overlaid and coloured by hierarchy B')


# %% [markdown]
# ## Skills coloured by hierarchy
# - skills in background (average embedding of sentences)
# - hierarchy similar colours

# %%
skill_embs = sentence_data.groupby(
        'Cluster number')[['reduced_points x', 'reduced_points y']].mean().to_dict(orient='index')

for k in skill_hierarchy.keys():
    skill_hierarchy[k]['Average reduced_points x'] = skill_embs[int(k)]['reduced_points x']
    skill_hierarchy[k]['Average reduced_points y'] = skill_embs[int(k)]['reduced_points y']


# %%
skill_hierarchy_df = pd.DataFrame(skill_hierarchy).T
skill_hierarchy_df['Skill number'] = skill_hierarchy_df.index
skill_hierarchy_df['Hierarchy level A name'] = skill_hierarchy_df['Hierarchy level A'].apply(
    lambda x: level_a_rename_dict[str(x)])
skill_hierarchy_df['Hierarchy level D'] = skill_hierarchy_df['Hierarchy level C'].astype(str) + '-'+ skill_hierarchy_df['Hierarchy level D'].astype(str)
skill_hierarchy_df.head(2)


# %%
def plot_average_levels_by_skill(
    skill_hierarchy_df, level_embs_df, filename,
    col_by_back, title, sub_title=False, pnt_rad=0.2, plot_mean_point=True, invisible_back=False):

    output_file(filename=filename)

    # Data for the background (skill average embeddings)
    colors_by_labels_back = skill_hierarchy_df[f'Hierarchy level {col_by_back}'].tolist()
    source_back = ColumnDataSource(dict(
        x=skill_hierarchy_df['Average reduced_points x'].tolist(),
        y=skill_hierarchy_df['Average reduced_points y'].tolist(),
        label=skill_hierarchy_df[f'Hierarchy level {col_by_back}'].tolist(),
        texts=skill_hierarchy_df[f"Skill name"].tolist()
    ))
    color_mapper_back = LinearColorMapper(palette="Turbo256", low=0, high=len(list(set(colors_by_labels_back))) + 1)

    hover = HoverTool(
            tooltips=[
                ("Name", "@texts"),
            ]
        )
    if plot_mean_point:
        # Data for the foreground (hierarchy level average embeddings)
        colors_by_labels = level_embs_df[f'Next hierarchy'].tolist()
        ds_dict_fore = dict(
            x=level_embs_df['reduced_points x'].tolist(),
            y=level_embs_df['reduced_points y'].tolist(),
            texts=level_embs_df["Name"].tolist(),
            label=colors_by_labels,
        )
        
        source_fore = ColumnDataSource(ds_dict_fore)
        color_mapper_fore = LinearColorMapper(palette="Turbo256", low=0, high=len(list(set(colors_by_labels))) + 1)

    p = figure(
        plot_width=500,
        plot_height=500,
        tools=[hover, ResetTool(), WheelZoomTool(), BoxZoomTool(), SaveTool()],
        title=title,
        toolbar_location="below",
    )
    if sub_title:
        p.add_layout(Title(text=sub_title, text_font_style="italic"), 'above')
    else:
        p.add_layout(Title(text=' ', text_font_style="italic"), 'above')

    if invisible_back:
        rad=0
    else:
        rad=0.04
    p.circle(
        x="x",
        y="y",
        radius=rad,
        alpha=0.6,
        source=source_back,
        color={"field": "label", "transform": color_mapper_back},
    )
    if plot_mean_point:
        p.circle(
            x="x",
            y="y",
            radius=pnt_rad,
            alpha=1,
            source=source_fore,
            line_width=1,
            line_color='black',
            color={"field": "label", "transform": color_mapper_fore},
        )
    p.xaxis.visible = False
    p.xgrid.visible = False
    p.yaxis.visible = False
    p.ygrid.visible = False

    save(p)


# %%
plot_average_levels_by_skill(
    skill_hierarchy_df,
    level_embs_df=[],
    filename='outputs/skills_taxonomy/figures/2021.09.06/skills_col_hier_A.html',
    col_by_back='A', 
    title='Skills coloured by hierarchy A level', plot_mean_point=False)

# %%
plot_average_levels_by_skill(
    skill_hierarchy_df,
    level_embs_df=[],
    filename='outputs/skills_taxonomy/figures/2021.09.06/skills_col_hier_B.html',
    col_by_back='B', 
    title='Skills coloured by hierarchy B level', plot_mean_point=False)

# %%
plot_average_levels_by_skill(
    skill_hierarchy_df,
    level_embs_df=[],
    filename='outputs/skills_taxonomy/figures/2021.09.06/skills_col_hier_C.html',
    col_by_back='C', 
    title='Skills coloured by hierarchy C level', plot_mean_point=False)

# %% [markdown]
# ### Overlay the average skill level

# %%
filename = 'outputs/skills_taxonomy/figures/2021.09.06/skills_col_hier_B_withav.html'
col_by_back = 'B'
title='Skills coloured by hierarchy B level'
sub_title='Average embeddings overlaid and coloured by hierarchy A'
pnt_rad=0.2
level_embs_df = level_b_embs
plot_average_levels_by_skill(skill_hierarchy_df, level_embs_df, filename, col_by_back, title, sub_title, pnt_rad, )



# %%
filename = 'outputs/skills_taxonomy/figures/2021.09.06/skills_col_hier_C_withav.html'
col_by_back = 'C'
title='Skills coloured by hierarchy C level'
sub_title='Average embeddings overlaid and coloured by hierarchy B'
pnt_rad=0.15
level_embs_df = level_c_embs
plot_average_levels_by_skill(skill_hierarchy_df, level_embs_df, filename, col_by_back, title, sub_title, pnt_rad, )


# %% [markdown]
# ### Only plot the average

# %%
filename = 'outputs/skills_taxonomy/figures/2021.09.06/skills_col_hier_B_withav_noback.html'
col_by_back = 'B'
title='Skills coloured by hierarchy B level'
sub_title='Average embeddings overlaid and coloured by hierarchy A'
pnt_rad=0.2
level_embs_df = level_b_embs
plot_average_levels_by_skill(skill_hierarchy_df, level_embs_df, filename, col_by_back, title, sub_title,
                             pnt_rad, invisible_back=True)


# %%
filename = 'outputs/skills_taxonomy/figures/2021.09.06/skills_col_hier_C_withav_noback.html'
col_by_back = 'C'
title='Skills coloured by hierarchy C level'
sub_title='Average embeddings overlaid and coloured by hierarchy B'
pnt_rad=0.15
level_embs_df = level_c_embs
plot_average_levels_by_skill(skill_hierarchy_df, level_embs_df, filename, col_by_back, title, sub_title,
                             pnt_rad, invisible_back=True)

# %% [markdown]
# ## How many skills per level group?

# %%
plt.figure(figsize=(12,3))

ax1 = plt.subplot(141)
skill_hierarchy_df.groupby(['Hierarchy level A'])['Skill number'].count().plot.bar(
    color=[255/255,0,65/255], ax=ax1, title="Number of skills in each\nlevel A group",ec='black')

ax2 = plt.subplot(142)
skill_hierarchy_df.groupby(['Hierarchy level B'])['Skill number'].count().plot.hist(
    color=[255/255,90/255,0/255], ax=ax2, title="Number of skills in each\nlevel B group",ec='black')

ax3 = plt.subplot(143)
skill_hierarchy_df.groupby(['Hierarchy level C'])['Skill number'].count().plot.hist(
    color=[165/255,148/255,130/255], ax=ax3, title="Number of skills in each\nlevel C group",ec='black')

ax4 = plt.subplot(144)
skill_hierarchy_df.groupby(['Hierarchy level D'])['Skill number'].count().plot.hist(
    color='black', ax=ax4, title="Number of skills in each\nlevel D group",ec='black',
    bins=50)

plt.tight_layout()
plt.savefig('outputs/skills_taxonomy/figures/2021.09.06/num_skills_per_level.pdf',bbox_inches='tight')

# %%
