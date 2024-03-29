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
skills_date_stamp = '2022.01.14'
hier_date_stamp = '2022.01.21'
output_folder = f"outputs/skills_taxonomy/figures/{hier_date_stamp}/"

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
    LabelSet,
    Text,
    LogTicker
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
# ## Load the hierarchy and original sentence information

# %%
hier_structure_file = f"outputs/skills_taxonomy/{hier_date_stamp}_hierarchy_structure.json"
hier_structure = load_s3_data(s3, bucket_name, hier_structure_file)

# %%
skill_hierarchy_file = f"outputs/skills_taxonomy/{hier_date_stamp}_skills_hierarchy_named.json"
skill_hierarchy = load_s3_data(s3, bucket_name, skill_hierarchy_file)

# %%
skill_hierarchy_df = pd.DataFrame(skill_hierarchy).T
skill_hierarchy_df["Skill number"] = skill_hierarchy_df.index
skill_hierarchy_df.head(2)


# %%
skill_hierarchy_df['Hierarchy level A name'] = skill_hierarchy_df['Hierarchy level A name'].apply(lambda x: x.replace("Cognitative", "Cognitive"))

# %%
# Find the number of unique skill names before the enhanced naming was applied
orig_skill_names = []
for _, level_a_info in hier_structure.items():
    for _, level_b_info in level_a_info["Level B"].items():
        for _, level_c_info in level_b_info["Level C"].items():
            for _, skill_info in level_c_info["Skills"].items():
                orig_skill_names.append(skill_info['Skill name'])
print(f"Before the skill naming based of location in the hierarchy, there were {len(set(orig_skill_names))} unique names for the {len(orig_skill_names)} skills")

# %% [markdown]
# ## Size of levels and how many sentences in each level?

# %%
print(skill_hierarchy_df["Hierarchy level A"].nunique())
print(skill_hierarchy_df["Hierarchy level B"].nunique())
print(skill_hierarchy_df["Hierarchy level C"].nunique())
print(skill_hierarchy_df[skill_hierarchy_df["Skill number"].astype(int)>=0]["Skill number"].nunique())
print(len(skill_hierarchy_df))

# %%
skill_hierarchy_df["Number of sentences that created skill"].sum()

# %%
skill_hierarchy_df.groupby(["Hierarchy level A"])["Number of sentences that created skill"].sum()

# %%
skill_hierarchy_df.groupby(["Hierarchy level A","Hierarchy level B", "Hierarchy level C"])["Number of sentences that created skill"].sum()

# %%
print(f"The average number of skills in each level A group is {skill_hierarchy_df.groupby('Hierarchy level A')['Skill number'].nunique().mean()}")
print(f"The average number of skills in each level B group is {skill_hierarchy_df.groupby('Hierarchy level B')['Skill number'].nunique().mean()}")
print(f"The average number of skills in each level C group is {skill_hierarchy_df.groupby('Hierarchy level C')['Skill number'].nunique().mean()}")


# %% [markdown]
# ## Unique skill names

# %%
skill_hierarchy_df['Skill name'].nunique()

# %% [markdown]
# ## Taxonomy examples

# %%
[k for k, v in skill_hierarchy.items() if "python" in v["Skill name"]][0:10]

# %%
skill_hierarchy["189"]

# %%
[
    (kk, vv["Skill name"])
    for kk, vv in hier_structure["7"]["Level B"]["38"]["Level C"]["182"][
        "Skills"
    ].items()
]

# %%
hier_structure["7"]["Level B"]["38"]["Level C"]["182"]["Skills"]["189"]

# %% [markdown]
# ## Get examples

# %%
level_a = "7"
level_b = "38"
for level_c, level_c_info in hier_structure[level_a]["Level B"][level_b]["Level C"].items():
    if level_c not in ['102']:
        print("---")
        print(level_c)
        print([v["Example sentences with skill in"] for v in level_c_info['Skills'].values()])

# %% [markdown]
# ## Level A colour shuffler
# random seed 1,9,12,15 is good

# %%
import random
def level_a_colour_shuffler(level_a_list):
    """
    Take a list of the level A numbers, then
    re-map these to different number consistently
    e.g. all 0's turn to 8's
    for colouring plots purposes
    """
    
    leva_groups = list(set(level_a_list))
    random.seed(15)
    random.shuffle(leva_groups)
    lev_a_col_shuffle_dict = {i:v for i,v in enumerate(leva_groups)}
    lev_a_col_shuffle_dict[10]="black"
    return  [lev_a_col_shuffle_dict[c] for c in level_a_list]


# %% [markdown]
# ## Plot skills in 2D space

# %%
skill_hierarchy_df['reduced_points x'] = skill_hierarchy_df['Skill centroid'].apply(lambda x: x[0])
skill_hierarchy_df['reduced_points y'] = skill_hierarchy_df['Skill centroid'].apply(lambda x: x[1])

# %%
skill_hierarchy_df.head(2)

# %%
len(skill_hierarchy_df)

# %%

output_file(filename=output_folder+"skills_2d.html")

source = ColumnDataSource(
    dict(
        x=skill_hierarchy_df["reduced_points x"].tolist(),
        y=skill_hierarchy_df["reduced_points y"].tolist(),
        texts=skill_hierarchy_df["Skill name"].tolist(),
        skill_num=skill_hierarchy_df["Skill number"].tolist(),
    )
)
hover = HoverTool(tooltips=[("Name", "@texts"),("ID", "@skill_num")])

p = figure(
    plot_width=500,
    plot_height=500,
    tools=[hover, ResetTool(), WheelZoomTool(), BoxZoomTool(), SaveTool()],
    title="Skills",
    toolbar_location="below",
)
p.circle(
    x="x",
    y="y",
    radius=0.05,
    alpha=0.2,
    source=source,
    color="black",
)

p.xaxis.visible = False
p.xgrid.visible = False
p.yaxis.visible = False
p.ygrid.visible = False

save(p)

# %%
skill_hierarchy_df.head(2)

# %% [markdown]
# ### Plot skills and label a few of them

# %%
cool_skills = ["4421","4435","4854","4070","2245",
              "4045", "1517","5768","6220","4540","4748","2719","4988",
              "5707","4128","130","4767","493","3420","2499","4238",
              "2284","1444","3234","5810","6558","902","501","3345","1275","326","4670",
              "3871","2998","2347"]

skill_hierarchy_df_texts = skill_hierarchy_df[skill_hierarchy_df['Skill number'].isin(cool_skills)]
skill_hierarchy_df_texts["Skill name"] = skill_hierarchy_df_texts["Skill name"].apply(lambda x: x.split("-")[0])
                                              
                                              
                                              

# %%
output_file(filename=output_folder+"skills_2d_labelled.html")

source = ColumnDataSource(
    dict(
        x=skill_hierarchy_df["reduced_points x"].tolist(),
        y=skill_hierarchy_df["reduced_points y"].tolist(),
        texts=skill_hierarchy_df["Skill name"].tolist(),
        skill_num=skill_hierarchy_df["Skill number"].tolist(),
    )
)
hover = HoverTool(tooltips=[("Name", "@texts"),("ID", "@skill_num")])

p = figure(
    plot_width=600,
    plot_height=600,
    tools=[hover, ResetTool(), WheelZoomTool(), BoxZoomTool(), SaveTool()],
    title="Skills",
    toolbar_location="below",
)
p.circle(
    x="x",
    y="y",
    radius=0.05,
    alpha=0.2,
    source=source,
    color="grey",
)

source_text = ColumnDataSource(
    dict(
        x=skill_hierarchy_df_texts["reduced_points x"].tolist(),
        y=skill_hierarchy_df_texts["reduced_points y"].tolist(),
        texts=skill_hierarchy_df_texts["Skill name"].tolist()
    )
)

glyph = Text(
    x="x", y="y", text="texts",
    angle=0, text_color="black", text_font_size="7pt", x_offset=-40,y_offset=0
)
p.add_glyph(source_text, glyph)

p.xaxis.visible = False
p.xgrid.visible = False
p.yaxis.visible = False
p.ygrid.visible = False

save(p)

# %% [markdown]
# ## Skills coloured by levels

# %%
skill_hierarchy_df.head(2)


# %%
def plot_skills_col_level(col_by,legend=False, col_skill_highlight='0'):
    
    colors_by_labels = skill_hierarchy_df[f"Hierarchy level {col_by}"].tolist()
    if col_by=="A":
        colors_by_labels = level_a_colour_shuffler(colors_by_labels)
        if col_skill_highlight:
            colors_by_labels = [c if c == col_skill_highlight else 0 for c in colors_by_labels]

    output_file(filename=f"{output_folder}skills_2d_col_lev_{col_by}.html")
    if col_skill_highlight:
        output_file(filename=f"{output_folder}skills_2d_col_lev_{col_by}_highlight.html")


    source = ColumnDataSource(
        dict(
            x=skill_hierarchy_df["reduced_points x"].tolist(),
            y=skill_hierarchy_df["reduced_points y"].tolist(),
            level_a=skill_hierarchy_df["Hierarchy level A"].tolist(),
            level_b=skill_hierarchy_df["Hierarchy level B"].tolist(),
            level_c=skill_hierarchy_df["Hierarchy level C"].tolist(),
            level_a_name=skill_hierarchy_df["Hierarchy level A name"].tolist(),
            level_b_name=skill_hierarchy_df["Hierarchy level B name"].tolist(),
            level_c_name=skill_hierarchy_df["Hierarchy level C name"].tolist(),
            label=colors_by_labels,
            texts=skill_hierarchy_df["Skill name"].tolist(),
            legend_name=skill_hierarchy_df[f"Hierarchy level {col_by} name"].tolist()
        )
    )
    hover = HoverTool(
        tooltips=[
            ("Name", "@texts"),("Level A", "@level_a_name"+"("+"@level_a"+")"),("Level B", "@level_b_name"+"("+"@level_b"+")"),("Level C", "@level_c_name"+"("+"@level_c"+")"),])

    color_mapper = LinearColorMapper(
            palette="Turbo256", low=0, high=len(list(set(colors_by_labels))),

        )
    if col_skill_highlight:
        color_mapper = LinearColorMapper(
            palette="Turbo256", low=0, high=12,

        )

    p = figure(
        plot_width=500,
        plot_height=500,
        tools=[hover, ResetTool(), WheelZoomTool(), BoxZoomTool(), SaveTool()],
        title=f"Skills coloured by level {col_by}",
        toolbar_location="below",
    )
    if legend:
        p.circle(
            x="x",
            y="y",
            radius=0.05,
            alpha=0.5,
            source=source,
            color={"field": "label", "transform": color_mapper},
            legend_field='legend_name'
        )
        p.legend.location = "bottom_left"
        p.legend.label_text_font_size= "5pt"
        p.legend.spacing=0
        p.legend.padding=0
        p.legend.glyph_height=10
    else:
        p.circle(
            x="x",
            y="y",
            radius=0.05,
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
plot_skills_col_level(col_by="A", col_skill_highlight=False)
plot_skills_col_level(col_by="B", col_skill_highlight=False)
plot_skills_col_level(col_by="C", col_skill_highlight=False)


# %%
plot_skills_col_level(col_by="A",col_skill_highlight=5)

# %% [markdown]
# ## Plot average embeddings for each level coloured by next level

# %%
level_a_names = {}
level_b_names = {}
level_c_names = {}
for level_a_num, level_a_info in hier_structure.items():
    level_a_names[level_a_num] = level_a_info["Name"]
    for level_b_num, level_b_info in level_a_info["Level B"].items():
        level_b_names[level_b_num] = level_b_info["Name"]
        for level_c_num, level_c_info in level_b_info["Level C"].items():
            level_c_names[level_c_num] = level_c_info["Name"]

# %%
skill_hierarchy_df.head(2)

# %%
a2o_dict = (
    skill_hierarchy_df.groupby("Hierarchy level A")["Hierarchy level A"]
    .unique()
    .apply(lambda x: 0)
)  # This is just for viz reasons
b2a_dict = (
    skill_hierarchy_df.groupby("Hierarchy level B")["Hierarchy level A"]
    .unique()
    .apply(lambda x: x[0])
)
c2b_dict = (
    skill_hierarchy_df.groupby("Hierarchy level C")["Hierarchy level B"]
    .unique()
    .apply(lambda x: x[0])
)


# %%
def get_level_average_emb(
    skill_hierarchy_df,
    level_name,
    level_a_names=level_a_names,
    level_b_names=level_b_names,
    level_c_names=level_c_names,
    a2o_dict=a2o_dict,
    b2a_dict=b2a_dict,
    c2b_dict=c2b_dict,
):
    level_embs = (
        skill_hierarchy_df.groupby(f"Hierarchy level {level_name}")[
            ["reduced_points x", "reduced_points y"]
        ]
        .mean()
        .to_dict(orient="index")
    )
    for k in level_embs.keys():
        if level_name == "A":
            level_embs[k]["Name"] = level_a_names[str(k)]
            level_embs[k]["Next hierarchy"] = a2o_dict[k]
        elif level_name == "B":
            level_embs[k]["Name"] = level_b_names[str(k)]
            level_embs[k]["Next hierarchy"] = b2a_dict[k]
        elif level_name == "C":
            level_embs[k]["Name"] = level_c_names[str(k)]
            level_embs[k]["Next hierarchy"] = c2b_dict[k]

    level_embs = pd.DataFrame(level_embs).T
    level_embs[f"Level {level_name} number"] = level_embs.index
    return level_embs


# %%
level_a_embs = get_level_average_emb(skill_hierarchy_df, "A")
level_b_embs = get_level_average_emb(skill_hierarchy_df, "B")
level_c_embs = get_level_average_emb(skill_hierarchy_df, "C")

# %%
level_a_embs

# %%
skill_hierarchy_df.columns


# %%
def plot_average_levels(
    level_embs_df,
    skill_hierarchy_df,
    col_by,
    filename,
    title="",
    sub_title="",
    pnt_rad=0.4,
    plot_skills=True
):
    output_file(filename=filename)

    # Data for the foreground (hierarchy level average embeddings)
    colors_by_labels = level_embs_df[f"Next hierarchy"].tolist()
    
    if col_by=="B":
        colors_by_labels = level_a_colour_shuffler(colors_by_labels)
        
    ds_dict_fore = dict(
        x=level_embs_df["reduced_points x"].tolist(),
        y=level_embs_df["reduced_points y"].tolist(),
        texts=level_embs_df["Name"].tolist(),
        label=colors_by_labels,
    )
    hover = HoverTool(tooltips=[("Name", "@texts"),])
    source_fore = ColumnDataSource(ds_dict_fore)
    color_mapper_fore = LinearColorMapper(
        palette="Turbo256", low=0, high=len(list(set(colors_by_labels))) + 1
    )

    # Data for the background (skill embeddings)
    colors_by_back_labels = skill_hierarchy_df[f"Hierarchy level {col_by}"].tolist()
    if col_by=="A":
        colors_by_back_labels = level_a_colour_shuffler(colors_by_back_labels)
        
    source_back = ColumnDataSource(
        dict(
            x=skill_hierarchy_df["reduced_points x"].tolist(),
            y=skill_hierarchy_df["reduced_points y"].tolist(),
            texts=skill_hierarchy_df["Skill name"].tolist(),
            label=colors_by_back_labels
        )
    )
    color_mapper_back= LinearColorMapper(
        palette="Turbo256", low=0, high=len(list(set(colors_by_back_labels))) + 1
    )

    # Plot
    p = figure(
        plot_width=500,
        plot_height=500,
        tools=[hover, ResetTool(), WheelZoomTool(), BoxZoomTool(), SaveTool()],
        title=title,
        toolbar_location="below",
    )
    p.add_layout(Title(text=sub_title, text_font_style="italic"), "above")

    # back
    if plot_skills:
        p.circle(
            x="x",
            y="y",
            radius=0.05,
            alpha=0.5,
            source=source_back,
            color={"field": "label", "transform": color_mapper_back},
        )
    # Front
    p.circle(
        x="x",
        y="y",
        radius=pnt_rad,
        alpha=1,
        source=source_fore,
        line_width=1,
        line_color="black",
        color={"field": "label", "transform": color_mapper_fore},
    )
    p.xaxis.visible = False
    p.xgrid.visible = False
    p.yaxis.visible = False
    p.ygrid.visible = False

    save(p)


# %%
plot_average_levels(
    level_a_embs,
    skill_hierarchy_df,
    "A",
    output_folder+"average_hier_A_withskills.html",
    title="Skills coloured by hierarchy A level",
    sub_title="Centers of level A groups overlaid",
)

plot_average_levels(
    level_b_embs,
    skill_hierarchy_df,
    "B",
    output_folder+"average_hier_B_withskills.html",
    title="Skills coloured by hierarchy B level",
    sub_title="Centers of level B groups overlaid",
    pnt_rad=0.3
)

plot_average_levels(
    level_c_embs,
    skill_hierarchy_df,
    "C",
    output_folder+"average_hier_C_withskills.html",
    title="Skills coloured by hierarchy C level",
    sub_title="Centers of level C groups overlaid",
    pnt_rad=0.1
)


# %% [markdown]
# ## Plot without skills in background

# %%
plot_average_levels(
    level_a_embs,
    skill_hierarchy_df,
    "A",
    output_folder+"average_hier_A.html",
    title="Centers of level A groups",
    sub_title="",
    plot_skills=False
)

plot_average_levels(
    level_b_embs,
    skill_hierarchy_df,
    "B",
    output_folder+"average_hier_B.html",
    title="Centers of level B groups",
    sub_title="Coloured by level A group",
    pnt_rad=0.3,
    plot_skills=False
)

plot_average_levels(
    level_c_embs,
    skill_hierarchy_df,
    "C",
    output_folder+"average_hier_C.html",
    title="Centers of level C groups",
    sub_title="Coloured by level B group",
    pnt_rad=0.1,
    plot_skills=False
)

# %% [markdown]
# ## How many skills per level group?

# %%
skill_hierarchy_df.head(2)

# %%
plt.figure(figsize=(12, 3))

ax1 = plt.subplot(131)
skill_hierarchy_df.groupby(["Hierarchy level A name"])["Skill number"].count().plot.barh(
    color=[255 / 255, 0, 65 / 255],
    ax=ax1,
    title="Number of skills in each\nlevel A group",
    ec="black",
)

ax2 = plt.subplot(132)
skill_hierarchy_df.groupby(["Hierarchy level B"])["Skill number"].count().plot.hist(
    color=[255 / 255, 90 / 255, 0 / 255],
    ax=ax2,
    title="Number of skills in each\nlevel B group",
    ec="black",
)

ax3 = plt.subplot(133)
skill_hierarchy_df.groupby(["Hierarchy level C"])["Skill number"].count().plot.hist(
    color=[165 / 255, 148 / 255, 130 / 255],
    ax=ax3,
    title="Number of skills in each\nlevel C group",
    ec="black",
)

plt.tight_layout()
plt.savefig(
    output_folder+"num_skills_per_level.pdf",
    bbox_inches="tight",
)

# %%
len(skill_hierarchy_df[skill_hierarchy_df["Hierarchy level A name"]=="Cognitive skills and languages"])


# %%
small_bits = []
big_bits = []
for level_a_num, level_a_info in hier_structure.items():
    for level_b_num, level_b_info in level_a_info["Level B"].items():
        for level_c_num, level_c_info in level_b_info["Level C"].items():
            if level_c_info['Number of skills']<2:
                small_bits.append([level_a_num, level_b_num, level_c_num])
            if level_c_info['Number of skills']>50:
                big_bits.append([level_a_num, level_b_num, level_c_num])
print(len(small_bits))
len(big_bits)

# %%
for a,b,c in small_bits:
    print('---')
    print(hier_structure[a]['Name'])
    print(hier_structure[a]['Level B'][b]['Name'])
    print(hier_structure[a]['Level B'][b]['Level C'][c]['Name'])
    print(hier_structure[a]['Level B'][b]['Level C'][c]['Number of skills'])
    for skills in hier_structure[a]['Level B'][b]['Level C'][c]['Skills'].values():
        print(skills['Skill name'])

# %%
print(hier_structure['9']['Level B']['95'])

# %%
for a,b,c in big_bits:
    print('---')
    print(hier_structure[a]['Name'])
    print(hier_structure[a]['Level B'][b]['Name'])
    print(hier_structure[a]['Level B'][b]['Level C'][c]['Name'])
    print(hier_structure[a]['Level B'][b]['Level C'][c]['Number of skills'])
    for v in list(hier_structure[a]['Level B'][b]['Level C'][c]['Skills'].values())[0:5]:
        print(v['Skill name'])

# %%
sql_skills = []
for level_a_num, level_a_info in hier_structure.items():
    for level_b_num, level_b_info in level_a_info["Level B"].items():
        for level_c_num, level_c_info in level_b_info["Level C"].items():
            for v in list(level_c_info['Skills'].values()):
                if 'must strong dementia care palliative' in v['Skill name']:
                    sql_skills.append((level_a_num, level_b_num, level_c_num))
sql_skills = list(set(sql_skills))
sql_skills

# %%
for a,b,c in sql_skills:
    print('---')
    print(hier_structure[a]['Name'])
    print(hier_structure[a]['Level B'][b]['Name'])
    print(hier_structure[a]['Level B'][b]['Level C'][c]['Name'])
    print(hier_structure[a]['Level B'][b]['Level C'][c]['Number of skills'])
    for v in list(hier_structure[a]['Level B'][b]['Level C'][c]['Skills'].values())[0:50]:
        print(v['Skill name'])

# %%
hier_structure[a]['Level B'][b]['Level C'][c]['Skills']

# %%
ml_skills = []
for level_a_num, level_a_info in hier_structure.items():
    for level_b_num, level_b_info in level_a_info["Level B"].items():
        for level_c_num, level_c_info in level_b_info["Level C"].items():
            for skill_i, v in level_c_info['Skills'].items():
                if 'machine learning' in ' '.join(v['Example sentences with skill in']):
                    ml_skills.append(skill_i)
ml_skills = list(set(ml_skills))


# %%
for ml_skill in ml_skills:
    print(f"{skill_hierarchy[ml_skill]['Hierarchy level A name']} * {skill_hierarchy[ml_skill]['Hierarchy level B name']} * {skill_hierarchy[ml_skill]['Hierarchy level C name']}")

# %% [markdown]
# ## Analysis on sentence level
# - So far it's been on the skill level

# %%
from skills_taxonomy_v2.getters.s3_data import  get_s3_data_paths
from tqdm import tqdm

# %%
skills_ex_date = "2022.01.14"

# %%
# The sentences ID + cluster num
clustered_sentences_path=f"outputs/skills_extraction/extracted_skills/{skills_ex_date}_sentences_skills_data_lightweight.json"
sentence_embs = load_s3_data(s3, bucket_name, clustered_sentences_path)
sentence_embs = pd.DataFrame(sentence_embs, columns=['job id', 'sentence id',  'Cluster number predicted'])

# Just use a sample for ease of plotting
sentence_embs_sample = sentence_embs.sample(n=100000, random_state=1)

# %%
sample_sent_ids = set(sentence_embs_sample['sentence id'].unique())

# %%
# Get the reduced embeddings + sentence texts and the sentence IDs

reduced_embeddings_dir =f"outputs/skills_extraction/reduced_embeddings/{skills_ex_date}"
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
sample_sentence_clusters["reduced_points x"] = sample_sentence_clusters["embedding"].apply(lambda x: x[0])
sample_sentence_clusters["reduced_points y"] = sample_sentence_clusters["embedding"].apply(lambda x: x[1])
sample_sentence_clusters["Cluster number"] = sample_sentence_clusters["Cluster number predicted"]
sample_sentence_clusters.head(3)

# %%
sentence_clusters_notnone = sample_sentence_clusters[sample_sentence_clusters["Cluster number"] >= 0]

# %%
sentence_clusters_notnone['Level A'] = sentence_clusters_notnone['Cluster number'].apply(lambda x: skill_hierarchy[str(x)]['Hierarchy level A'])

sentence_clusters_notnone['Level B'] = sentence_clusters_notnone['Cluster number'].apply(lambda x: skill_hierarchy[str(x)]['Hierarchy level B'])
sentence_clusters_notnone['Level C'] = sentence_clusters_notnone['Cluster number'].apply(lambda x: skill_hierarchy[str(x)]['Hierarchy level C'])
sentence_clusters_notnone['Hierarchy ID'] = sentence_clusters_notnone['Cluster number'].apply(lambda x: skill_hierarchy[str(x)]['Hierarchy ID'])

sentence_clusters_notnone.head(2)


# %%
def plot_sample_sentences_coloured(colour_by_col):
    if colour_by_col=="Cluster number":
        output_name="skill"
    elif colour_by_col=="Level A":
        output_name="A"
    elif colour_by_col=="Level B":
        output_name="B"
    elif colour_by_col=="Level C":
        output_name="C"
        
    output_file(
        filename=output_folder+f"sentences_col_hier_{output_name}.html",)

    colors_by_labels = sentence_clusters_notnone[colour_by_col].astype(str).tolist()
    reduced_x = sentence_clusters_notnone["reduced_points x"].tolist()
    reduced_y = sentence_clusters_notnone["reduced_points y"].tolist()
    color_palette = viridis
    
    if colour_by_col=="Level A":
        colors_by_labels = sentence_clusters_notnone[colour_by_col].tolist()
        colors_by_labels = level_a_colour_shuffler(colors_by_labels)

    ds_dict = dict(
        x=reduced_x,
        y=reduced_y,
        texts=sentence_clusters_notnone["original sentence"].tolist(),
        label=colors_by_labels,
        level_info=sentence_clusters_notnone["Hierarchy ID"].tolist(),
    )
    hover = HoverTool(
        tooltips=[
            ("Sentence", "@texts"),
            (f"Level information", "@level_info"),
        ]
    )
    source = ColumnDataSource(ds_dict)
    unique_colors = list(set(colors_by_labels))
    color_mapper = LinearColorMapper(palette="Turbo256", low=0, high=len(unique_colors) + 1)

    p = figure(
        plot_width=500,
        plot_height=500,
        tools=[hover, ResetTool(), WheelZoomTool(), BoxZoomTool(), SaveTool()],
        title=f"Sample of skill sentences coloured by {output_name} number",
        toolbar_location="below",
    )
    p.circle(x="x", y="y", radius=0.04, alpha=0.2, source=source,
             color={"field": "label", "transform": color_mapper})
    p.xaxis.visible = False
    p.xgrid.visible = False
    p.yaxis.visible = False
    p.ygrid.visible = False

    save(p)


# %%
plot_sample_sentences_coloured("Level A")
plot_sample_sentences_coloured("Level B")
plot_sample_sentences_coloured("Level C")
