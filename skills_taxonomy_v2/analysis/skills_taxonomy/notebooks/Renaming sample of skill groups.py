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

# %%
# cd ../../../..

# %%
import matplotlib.pyplot as plt
import numpy as np
import json
from collections import Counter, defaultdict
import boto3
import pandas as pd
from skills_taxonomy_v2.getters.s3_data import load_s3_data

# %%
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

# %%
bucket_name = "skills-taxonomy-v2"
s3 = boto3.resource("s3")

# %%
sentence_data = load_s3_data(
    s3,
    bucket_name,
    "outputs/skills_extraction/extracted_skills/2021.08.31_sentences_data.json",
)

# %%
sentence_data = pd.DataFrame(sentence_data)
sentence_data = sentence_data[sentence_data["Cluster number"] != -1]

# %%
hier_structure_file = "outputs/skills_hierarchy/2021.09.06_hierarchy_structure.json"
hier_structure = load_s3_data(s3, bucket_name, hier_structure_file)

# %%
skill_hierarchy_file = "outputs/skills_hierarchy/2021.09.06_skills_hierarchy.json"
skill_hierarchy = load_s3_data(s3, bucket_name, skill_hierarchy_file)

# %%
sentence_data["Hierarchy level A"] = (
    sentence_data["Cluster number"]
    .astype(str)
    .apply(lambda x: skill_hierarchy[x]["Hierarchy level A"])
)
sentence_data["Hierarchy level B"] = (
    sentence_data["Cluster number"]
    .astype(str)
    .apply(lambda x: skill_hierarchy[x]["Hierarchy level B"])
)
sentence_data["Hierarchy level C"] = (
    sentence_data["Cluster number"]
    .astype(str)
    .apply(lambda x: skill_hierarchy[x]["Hierarchy level C"])
)
sentence_data["Hierarchy level D"] = (
    sentence_data["Cluster number"]
    .astype(str)
    .apply(lambda x: skill_hierarchy[x]["Hierarchy level D"])
)
sentence_data["Hierarchy ID"] = (
    sentence_data["Cluster number"]
    .astype(str)
    .apply(lambda x: skill_hierarchy[x]["Hierarchy ID"])
)

# %% [markdown]
# ## Plot by hier levels

# %%
col_by_level = "B"
colors_by_labels = sentence_data[f"Hierarchy level {col_by_level}"].astype(str)
reduced_x = sentence_data["reduced_points x"].tolist()
reduced_y = sentence_data["reduced_points y"].tolist()
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

color_mapper = LinearColorMapper(
    palette="Turbo256", low=0, high=len(list(set(colors_by_labels))) + 1
)

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

show(p)

# %%
level_a_n = 2
level_b_n = 38
level_c_n = 132
print(
    hier_structure[str(level_a_n)]["Level B"][str(level_b_n)]["Level C"][
        str(level_c_n)
    ]["Name"]
)

# %%
for leva, v_a in hier_structure.items():
    for levb, v_b in v_a["Level B"].items():
        for levc, v_c in v_b["Level C"].items():
            if int(levc) in level_c_rename_dict.keys():
                print(f"{leva}-{levb}-{levc}")

# %%
level_c_rename_dict = {
    78: "Microsoft Office",
    5: "Microsoft Word/Powerpoint/Excel",
    203: "Microsoft Excel",
    208: "Agile Methods",
    216: "Javascript/html",
    36: "SQL and cloud engineering",
    178: "Python and Linux",
    132: "Vehicle maintenance",
    189: "Driving license",
}

# %%
level_b_rename_dict = {
    49: "Food production",
    9: "Security, GDPR, policing and medical",
    20: "Cleaning, waste management and maintenance",
    16: "Maintenance and construction",
    38: "Mechanics, transport and driving",
    62: "Property administration",
    15: "Teaching and training",
    52: "Childcare",
    47: "Social care and machine learning",
    21: "Healthcare",
    8: "Written and verbal language",
    60: "Agile work",
    26: "Computer software",
    13: "Software development",
    19: "Personal organisation",
}

# %%
level_a_rename_dict = {
    0: "Business administration and management",
    1: "Information technology and languages",
    2: "Safety, finance, maintenance and service",
    3: "Customer service and marketing",
    4: "Personal attributes",
    5: "Teaching and care",
}
level_a_rename_dict

# %%
with open("skills_taxonomy_v2/utils/2021.09.06_level_c_rename_dict.json", "w") as f:
    json.dump(level_c_rename_dict, f)
with open("skills_taxonomy_v2/utils/2021.09.06_level_b_rename_dict.json", "w") as f:
    json.dump(level_b_rename_dict, f)
with open("skills_taxonomy_v2/utils/2021.09.06_level_a_rename_dict.json", "w") as f:
    json.dump(level_a_rename_dict, f)

# %%
