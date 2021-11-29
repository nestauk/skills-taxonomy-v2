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
# ## Analysing labelled data to find good distance threshold for merging clusters
#
# The data that was labelled was created by clustering using `dbscan_eps = 0.01` and `dbscan_min_samples = 4`. When fit to 300000 random sentences this produces:
# - 11551 clusters
# - 0 clusters which are larger than 10,000 sentences
# - 8892 clusters which have <10 sentences
# - 117,923 sentences not clustered
# - Average size of cluster is 16 sentences
#
# Here we find a threshold of about `0.05` gives a good prediction of whether two clusters should be merged or not based off 108 labelled data points.

# %%
import yaml
import random
from tqdm import tqdm
import json
from collections import Counter

import pandas as pd
import numpy as np
import boto3
from sklearn import metrics
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import euclidean_distances

from skills_taxonomy_v2.getters.s3_data import get_s3_data_paths, save_to_s3, load_s3_data
from skills_taxonomy_v2.pipeline.skills_extraction.extract_skills_utils import (
    load_sentences_embeddings,ExtractSkills
    )
from skills_taxonomy_v2 import BUCKET_NAME

# %%
test_merge_thresh_labelled = pd.read_csv("test_merge_thresh_manual_labels_IMAN2.csv")

# %%
test_merge_thresh_labelled = test_merge_thresh_labelled[test_merge_thresh_labelled["Should merge?"].notnull()]
test_merge_thresh_labelled["Should merge?"] = test_merge_thresh_labelled["Should merge?"].str.rstrip()
test_merge_thresh_labelled["Small cluster is a skill?"] = test_merge_thresh_labelled["Small cluster is a skill?"].str.rstrip()

test_merge_thresh_labelled["Centroid distance log"] = test_merge_thresh_labelled["Centroid distance"].apply(lambda x: np.log10(x))
test_merge_thresh_labelled["Average small cluster sentence length"] = test_merge_thresh_labelled["Small cluster sentences"].apply(lambda x: np.mean([len(s) for s in x.split(",")]))

len(test_merge_thresh_labelled)


# %%
test_merge_thresh_labelled["Should merge?"].value_counts()

# %%
test_merge_thresh_labelled.boxplot(column="Centroid distance", by="Should merge?")

# %%
# test_merge_thresh_labelled.boxplot(column="Centroid distance", by="Should merge?")

# %%
distances = test_merge_thresh_labelled["Centroid distance"].tolist()
truth = test_merge_thresh_labelled["Should merge?"].tolist()
# If you want to convert the maybes to true:
truth = ["TRUE" if t=="MAYBE" else t for t in truth]
# # If you want to convert the maybes to false:
# truth = ["FALSE" if t=="MAYBE" else t for t in truth]
    
dist_thresh_dict = {}
for dist_thresh in list(np.arange(0.01, 0.5, step=0.001)):
    
    prediction = ["TRUE" if d<dist_thresh else "FALSE" for d in distances]
    num_greater_than_thresh = len([d for d in distances if d>=dist_thresh])
    
    dist_thresh_dict[dist_thresh] = {
        "prop_correct": sum([a==b for a,b in zip(prediction, truth)])/len(truth),
        "prop_correct_truth_true": sum([((a==b) and (b=="TRUE")) for a,b in zip(prediction, truth)])/len(truth),
        "prop_correct_truth_false": sum([((a==b) and (b=="FALSE")) for a,b in zip(prediction, truth)])/len(truth),
        "prop_incorrect_truth_true": sum([((a!=b) and (b=="TRUE")) for a,b in zip(prediction, truth)])/len(truth),
        "prop_incorrect_truth_false": sum([((a!=b) and (b=="FALSE")) for a,b in zip(prediction, truth)])/len(truth),
        "num_data": num_greater_than_thresh
    }

# %%
best_stats = dist_thresh_dict[0.04999999999999997]
print(f"The predictions are true {round(best_stats['prop_correct'],2)} of the time")
print(f"When the truth is to merge {round(best_stats['prop_correct_truth_true'],2)} of the time we do merge")
print(f"When the truth is to not merge {round(best_stats['prop_correct_truth_false'],2)} of the time we don't merge")
print(f"When the truth is to merge {round(best_stats['prop_incorrect_truth_true'],2)} of the time we don't merge")
print(f"When the truth is to not merge {round(best_stats['prop_incorrect_truth_false'],2)} of the time we do merge")


# %%
fig,ax = plt.subplots()
x = [k for k,v in dist_thresh_dict.items()]
y = [v["prop_correct"] for k,v in dist_thresh_dict.items()]
yt = [v["prop_correct_truth_true"] for k,v in dist_thresh_dict.items()]
yf = [v["prop_correct_truth_false"] for k,v in dist_thresh_dict.items()]
y2 = [v["num_data"] for k,v in dist_thresh_dict.items()]

ax.plot(x, y, color="black", label="Accuracy")
ax.plot(x, yt, color="red", label="TP rate")
ax.plot(x, yf, color="purple", label="TN rate")
ax.set_ylabel("Proportion of data points\ncorrectly predicted")
ax.set_xlabel("Centroid distance threshold")
plt.title("Finding a threshold distance to merge clusters\n(if distance between 2 clusters is < this then merge)")
plt.legend()

ax2=ax.twinx()
ax2.plot(x, y2, color="blue")
ax2.set_ylabel("Number of data points with\ndistance greater than this", color="blue")

plt.axvline(0.05, color="orange");
plt.savefig("../figures/nov_2021/finding_good_merge_params2.pdf")
plt.savefig("../figures/nov_2021/finding_good_merge_params2.png")


# %%
### Where did things go wrong?
dist_thresh = 0.05
predicted = []
wrong_rows=pd.DataFrame()
for i, row in test_merge_thresh_labelled.iterrows():
    pred_label = "FALSE"
    if row["Centroid distance"] < dist_thresh:
        pred_label = "TRUE"
    predicted.append(pred_label)
    true_label = "TRUE" if row["Should merge?"]=="MAYBE" else row["Should merge?"]
    if pred_label!= true_label:
        new_row = pd.DataFrame(row).T
        new_row["Merge prediction"] = pred_label
        wrong_rows = pd.concat([wrong_rows, new_row])

# %%
test_merge_thresh_labelled["Merge prediction"] = predicted
test_merge_thresh_labelled["Should merge - maybe is true"] = test_merge_thresh_labelled["Should merge?"].apply(
    lambda x: "TRUE" if x=="MAYBE" else x)


# %%
test_merge_thresh_labelled.groupby(["Should merge - maybe is true",
                                    "Merge prediction"])['Centroid distance'].count()


# %%
test_merge_thresh_labelled.groupby(["Should merge - maybe is true","Merge prediction","Small cluster is a skill?"])['Centroid distance'].count()


# %%
test_merge_thresh_labelled["Should merge?"].value_counts()

# %%
wrong_rows["Should merge?"].value_counts()

# %%
wrong_rows["Merge prediction"].value_counts()

# %%
right_rows = test_merge_thresh_labelled[~test_merge_thresh_labelled["Unnamed: 0"].isin(wrong_rows["Unnamed: 0"].tolist())]

wrong_rows["Average small cluster sentence length"].hist(alpha=0.5, color="green")
right_rows["Average small cluster sentence length"].hist(alpha=0.5, color="red")


# %%
print(f"With a distance threshold of {dist_thresh} we find {round(len(right_rows)/len(test_merge_thresh_labelled),2)} correct merge decisions on our sample of {len(test_merge_thresh_labelled)} data points.")
print(f"We find the mean sentence length for wrong merge decisions as {wrong_rows['Average small cluster sentence length'].mean().round(2)},")
print(f"and the mean sentence length of correct merge decisions as {right_rows['Average small cluster sentence length'].mean().round(2)}.")
print(f"The proportion of small clusters not being a good skill cluster is {round(wrong_rows['Small cluster is a skill?'].value_counts()['FALSE']/len(wrong_rows),2)} in the wrong merge decisions")
print(f"and {round(right_rows['Small cluster is a skill?'].value_counts()['FALSE']/len(right_rows),2)} in the correct merge decisions.")
print("In all - bad merge decisions are often because the cluster isn't really a skill anyway.")


# %%
# Truth is don't merge and the small cluster is a good skill cluster
wrong_rows[((wrong_rows["Should merge?"]=="FALSE") & (wrong_rows['Small cluster is a skill?']!="FALSE"))][
    ["Small cluster sentences", "Merge into cluster sentences (10 examples)"]].values.tolist()

# %%
test_merge_thresh_labelled_nomaybe = test_merge_thresh_labelled[test_merge_thresh_labelled["Should merge?"]!="MAYBE"]
distances = test_merge_thresh_labelled_nomaybe["Centroid distance"].tolist()
truth = test_merge_thresh_labelled_nomaybe["Should merge?"].tolist()
    
dist_thresh_dict_nomaybe = {}
for dist_thresh in list(np.arange(0.01, 0.5, step=0.001)):
    
    prediction = ["TRUE" if d<dist_thresh else "FALSE" for d in distances]
    num_greater_than_thresh = len([d for d in distances if d>=dist_thresh])
    dist_thresh_dict_nomaybe[dist_thresh] = {
        "prop_correct": sum([a==b for a,b in zip(prediction, truth)])/len(truth),
        "num_data": num_greater_than_thresh
    }
    
fig,ax = plt.subplots()
x = [k for k,v in dist_thresh_dict_nomaybe.items()]
y = [v["prop_correct"] for k,v in dist_thresh_dict_nomaybe.items()]
y2 = [v["num_data"] for k,v in dist_thresh_dict_nomaybe.items()]

ax.plot(x, y, color="black")
ax.set_ylabel("Proportion of data points\ncorrectly predicted")
ax.set_xlabel("Centroid distance threshold")
plt.title("Finding a threshold distance to merge clusters\n(if distance between 2 clusters is < this then merge)")

ax2=ax.twinx()
ax2.plot(x, y2, color="blue")
ax2.set_ylabel("Number of data points with\ndistance greater than this", color="blue")

plt.axvline(0.04, color="orange");


# %% [markdown]
# ## Is there anything about the small cluster being a good skill or not based off average sentence length?
# Nothing for this sample

# %%
test_merge_thresh_labelled["Small cluster is a skill?"].value_counts()

# %%
test_merge_thresh_labelled.boxplot(column="Average small cluster sentence length", by="Small cluster is a skill?")


# %%
