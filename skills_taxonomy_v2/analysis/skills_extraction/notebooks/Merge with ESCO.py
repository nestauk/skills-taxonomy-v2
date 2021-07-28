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
# ## Bottom up skills extraction merged with ESCO
#
# We will combine and deduplicate the topic modelling extracted skills with the ESCO skills.
#
# Note: going to get rid of -1 skill ID completely for now - it isn't useful for this bit.
#
# - How much is in ESCO but not mapped to TK?
# - How much is in TK but not mapped to ESCO?

# %%
# cd ../../../..

# %%
import json
from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

# %%
# Bottom up skills. Don't include the not-skill class '-1'
tk_skills = pd.read_csv("outputs/skills_extraction/data/clustered_data_skillnames.csv")
tk_skills = tk_skills[tk_skills["Unnamed: 0"] != -1]
tk_skills = tk_skills.drop(columns=["Unnamed: 0"])
tk_skills.reset_index(drop=True, inplace=True)
print(len(tk_skills))

# %%
# ESCO skills
# Combine skills + language skills + ICT skills + transversal skills
esco_skills = pd.read_csv("inputs/ESCO/v1.0.8/skills_en.csv")
print(len(esco_skills))
lang_skills = pd.read_csv("inputs/ESCO/v1.0.8/languageSkillsCollection_en.csv")
ict_esco_skills = pd.read_csv("inputs/ESCO/v1.0.8/ictSkillsCollection_en.csv")
trans_esco_skills = pd.read_csv("inputs/ESCO/v1.0.8/transversalSkillsCollection_en.csv")
esco_skills = pd.concat([esco_skills, lang_skills, ict_esco_skills, trans_esco_skills])
esco_skills.reset_index(drop=True, inplace=True)
len(esco_skills)

# %% [markdown]
# ## Clean ESCO descriptions in a similar way to tk skills
# So we are comparing like with like

# %%
import re

from nltk.stem import WordNetLemmatizer

# Init the Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()


# %%
def clean_description(text):
    # Remove capitals, but not when it's an acronym
    acronyms = re.findall("[A-Z]{2,}", text)
    # Lemmatize
    lemmatized_output = [
        lemmatizer.lemmatize(w) if w in acronyms else lemmatizer.lemmatize(w.lower())
        for w in text.split(" ")
    ]
    return " ".join(lemmatized_output).strip()


# %%
# esco_skills_texts = esco_skills['description'].apply(clean_description).tolist()

# %%
# Remove any dashes (for the tfidf tuples) in tk_skills
# Skill name or Sentences with this skill

# tk_skills_texts = tk_skills['text'].apply(lambda x: str(x).replace('-',' ')).tolist()

# %% [markdown]
# ### Or just using original descriptions

# %%
# Use original descriptions for comparisons
esco_skills_texts = esco_skills["description"].tolist()
tk_skills_texts = tk_skills["Description"].tolist()

# %%
print(esco_skills_texts[0])
print(tk_skills_texts[0])

# %% [markdown]
# ## Get mapper to id for ESCO
# This is because somtimes the ESCO skills are referred to by their preferred label not an ID.
#
# {0: 'manage musical staff', 1: ...}

# %%
esco_skill2ID = {v: k for k, v in esco_skills["preferredLabel"].to_dict().items()}
esco_ID2skill = esco_skills["preferredLabel"].to_dict()

# %%
with open("outputs/skills_extraction/data/esco_ID2skill.json", "w") as file:
    json.dump(esco_ID2skill, file)

# %% [markdown]
# ## Get most similar pairs of tk-esco skills.
# vectorization + cosine similarity over a threshold

# %% [markdown]
# ### TFIDF vectorization - should be on cleaned text

# %%
# from sklearn.feature_extraction.text import CountVectorizer

# %%
# # vectorizer = TfidfVectorizer(stop_words="english")
# vectorizer = CountVectorizer(stop_words="english")
# vectorizer.fit(esco_skills_texts + tk_skills_texts)

# %%
# esco_vects = vectorizer.transform(esco_skills_texts)
# esco_vects.shape

# %%
# tk_vects = vectorizer.transform(tk_skills_texts)
# tk_vects.shape

# %% [markdown]
# ### BERT vectorization - can be on original text

# %%
from skills_taxonomy_v2.pipeline.sentence_classifier.sentence_classifier import (
    BertVectorizer,
)

# %%
bert_vectorizer = BertVectorizer(
    bert_model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
    multi_process=True,
)
bert_vectorizer.fit()

# %%
embedded_esco = bert_vectorizer.transform(esco_skills_texts)

# %%
embedded_tk = bert_vectorizer.transform(tk_skills_texts)

# %%
tk_vects = embedded_tk
esco_vects = embedded_esco

# %% [markdown]
# ### Cosine similarities

# %%
similarities = cosine_similarity(tk_vects, esco_vects)
similarities.shape

# %%
all_sims = similarities.flatten()
plt.hist(all_sims[all_sims != 0], bins=100)

# %% [markdown]
# For each TK skill do you want to find the closest ESCO skill to it (over a threshold) and just map to that one? OR you can find all the ESCO skills over a threshold similar to it and map it to multiple?

# %%
## Only map to one ESCO skill if the similarity is over 0.5
tk_esco_skill_close_pairs = []
for tk_id, esco_id in enumerate(np.argmax(similarities, axis=1)):
    similarity_score = similarities[tk_id, esco_id]
    if similarity_score > 0.5:
        tk_esco_skill_close_pairs.append(
            {
                "TK skill id": tk_id,
                "ESCO skill id": esco_id,
                "Cosine similarity": similarity_score,
            }
        )

# %%
## Map to all ESCO skills over 0.5
# tk_esco_skill_close_pairs = []
# for tk_id, esco_id in np.argwhere(similarities>0.5):
#     similarity_score = similarities[tk_id, esco_id]
#     tk_esco_skill_close_pairs.append(
#         {
#             'TK skill id': tk_id,
#             'ESCO skill id': esco_id,
#             'Cosine similarity': similarity_score
#         }
#     )

# %%
# It isn't a one-one mapping, but we can have a mapping {tk_id: list of esco ids}
esco2tk_dict = {}
tk2esco_dict = {}
for sims in tk_esco_skill_close_pairs:
    tk_id = sims["TK skill id"]
    esco_id = sims["ESCO skill id"]
    if esco_id in esco2tk_dict:
        esco2tk_dict[esco_id].append(tk_id)
    else:
        esco2tk_dict[esco_id] = [tk_id]
    if tk_id in tk2esco_dict:
        tk2esco_dict[tk_id].append(esco_id)
    else:
        tk2esco_dict[tk_id] = [esco_id]

# %%
# How many TK skills mapped?
tk_skills_mapped = [t["TK skill id"] for t in tk_esco_skill_close_pairs]
print(
    f"{len(set(tk_skills_mapped))} out of {len(tk_skills)} TK skills were linked with ESCO skills"
)
print(
    f"{len([k for k,v in Counter(tk_skills_mapped).items() if v!=1])} of these were linked to multiple ESCO skills"
)

# %%
# Some ESCO skills matched to multiple TK skills!
# Perhaps there are some TK skills which are very close together?
esco_skills_mapped = [t["ESCO skill id"] for t in tk_esco_skill_close_pairs]
print(
    f"{len(set(esco_skills_mapped))} out of {len(esco_skills)} ESCO skills were linked with TK skills"
)
print(
    f"{len([k for k,v in Counter(esco_skills_mapped).items() if v!=1])} of these were linked to multiple TK skills"
)

# %%
for sims in tk_esco_skill_close_pairs:
    tk_id = sims["TK skill id"]
    esco_id = sims["ESCO skill id"]
    similarity_score = sims["Cosine similarity"]

    tk_skill_details = tk_skills.loc[tk_id][
        ["Skill name", "Description", "text"]
    ].tolist()
    if len(tk_skill_details[2]) < 100:
        print("----")
        print(similarity_score)
        print("ESCO skill:")
        print(esco_skills.loc[esco_id][["preferredLabel", "description"]].tolist())
        print("TK skill:")
        print(tk_skill_details)

# %%
# The very different skills mostly look like junk
sim_low_threshold = 0.1

tk_max_esco_ids = np.argmax(similarities, axis=1)

very_different_tk = []
for tk_ix, esco_id in enumerate(tk_max_esco_ids):
    similarity_score = similarities[tk_ix, esco_id]
    if similarity_score < sim_low_threshold:
        very_different_tk.append(tk_ix)
print(len(very_different_tk))

for tk_id in very_different_tk:
    print(tk_skills.loc[tk_id][["Skill name", "Description", "text"]].tolist())

# %% [markdown]
# ### One id system
# joined_skills = {skill_id: {'Name': , 'Description': , 'ESCO id': esco_id, 'TK id': }}

# %%
joined_skills = {}
skill_id = 0

# Add ESCO skills
for esco_id, row in esco_skills.iterrows():
    joined_skills[skill_id] = {
        "Name": row["preferredLabel"],
        "Description": row["description"],
        "ESCO id": esco_id,
        "TK id": esco2tk_dict.get(esco_id),
    }
    skill_id += 1
print(len(joined_skills))

# The TK skills not mapped to ESCO
tk_not_mapped = set(tk_skills.index.tolist()).difference(set(tk2esco_dict.keys()))

# Add TK skills
for tk_id, row in tk_skills.loc[tk_not_mapped].iterrows():
    joined_skills[skill_id] = {
        "Name": row["Skill name"],
        "Description": row["Description"],
        "ESCO id": None,
        "TK id": tk_id,
    }
    skill_id += 1
print(len(joined_skills))

# %%
print(f"{len([k for k,v in joined_skills.items() if v['ESCO id']])} skills in ESCO")
print(f"{len([k for k,v in joined_skills.items() if v['TK id']])} skills in TK")
print(
    f"{len([k for k,v in joined_skills.items() if (v['ESCO id'] and v['TK id'])])} skills in TK and ESCO"
)
print(
    f"{len([k for k,v in joined_skills.items() if (v['ESCO id'] and not v['TK id'])])} skills in ESCO only"
)
print(
    f"{len([k for k,v in joined_skills.items() if (v['TK id'] and not v['ESCO id'])])} skills in TK only"
)

# %%
joined_skills[0]

# %%
joined_skills[22]

# %%
joined_skills[len(joined_skills) - 1]

# %% [markdown]
# ## What is in ESCO but not TK and vice versa?
#
# Could reflect that the ESCO descriptions have been curated whereas TK come straight from job adverts.

# %%
look_at_col = "Name"  #'Description'
esco_only_words = " ".join(
    [
        v[look_at_col].lower()
        for k, v in joined_skills.items()
        if (v["ESCO id"] and not v["TK id"])
    ]
)
top_esco_words = set(
    [word for word, v in Counter(esco_only_words.split(" ")).most_common(100)]
)

tk_only_words = " ".join(
    [
        v[look_at_col].lower()
        for k, v in joined_skills.items()
        if (v["TK id"] and not v["ESCO id"])
    ]
)
top_tk_words = set(
    [word for word, v in Counter(tk_only_words.split(" ")).most_common(100)]
)


# %%
print(top_esco_words.difference(top_tk_words))

# %%
print(top_tk_words.difference(top_esco_words))

# %% [markdown]
# ## Save merged skills data

# %%
# Save skills
pd.DataFrame(joined_skills).T.to_csv("outputs/skills_extraction/data/merged_skills.csv")

# %%
