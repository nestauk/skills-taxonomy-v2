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
# I extracted ESCO skills using Karlis' algorithm on a sample of the TextKernel data (100,000 job adverts).
#
# The commands I used to run this are given [below](#section1).
#
# Look at this dataset and common skills.
#

# %%
import json
from collections import Counter

import matplotlib.pyplot as plt

# %%
with open('../../../../inputs/TextKernel_sample/karlis_skill_extraction_jobs_new.1.jsonl', 'r') as file:
    esco_data = {}
    for line in file:
        line_dict = json.loads(line)
        for job_id, skills in line_dict.items():
            esco_data[job_id] = skills

# %%
len(esco_data)

# %%
## How many skills per job
num_skills = [len(job_skills) for job_skills in esco_data.values()]

# %%
plt.figure(figsize=(15,3))
plt.hist(num_skills, 50, alpha = 0.7)
plt.title('Number of skills extracted per job')
plt.show()

# %%
all_skills = []
for skills in esco_data.values():
    for skill in skills:
        all_skills.append(skill['surface_form'])

# %%
Counter(all_skills).most_common(20)

# %% [markdown]
# ## Commands used to get sample data
# <a id='section1'></a>
# This isn't in a script, but the flow I ran (from the `ojd_daps` base folder) was:
# ```python
# import json
# import pandas as pd
# import tqdm as tqdm
# from ojd_daps.flows.enrich.labs.skills.skills_detection_utils import (
#   load_model,
#   setup_spacy_model,
#   detect_skills,
#   clean_text)
#
# model = load_model("v02")
# nlp = setup_spacy_model(model["nlp"])
#
# with open('../skills-taxonomy-v2/inputs/TextKernel_sample/jobs_new.1.jsonl', 'r') as file:
# 	data = [json.loads(line) for line in file]
#
# job_ads = pd.DataFrame(data)
#
# description_column = 'full_text' # May want to apply to others
# job_id_column = 'job_id'
#
# for job_id, description_text in tqdm.tqdm(list(job_ads[[job_id_column, description_column]].itertuples(index=False))):
#     dict_obj = {job_id: detect_skills(clean_text(description_text), model, nlp, return_dict=True)}
#     with open('../skills-taxonomy-v2/inputs/TextKernel_sample/karlis_skill_extraction_jobs_new.1.jsonl', 'a') as f:
#         f.write(json.dumps(dict_obj))
#         f.write('\n')
# ```

# %%
