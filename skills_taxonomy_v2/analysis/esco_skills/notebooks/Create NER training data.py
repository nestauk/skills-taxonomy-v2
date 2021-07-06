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

# %%
import json
import re

from tqdm import tqdm

# %%
with open('../../../../inputs/TextKernel_sample/karlis_skill_extraction_jobs_new.1.jsonl', 'r') as file:
    esco_data = {}
    for line in file:
        line_dict = json.loads(line)
        for job_id, skills in line_dict.items():
            esco_data[job_id] = skills

# %%
with open('../../../../inputs/TextKernel_sample/jobs_new.1.jsonl', 'r') as file:
    tk_data = [json.loads(line) for line in file]


# %% [markdown]
# ## Get out some potential skill entities using the ESCO predictions
# ```
# [('he team enthusiastically, with excellent focus on customer service. The Ideal candidate should have previous experie',
#   {'entities': [(50, 66, 'SKILL')]}),
#  ('in a fast pace QSR environment with a can do work ethic, ability to take initiative on project , ability ',
#   {'entities': [(50, 55, 'SKILL')]}),
#   ...
#  ```
# - For ESCO surface forms found, search for them in the text and if they exist (sometimes they won't be there) then add the indexes to training data
# - Use this to train a NER model
#
# To try to mitigate falsely labelled entities:
# 1. Only label if the surface form exists once (multiple times means you had to pick the first time, which may be incorrect)
# 2. Only label if the surface form exists separated by blanks (otherwise it might not be a word, but a sub part of a word e.g. 'cat' in 'category'. Also spacy wouldnt like the fact the boundaries to the 'entity' weren't actually the entire entity.
#
# Problems:
# 1. Some windows of text may contain multiple skills, but only one will be labelled
# 2. This is a model trained on the results of another model - accumulation of assumptions!
# 3. The word labelled may not actually be the skill entity

# %%
def look_forward(entity_i, entity_cluster):
    if entity_i + 1 < len(single_surface_forms):
        current_end = single_surface_forms[entity_i][1]
        next_start = single_surface_forms[entity_i+1][0]
        if (next_start - current_end) < window_char_len:
            entity_cluster.append(entity_i+1)
            entity_cluster = look_forward(entity_i+1, entity_cluster)
    return entity_cluster


# %%
len(esco_data)

# %%
window_char_len = 50
first_n = 10000

# %%
training_data = []
surface_forms_set = set()
for job_id in tqdm(list(esco_data.keys())[0:first_n]):
    surface_forms = [e['surface_form'] for e in esco_data[job_id]]
    full_text = [d['full_text'] for d in tk_data if d['job_id']==job_id][0]
    
    single_surface_forms = []
    for s in surface_forms:
        # You can end up getting a multiple repeat error, so need to convert 'c++' -> 'c\+\+'
        s = s.replace('++', '\+\+')
        entities_list = []
        for match in re.finditer(
            re.compile(r'\s' + s + '[\s\,\.\;\-]', re.IGNORECASE),
            full_text
            ):
            entities_list.append((match.start()+1, match.end()-1))
        # Only add if one instance of the surface form was found    
        if len(entities_list) == 1:
            single_surface_forms.append(entities_list[0])

    single_surface_forms = sorted(single_surface_forms, key=lambda x: x[0])
    entity_clusters = []
    entity_i = 0
    while entity_i < len(single_surface_forms):
        entity_cluster = [entity_i]
        entity_cluster = look_forward(entity_i, entity_cluster)
        entity_clusters.append(entity_cluster)
        entity_i += len(entity_cluster)

    for entity_cluster in entity_clusters:
        starting_sf = single_surface_forms[entity_cluster[0]]
        last_sf = single_surface_forms[entity_cluster[-1]]
        shift_len = max(0, starting_sf[0] - window_char_len)
        window_text = full_text[shift_len: (last_sf[1] + window_char_len)]
        entities_list = []
        for entity in entity_cluster:
            sf = single_surface_forms[entity]
            entities_list.append((sf[0] - shift_len, sf[1] - shift_len, 'SKILL'))
        to_enter = (window_text, {'entities': entities_list})
        training_data.append(to_enter)

# %%
len(training_data)

# %%
with open(
    f'../../../../outputs/skills_ner/data/training_data_windowcharlen{window_char_len}_firstn{first_n}_jobs_new.1.jsonl',
    'w') as f:
    for data_point in training_data:
        f.write(json.dumps(data_point))
        f.write('\n')

# %%
