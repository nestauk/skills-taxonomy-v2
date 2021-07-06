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
# !pip install nervaluate

# %%
import json
from collections import Counter
import re
import random
import warnings

import matplotlib.pyplot as plt
import spacy
from spacy.util import minibatch, compounding
from spacy.training import Example
from spacy import displacy
from nervaluate import Evaluator

# %%
# Parameters for training data file
window_char_len = 50
first_n = 10000

# %%
training_data = []
with open(
    f'../../../../outputs/skills_ner/data/training_data_windowcharlen{window_char_len}_firstn{first_n}_jobs_new.1.jsonl',
    'r') as f:
    for line in f:
        data = json.loads(line)
        training_data.append((data[0], data[1]))


# %%
len(training_data)

# %% [markdown]
# ## Train NER model

# %%
train_split_size = round(len(training_data)*0.8)
TRAIN_DATA = training_data[0:train_split_size]
TEST_DATA = training_data[train_split_size:]
print(len(TRAIN_DATA))
print(len(TEST_DATA))

# %%
nlp = spacy.blank("en")

# %%
if "ner" not in nlp.pipe_names:
    # If you are training on a blank model
    ner = nlp.add_pipe("ner")
else:
    ner = nlp.get_pipe("ner")

# add labels from data
for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# %% [markdown]
# ## Version 1

# %%
examples = []
dud_examples = []
for text, annotations in TRAIN_DATA:
    doc = nlp.make_doc(text)
    try:
        example = Example.from_dict(doc, annotations)
        examples.append(example)
    except:
        dud_examples.append((text, annotations))

# %%
print(len(examples))
print(len(dud_examples))

# %%
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.select_pipes(disable=other_pipes):  # only train NER
    optimizer = nlp.initialize(lambda: examples)

    for i in range(10):
        random.shuffle(TRAIN_DATA)
        losses = {}

        for example in examples:
            nlp.update(
                [example],
                sgd=optimizer,
                losses=losses,
                drop=0.5,
            )
        print(f"{i}: Losses {losses['ner']}")

# %% [markdown]
# ### Test model - v1
# version1: {'SKILL': 0.8662140905704113, 'Overall': 0.8662140905704113}
#
# - n_iter = 10
# - n_train_data = 20734

# %%
y_true = []
y_pred = []
for text, annotations in TEST_DATA:
    doc = nlp(text)
    pred_entities = []
    for ent in doc.ents:
        pred_entities.append(
            {"start": ent.start_char, "end": ent.end_char, "label": ent.label_}
        )
    y_true.append([{"start": ent[0], "end": ent[1], "label": ent[2]} for ent in annotations['entities']])
    y_pred.append(pred_entities)

# %%
tags = ['SKILL']
evaluator = Evaluator(y_true, y_pred, tags=tags)
results, results_by_tag = evaluator.evaluate()

score = {tag: results_by_tag[tag]["partial"]["f1"] for tag in tags}
score["Overall"] = results["partial"]["f1"]
score

# %% [markdown]
# ## Version 2

# %%
len(TRAIN_DATA)

# %%
# show warnings for misaligned entity spans once 
warnings.filterwarnings("once", category=UserWarning, module='spacy') 

# %%
mini_batch_size = 200
n_iter = 3
n_train_data = len(TRAIN_DATA)

# %%
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.select_pipes(disable=other_pipes), warnings.catch_warnings():  # only train NER
    warnings.simplefilter("ignore")
    optimizer = nlp.initialize()

    for iteration_n in range(n_iter):
        print(f'--- Iteration {iteration_n} - {round(n_train_data/mini_batch_size)} minibatches---')
        # Shuffle of the data
        random.shuffle(TRAIN_DATA)
        
        # Update on each batch
        losses = {}
        dud_examples = []
        batches = spacy.util.minibatch(TRAIN_DATA[0:n_train_data], size=mini_batch_size)
        for batch_i, batch in enumerate(batches):
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                try:
                    example = Example.from_dict(doc, annotations)
                    nlp.update(
                        [example],
                        sgd=optimizer,
                        drop=0.5,
                        losses=losses,         
                    )
                except:
                    dud_examples.append((text, annotations))
            if batch_i % 10 == 0:
                print(f"Minibatch {batch_i}: Losses {losses['ner']}")
        if iteration_n == 0:
            # These will be similar for each shuffle
            print('Number of dud examples')
            print(len(dud_examples))

# %% [markdown]
# ### Test model - v2
# version2 :{'SKILL': 0.8355133794990168, 'Overall': 0.8355133794990168}. 
#
# - mini_batch_size = 200
# - n_iter = 3
# - n_train_data = len(TRAIN_DATA) = 20734

# %%
y_true = []
y_pred = []
for text, annotations in TEST_DATA:
    doc = nlp(text)
    pred_entities = []
    for ent in doc.ents:
        pred_entities.append(
            {"start": ent.start_char, "end": ent.end_char, "label": ent.label_}
        )
    y_true.append([{"start": ent[0], "end": ent[1], "label": ent[2]} for ent in annotations['entities']])
    y_pred.append(pred_entities)

# %%
tags = ['SKILL']
evaluator = Evaluator(y_true, y_pred, tags=tags)
results, results_by_tag = evaluator.evaluate()

score = {tag: results_by_tag[tag]["partial"]["f1"] for tag in tags}
score["Overall"] = results["partial"]["f1"]
score

# %% [markdown]
# ## View some labels

# %%
text, annotations = TEST_DATA[5]
doc = nlp(text)
displacy.render(doc, style="ent")

# %%
