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
#     display_name: skills-taxonomy-v2
#     language: python
#     name: skills-taxonomy-v2
# ---

# %%
import json
import random
from collections import Counter
import re

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from transformers import BertTokenizer, BertModel
import torch
import tqdm as tqdm
import pandas as pd
import numpy as np
import spacy
import pickle
import re
from xgboost import XGBClassifier
import nlpaug.augmenter.word as naw
from nltk.corpus import stopwords
import nltk
from itertools import zip_longest
import matplotlib.pyplot as plt  
from sklearn import metrics

# %% [markdown]
# ## import liz's pipeline

# %%
from skills_taxonomy_v2.pipeline.sentence_classifier.create_training_data import *
from skills_taxonomy_v2.pipeline.sentence_classifier.predict_sentence_class import *
from skills_taxonomy_v2.pipeline.sentence_classifier.sentence_classifier import *

# %% [markdown]
# ## import data

# %%
#both karlis and label-studio

with open('/inputs/new_training_data/training_data.pickle', 'rb') as fp:
    training_data = pickle.load(fp)

len(training_data)

# %%
#just label-studio

with open('/inputs/new_training_data/label_studio_training_data.pickle', 'rb') as fp:
    label_training_data = pickle.load(fp)

len(label_training_data)


# %% [markdown]
# ## preprocess sentence data

# %%
# remove numbers, symbols, lowercase, strip trailing white space

def preprocess_training_data(training):
    return [(re.sub(r'\d+[^\w]', ' ', string[0]).lower().strip(), string[1]) for string in training]


# %% [markdown]
# ## Experiment No. 1 - test liz's pipeline

# %%
#print results 

sent_class = SentenceClassifier(
    split_random_seed=1,
    test_size=0.1,
    log_reg_max_iter=1000,
    bert_model_name='paraphrase-MiniLM-L6-v2',
    multi_process=True,
)

X_train, X_test, y_train, y_test = sent_class.split_data(
    training_data, verbose=True
)

X_train_vec = sent_class.fit_transform(X_train)
sent_class.fit(X_train_vec, y_train)

# Training evaluation
y_train_pred = sent_class.predict(X_train_vec)
class_rep_train = sent_class.evaluate(y_train, y_train_pred, verbose=True)

# Test evaluation
y_test_pred = sent_class.predict_transform(X_test)
class_rep_test = sent_class.evaluate(y_test, y_test_pred, verbose=True)

# %% [markdown]
# ## Experiment No. 2 - test liz's pipeline on label studio data alone 

# %%
#print results 

sent_class = SentenceClassifier(
    split_random_seed=1,
    test_size=0.1,
    log_reg_max_iter=1000,
    bert_model_name='paraphrase-MiniLM-L6-v2',
    multi_process=True,
)

X_train, X_test, y_train, y_test = sent_class.split_data(
    label_training_data, verbose=True
)

X_train_vec = sent_class.fit_transform(X_train)
sent_class.fit(X_train_vec, y_train)

# Training evaluation
y_train_pred = sent_class.predict(X_train_vec)
class_rep_train = sent_class.evaluate(y_train, y_train_pred, verbose=True)

# Test evaluation
y_test_pred = sent_class.predict_transform(X_test)
class_rep_test = sent_class.evaluate(y_test, y_test_pred, verbose=True)

# %% [markdown]
# ## Experiment No. 3 - preprocess text + all training data

# %%
clean_training = preprocess_training_data(training_data)

sent_class = SentenceClassifier(
    split_random_seed=1,
    test_size=0.1,
    log_reg_max_iter=1000,
    bert_model_name='paraphrase-MiniLM-L6-v2',
    multi_process=True,
)

X_train, X_test, y_train, y_test = sent_class.split_data(
    clean_training, verbose=True
)

X_train_vec = sent_class.fit_transform(X_train)
sent_class.fit(X_train_vec, y_train)

# Training evaluation
y_train_pred = sent_class.predict(X_train_vec)
class_rep_train = sent_class.evaluate(y_train, y_train_pred, verbose=True)

# Test evaluation
y_test_pred = sent_class.predict_transform(X_test)
class_rep_test = sent_class.evaluate(y_test, y_test_pred, verbose=True)

# %% [markdown]
# ## Experiment No. 4 - use XGBOOST 

# %%
#vectorise test data 
X_test_vec = sent_class.fit_transform(X_test)

# run on xgboost

xgb = XGBClassifier(max_depth= 7, min_child_weight= 1)
xgb.fit(X_train_vec, y_train)
predict = xgb.predict(X_test_vec)
print(classification_report(y_test, predict))

# %% [markdown]
# ## Experiment No. 5 - Balance training data - undersample 0 class

# %%
# get all skill labels in training set from clean_training
skills = [(train, label) for train, label in zip(X_train, y_train) if label == 1]

# randomly sample non skill sentences 
no_skill_undersample = random.sample([(train, label) for train, label in zip(X_train, y_train) if label == 0], len(skills))

# create new balanced training set 
balanced_training = no_skill_undersample + skills

X_train_undersample = [x[0] for x in balanced_training]
y_train_undersample = [x[1] for x in balanced_training]

print(f'balanced training sentences is now {Counter(y_train_undersample)}')
print(f'test set is {Counter(y_test)}')

# %%
# vectorise training data
X_train_vec = sent_class.fit_transform(X_train_undersample)

# %%
# run balanced data on xgboost

xgb = XGBClassifier(max_depth= 7, min_child_weight= 1)
xgb.fit(X_train_vec, y_train_undersample)
predict = xgb.predict(X_test_vec)
print(classification_report(y_test, predict))


# %% [markdown]
# ## Experiment No. 6 - Balance training data - use nlpaug word synonyms to oversample 1 class

# %%
#generate augmented skills sentences with wordnet + balance classes
# oversample 1 class

def oversample_skills_wordnet(skills_data):
    aug = naw.SynonymAug(aug_src='wordnet')
    augmented_skills = [(aug.augment(train[0]), train[1]) for train in skills_data]
    return skills_data + augmented_skills


# %%
#make training data
skills_augment = oversample_skills_wordnet(skills)
balanced_augment_training = skills_augment + [(train, label) for train, label in zip(X_train, y_train) if label == 0]

X_train_oversample_syns = [x[0] for x in balanced_augment_training]
y_train_oversample_syns = [x[1] for x in balanced_augment_training]

print(f'balanced training sentences is now {Counter(y_train_oversample_syns)}')
print(len(y_train_oversample_syns))
print(f'test set is {Counter(y_test)}')
print(len(y_test))

# %%
# vectorise new training
X_train_vec = sent_class.fit_transform(X_train_oversample_syns)

# %%
# run balanced data on xgboost

xgb = XGBClassifier(max_depth= 7, min_child_weight= 1)
xgb.fit(X_train_vec, y_train_oversample_syns)
predict = xgb.predict(X_test_vec)
print(classification_report(y_test, predict))


# %% [markdown]
# ## Experiment No. 7 - Balance training data - use contextual word embeddings to oversample 1 class

# %%
#generate augmented skills sentences with wordnet + balance classes

def oversample_skills_embeds(skills_data):
    stops = stopwords.words('english')
    aug = naw.ContextualWordEmbsAug(aug_min = 1, stopwords = stops)
    augmented_embed_skill_sents = []
    for index, train in enumerate(skills):
        augment_word_embeds = aug.augment(train[0])
        print(f'augmented {index} sentence!')
        augmented_embed_skill_sents.append((augment_word_embeds, train[1]))
    return skills_data + augmented_embed_skill_sents



# %%
#make training data - word embeds 

skills_augment_embed = oversample_skills_embeds(skills)
balanced_augment_embed = skills_augment_embed + [(train, label) for train, label in zip(X_train, y_train) if label == 0]

X_train_oversample_embeds = [x[0] for x in balanced_augment_embed]
y_train_oversample_embeds = [x[1] for x in balanced_augment_embed]

print(f'balanced training sentences is now {Counter(y_train_oversample_embeds)}')
print(len(y_train_oversample_embeds))

print(f'test set is {Counter(y_test)}')
print(len(y_test))

# %%
# vectorise new training
X_train_vec = sent_class.fit_transform(X_train_oversample_embeds)

# %%
# run balanced data on xgboost

xgb = XGBClassifier(max_depth= 7, min_child_weight= 1)
xgb.fit(X_train_vec, y_train_oversample_embeds)
predict = xgb.predict(X_test_vec)
print(classification_report(y_test, predict))

# %% [markdown]
# ## Experiment No. 8 - Balance training data - oversample 1 class (word synonyms) + under sample 0 class

# %%
oversample_undersample_training_syns = skills_augment + random.sample([(train, label) for train, label in zip(X_train, y_train) if label == 0], 
                                     len(skills_augment))

X_train_overunder_syns = [x[0] for x in oversample_undersample_training_syns]
y_train_overunder_syns = [x[1] for x in oversample_undersample_training_syns]

print(f'balanced training sentences is now {Counter(y_train_overunder_syns)}')
print(len(y_train_overunder_syns))
print(f'test set is {Counter(y_test)}')
print(len(y_test))

X_train_vec = sent_class.fit_transform(X_train_overunder_syns)

# run balanced data on xgboost

xgb = XGBClassifier(max_depth= 7, min_child_weight= 1)
xgb.fit(X_train_vec, y_train_overunder_syns)
predict = xgb.predict(X_test_vec)
print(classification_report(y_test, predict))

# %% [markdown]
# ## Experiment No. 9 - Balance training data - oversample 1 class (word embeds) + under sample 0 class

# %%
oversample_undersample_training_embeds = skills_augment_embed + random.sample([(train, label) for train, label in zip(X_train, y_train) if label == 0], 
                                     len(skills_augment_embed))
X_train_overunder_embeds = [x[0] for x in oversample_undersample_training_embeds]
y_train_overunder_embeds = [x[1] for x in oversample_undersample_training_embeds]

print(f'balanced training sentences is now {Counter(y_train_overunder_embeds)}')
print(len(y_train_overunder_embeds))
print(f'test set is {Counter(y_test)}')
print(len(y_test))

X_train_vec = sent_class.fit_transform(X_train_overunder_embeds)

# run balanced data on xgboost
xgb = XGBClassifier(max_depth= 7, min_child_weight= 1)
xgb.fit(X_train_vec, y_train_overunder_embeds)
predict = xgb.predict(X_test_vec)
print(classification_report(y_test, predict))


# %% [markdown]
# ## Experiment No. 10 - use one hot encoding of verb positionality w/o balancing

# %%
def count_verbs(text_data, max_len = 600):
    verb_pos = []
    for text in text_data:
        pos_tags = nltk.pos_tag(nltk.word_tokenize(text))
        verb_pos.append([tag[1].count('VBG') for tag in pos_tags])
    
    # to array 
    verb_array = np.array(list(zip_longest(*verb_pos, fillvalue=0))).T
    
    return np.array([np.pad(x, (0, max_len - len(x)), 'constant') for x in verb_array])


# %%
# stack verb counts 
train_verb_stack = np.hstack((X_train_vec, count_verbs(X_train)))
test_verb_stack = np.hstack((X_test_vec, count_verbs(X_test)))

# %%
# run xgboost

xgb = XGBClassifier(max_depth= 7, min_child_weight= 1)
xgb.fit(train_verb_stack, y_train)
predict = xgb.predict(test_verb_stack)
print(classification_report(y_test, predict))

# %% [markdown]
# ## Experiment No. 11 - adjust probability threshold to 0.4

# %%
#print false positives

predicted_probabilities = xgb.predict_proba(test_verb_stack)
for sentence, probability, test_label, predicted_label in zip(X_test, predicted_probabilities, y_test, predict):
    if test_label == 0 and predicted_label == 1:
        print(sentence)

# %%
# look into edge case sentences

edge_cases = []
for index, probabilities in enumerate(predicted_probabilities):
    if 0.45 < probabilities[0] < 0.65:
        edge_cases.append((index, probabilities[0], probabilities[1]))

for i in edge_cases:
    print(f'the sent is: {X_test[i[0]]}')
    print(f'the predicted label is: {predict[i[0]]}')
    print(f'the true label is: {y_test[i[0]]}')
    print(f'the probability is: {str(round(i[1], 3))}')

# %%
# adjust threshold

y_pred_adjusted = []
for prob in predicted_probabilities:
    y_pred_adjusted.append(int(np.where(prob[1] > 0.40, 1, 0)))
    
print(classification_report(y_test, y_pred_adjusted))

# %% [markdown]
# ## Experiment No. 12 - adjust probability threshold to 0.7

# %%
# adjust threshold

y_pred_adjusted = []
for prob in predicted_probabilities:
    y_pred_adjusted.append(int(np.where(prob[1] > 0.60, 1, 0)))
    
print(classification_report(y_test, y_pred_adjusted))

# %%
# plot precision_recall

metrics.plot_precision_recall_curve(xgb, test_verb_stack, y_test)

# %%
#plot ROC curve

metrics.plot_roc_curve(xgb, test_verb_stack, y_test)

# %% [markdown]
# ## Stochasticity

# %%
# running latest pipeline with different seeds

random_seeds = [4, 22, 235, 42, 55]
different_splits = {}
for seed in random_seeds:
    sent_class = SentenceClassifier(
    split_random_seed=seed,
    test_size=0.1,
    log_reg_max_iter=1000,
    bert_model_name='paraphrase-MiniLM-L6-v2',
    multi_process=True)
    
    X_train, X_test, y_train, y_test = sent_class.split_data(clean_training, verbose=True)
    
    different_splits[f'seed_{seed}'] = {}
    different_splits[f'seed_{seed}']['X_train'] = X_train
    different_splits[f'seed_{seed}']['X_test'] = X_test
    different_splits[f'seed_{seed}']['y_train'] = y_train
    different_splits[f'seed_{seed}']['y_test'] = y_test


# %%
def current_pipeline(X_train, y_train, X_test, y_test):
    #vectorise 
    X_train_vec = sent_class.fit_transform(X_train)
    X_test_vec = sent_class.fit_transform(X_test)
    #stack 
    train_verb_stack = np.hstack((X_train_vec, count_verbs(X_train)))
    test_verb_stack = np.hstack((X_test_vec, count_verbs(X_test)))
    #classify
    xgb = XGBClassifier(max_depth= 7, min_child_weight= 1)
    xgb.fit(train_verb_stack, y_train)
    predict = xgb.predict(test_verb_stack)
    print(classification_report(y_test, predict))
    
    return X_train_vec, X_test_vec, train_verb_stack, test_verb_stack, xgb, predict


# %%
for key, value in different_splits.items():
    print(f'model trained on {key}')
    X_train_vec, X_test_vec, train_verb_stack, test_verb_stack, xgb, predict = current_pipeline(different_splits[key]['X_train'], different_splits[key]['y_train'], 
                     different_splits[key]['X_test'], different_splits[key]['y_test'])
    print('------------')

# %%
#best random seed + experiment based on maximising precision 
#print false positive sentences

X_train_vec, X_test_vec, train_verb_stack, test_verb_stack, xgb, predict = current_pipeline(different_splits['seed_22']['X_train'], different_splits['seed_22']['y_train'],
                 different_splits['seed_22']['X_test'], different_splits['seed_22']['y_test'])

predicted_probabilities = xgb.predict_proba(test_verb_stack)
for sentence, probability, test_label, predicted_label in zip(different_splits['seed_22']['X_test'], predicted_probabilities, different_splits['seed_22']['y_test'], predict):
    if test_label == 0 and predicted_label == 1:
        print(sentence, '------',  probability[1])
