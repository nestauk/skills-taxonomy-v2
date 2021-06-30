# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Classifying sentences as containing skills or not
# #### Improving the baseline classifier:
# - Use BERT
# - Split sentences using spacy sentence splitter
# - Split on camel case
# - More training data
# - Clean text
#
# #### Analysis:
# - anything odd about the ones it misclassifies? (short?)
# - using 'skill' and 'skill related' tags - should I just use 'skill'?

# +
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
# -

# Download some models
nlp = spacy.load("en_core_web_sm")
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained("bert-base-uncased")

# ## Import data

with open('../../../../inputs/karlis_ojo_manually_labelled/OJO_test_labelling_April2021_jobs.jsonl', 'r') as file:
    jobs_data = [json.loads(line) for line in file]

with open('../../../../inputs/karlis_ojo_manually_labelled/OJO_test_labelling_April2021_labels.json', 'r') as file:
    labels_data = json.load(file)

label_type_dict = {label_type['id']: label_type['text'] for label_type in labels_data}
label_type_dict

with open('../../../../outputs/sentence_classifier/data/generated_training_data/TextKernel_no_skills_sections.json', 'r') as file:
    additional_no_skills = json.load(file)

with open('../../../../outputs/sentence_classifier/data/generated_training_data/TextKernel_skills_texts.json', 'r') as file:
    additional_skills = json.load(file)

# ## Clean sentences functions
# - Mask numbers using spacy NER
# - Remove camel case abcxXxdef -> abcx. Xxdef
# - Separate sentences

# Pattern for fixing a missing space between enumerations, for split_sentences()
compiled_missing_space_pattern = re.compile("([a-z])([A-Z])([a-z])")


def mask_job(text, spacy_ner_types = ['DATE', 'MONEY', 'CARDINAL', 'TIME', 'ORDINAL', 'QUANTITY']):
    doc = nlp(text)
    # '6 months' -> '######'
    # Go through detected entities backwards and replace in the sentence
    # (if you did it it going forwards the indexes would be incorrect)
    # Need to mask with the same length of entity so indexes of skills are still valid
    for entity in reversed(doc.ents):
        if entity.label_ in spacy_ner_types:
            text = text[:entity.start_char] + '#'*len(str(entity)) + text[entity.end_char:]
            
    # Replace any * with comma
    text = text.replace('*', ',')
    text = text.replace('•', ',')
    text = text.replace('-', ',')
    return text


def remove_camelcase(text, skill_spans):
    # Find camel case, replace, and change skills_spans indexes accordingly

    # Update skill_spans indices
    skill_spans = sorted(skill_spans, reverse=True)
    # Where does the camel case change things?
    change_points = [instance.start()+1 for instance in re.finditer(compiled_missing_space_pattern, text)]
    for span_i, (span_s, span_e) in enumerate(skill_spans):
        # How many change points happen before this span?
        # we have added 2 characters- a '.' and a space
        num_changes_before_s = 2*sum([span_s>=c for c in change_points])
        num_changes_before_e = 2*sum([span_e>=c for c in change_points])
        # Update skill span indices
        skill_spans[span_i] = (span_s+num_changes_before_s, span_e+num_changes_before_e)

    # Clean text of camel case
    text = compiled_missing_space_pattern.sub(r"\1. \2\3", text)

    return text, skill_spans


def label_sentences(job_id):
    annotations = jobs_data[job_id]['annotations']
    skill_spans = [(label['start_offset'], label['end_offset']) for label in annotations if label['label'] in [1, 5]]
    
    # Mask out numbers and remove *, but doesn't effect number of characters
    text = mask_job(jobs_data[job_id]['text'])
    # Remove camel case and update skill_spans accordingly
#     text, skill_spans = remove_camelcase(text, skill_spans)
    
    # Split up sentences
    skill_span_sets = [set(range(s,e)) for s,e in skill_spans]
    doc = nlp(text)
    sentences = []
    sentences_label = []
    for sent in doc.sents:
        sentences.append(sent.text)
        sentence_set = set(range(sent.start_char, sent.end_char))
        if any([entity_set.issubset(sentence_set) for entity_set in skill_span_sets]):
            sentences_label.append(1)
        else:
            sentences_label.append(0)

    return sentences, sentences_label


# +
# def label_sentences(job_id):
#     annotations = jobs_data[job_id]['annotations']
#     skill_spans = [(label['start_offset'], label['end_offset']) for label in annotations if label['label'] in [1, 5]]
#     # Text cleaning, but doesn't effect number of characters
#     text = mask_job(jobs_data[job_id]['text'])
#     # Split up sentences
#     sentences = text.split('.')

#     # Indices of where sentences start and end
#     sentences_ix = []
#     for i, sentence in enumerate(sentences):
#         if i==0:
#             start = 0
#         else:
#             start = sentences_ix[i-1][1]+1
#         sentences_ix.append((start, start + len(sentence)))

#     # Find which sentences contain skills
#     sentences_label = [0]*len(sentences)
#     for (skill_start, skill_end) in skill_spans:
#         for i, (sent_s, sent_e) in enumerate(sentences_ix):
#             if sent_s <= skill_start and sent_e >= skill_end:
#                 sentences_label[i] = 1
                
#     return sentences, sentences_label
# -

def sentence_cleaning(sentence):
    # Cleaning where it doesnt matter if you mess up the indices
    sentence = re.sub(r'[#]+','NUMBER', sentence)
    return sentence


# ## Create training data
# Label sentences with containing skills (1) or not (0)

# A threshold of how big the sentence has to be in order to include it in the training/test data
sentence_train_threshold = 15

# Create training dataset
X = []
y = []
for job_id in range(len(jobs_data)):
    sentences, sentences_label = label_sentences(job_id)
    for sentence, sentence_label in zip(sentences, sentences_label):
        if len(sentence) > sentence_train_threshold:
            X.append(sentence_cleaning(sentence))
            y.append(sentence_label)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

print(len(X))
print(len(y_train))
print(len(y_test))
print(Counter(y_train))
print(Counter(y_test))


# ### Add some additional data from TextKernel to the training set

def add_tk_data(additional_data, X_train, y_train, sentence_train_threshold, label):
    for text in additional_data:
        text = mask_job(text)
        # Split up sentences
        doc = nlp(text)
        sentences = []
        sentences_label = []
        for sent in doc.sents:
            sentences.append(sent.text)
            sentences_label.append(label)

        for sentence, sentence_label in zip(sentences, sentences_label):
            if len(sentence) > sentence_train_threshold:
                X_train.append(sentence_cleaning(sentence))
                y_train.append(sentence_label)
                
    return X_train, y_train


# +
# Add additional no-skill and skill data
sample_no_skills_seed = 1

random.seed(sample_no_skills_seed)
additional_no_skills_sample = random.sample(additional_no_skills, 50)

X_train, y_train = add_tk_data(
    additional_no_skills_sample, X_train, y_train, sentence_train_threshold, label=0)

random.seed(sample_no_skills_seed)
additional_skills_sample = random.sample(additional_skills, 200)

X_train, y_train = add_tk_data(
    additional_skills_sample, X_train, y_train, sentence_train_threshold,  label=1)
# -

print(len(X))
print(len(y_train))
print(len(y_test))
print(Counter(y_train))
print(Counter(y_test))

plt.figure(figsize=(15,3))
plt.hist([len(x) for i, x in enumerate(X) if y[i]==0], 30, color='red', alpha = 0.7, density=True)
plt.hist([len(x) for i, x in enumerate(X) if y[i]==1], 30, color='green', alpha = 0.5, density=True)
plt.title('Number of characters in skill (green) and not skill (red) sentences')
plt.show()


# ## Train

def get_embedding(x, layer_type='last_hidden_state'):
    """
    layer_type: which layer to output, 'last_hidden_state' or 'pooler_output'
    """
    encoded_input = bert_tokenizer.encode(x, return_tensors="pt")
    encoded_input = encoded_input[:,:510]# could do something better?
    output = bert_model(encoded_input)
    embedded_x = output[layer_type]    
    if layer_type == 'last_hidden_state':
        embedded_x = embedded_x.mean(dim=1)
    return embedded_x.detach().numpy().flatten()


need_cut = [i for i,x in enumerate(X_train) if (bert_tokenizer.encode(x, return_tensors="pt")).shape[1]>=510]
print(f'{len(need_cut)} out of {len(X_train)} of the training data points needed cutting down so will lose information')

X_train_vect = [get_embedding(x) for x in tqdm.tqdm(X_train)]

# Fit scaler
scaler = MinMaxScaler()
X_train_vect_scaled = scaler.fit_transform(X_train_vect)

# Fit classifier
classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
classifier = classifier.fit(X_train_vect_scaled, y_train)

y_train_pred = classifier.predict(X_train_vect_scaled)
print(classification_report(y_train, y_train_pred))

confusion_matrix(y_train, y_train_pred)

# ## Predict

X_test_vect = [get_embedding(x) for x in tqdm.tqdm(X_test)]

need_cut = [i for i,x in enumerate(X_test) if (bert_tokenizer.encode(x, return_tensors="pt")).shape[1]>=510]
print(f'{len(need_cut)} out of {len(X_test)} of the test data points needed cutting down so will lose information')

X_test_vect_scaled = scaler.transform(X_test_vect)
y_test_pred = classifier.predict(X_test_vect_scaled)
print(classification_report(y_test, y_test_pred))

confusion_matrix(y_test, y_test_pred)

# ## Look at incorrect tags

wrong_preds = [i for i, (a, p) in enumerate(zip(y_test, y_test_pred)) if a!=p]
correct_preds = [i for i, (a, p) in enumerate(zip(y_test, y_test_pred)) if a==p]

wrong_preds_len = [len(X_test[i]) for i in wrong_preds]
correct_preds_len = [len(X_test[i]) for i in correct_preds]

np.mean(wrong_preds_len)

np.mean(correct_preds_len)

plt.figure(figsize=(15,3))
plt.hist(wrong_preds_len, 10, color='red', alpha = 0.7, density=True)
plt.hist(correct_preds_len, 30, color='green', alpha = 0.5, density=True)
plt.title('Number of characters in sentences which had wrong predictions')
plt.show()

sum(['NUMBER' in X_test[i] for i in wrong_preds])/len(wrong_preds)

sum(['NUMBER' in X_test[i] for i in correct_preds])/len(correct_preds)

len(wrong_preds)

i = wrong_preds[4]
print(X_test[i])
print(y_test[i])
print(y_test_pred[i])

df = pd.DataFrame({
    'Sentence':X_test ,
    'Actual': y_test,
    'Predicted': y_test_pred,
    'Correct prediction?': ['True' if a==p else 'False' for a, p in zip(y_test, y_test_pred)]
})
df[df['Correct prediction?']=='False']

df.to_csv('../data/bert_logreg_test.csv')


# ## Try out different classifiers

def use_model(X_train_vect, y_train, X_test_vect, y_test, classfier_type, use_scaler=None):
    if use_scaler:
        # Fit scaler
        scaler = MinMaxScaler()
        X_train_vect = scaler.fit_transform(X_train_vect)
        X_test_vect = scaler.transform(X_test_vect)
    
    # Fit classifier
    if classfier_type=='mnb':
        classifier = MultinomialNB()
    elif classfier_type=='svm':
        classifier = SVC(probability=True)
    elif classfier_type=='log_reg':
        classifier = LogisticRegression(max_iter=1000)
    else:
        print('Not valid classifier type')
    classifier = classifier.fit(X_train_vect, y_train)

    # Training scores
    y_train_pred = classifier.predict(X_train_vect)

    # Test scores
    y_test_pred = classifier.predict(X_test_vect)
    
    return {
        'Train F1': f1_score(y_train, y_train_pred),
        'Train precision': precision_score(y_train, y_train_pred),
        'Train recall': recall_score(y_train, y_train_pred),
        'Test F1': f1_score(y_test, y_test_pred),
        'Test precision': precision_score(y_test, y_test_pred),
        'Test recall': recall_score(y_test, y_test_pred),
    }


all_results = {}
for classfier_type in ['mnb', 'svm', 'log_reg']:
    for use_scaler in [True, None]:
        if classfier_type=='mnb':
            use_scaler = True
        all_results[f"{classfier_type}_{str(use_scaler)}"] = use_model(X_train_vect, y_train, X_test_vect, y_test, classfier_type=classfier_type, use_scaler=use_scaler)

pd.DataFrame(all_results)

# +
# No class weighting, using last_hidden_state
#pd.DataFrame(all_results)
