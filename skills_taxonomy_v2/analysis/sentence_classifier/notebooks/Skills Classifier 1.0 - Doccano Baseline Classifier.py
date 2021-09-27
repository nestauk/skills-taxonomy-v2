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

# # Existing skill tags data
# 1. Look at data
# 2. Build a simple baseline classifier
#
# Karlis tagged 50 jobs with where the skills were mentioned. Can we train something to identify sentences as about skills or not?
#
# Would be helpful for taking out the junk.

# +
from sklearn.linear_model import LogisticRegression
import json
import random

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
# -

# ### Import data

with open('../../../../inputs/karlis_ojo_manually_labelled/OJO_test_labelling_April2021_jobs.jsonl', 'r') as file:
    jobs_data = [json.loads(line) for line in file]

jobs_data[0].keys()

with open('../../../../inputs/karlis_ojo_manually_labelled/OJO_test_labelling_April2021_labels.json', 'r') as file:
    labels_data = json.load(file)

label_type_dict = {label_type['id']: label_type['text']
                   for label_type in labels_data}
label_type_dict

# ### Restructuring to have a look

# +
all_job_tags_text = {}

for job_id, job_info in enumerate(jobs_data):
    text = job_info['text']
    annotations = job_info['annotations']
    job_tags_text = {}
    for label_number, label_type in label_type_dict.items():
        job_tags_text[label_type] = [text[label['start_offset']:label['end_offset']]
                                     for label in annotations if label['label'] == label_number]
    all_job_tags_text[job_id] = job_tags_text
# -

job_id = 1
print(jobs_data[job_id]['text'])
print("\n")
print(all_job_tags_text[job_id]['SKILL'])
print(all_job_tags_text[job_id]['SKILL-RELATED'])


# ## Create a basic classifier
# Label sentences with containing skills (1) or not (0)
#
# Method assumes sentences are split by full stop and will run into problems if the skill has a full stop in.

def label_sentences(job_id):
    annotations = jobs_data[job_id]['annotations']
    skill_spans = [(label['start_offset'], label['end_offset'])
                   for label in annotations if label['label'] in [1, 5]]
    sentences = jobs_data[job_id]['text'].split('.')

    # Indices of where sentences start and end
    sentences_ix = []
    for i, sentence in enumerate(sentences):
        if i == 0:
            start = 0
        else:
            start = sentences_ix[i - 1][1] + 1
        sentences_ix.append((start, start + len(sentence)))

    # Find which sentences contain skills
    sentences_label = [0] * len(sentences)
    for (skill_start, skill_end) in skill_spans:
        for i, (sent_s, sent_e) in enumerate(sentences_ix):
            if sent_s <= skill_start and sent_e >= skill_end:
                sentences_label[i] = 1

    return sentences, sentences_label


# Testing
job_id = 2
sentences, sentences_label = label_sentences(job_id)
print(all_job_tags_text[job_id]['SKILL'])
print(all_job_tags_text[job_id]['SKILL-RELATED'])
print([sentences[i] for i, label in enumerate(sentences_label) if label == 1])
print([sentences[i] for i, label in enumerate(sentences_label) if label == 0])

# Create training dataset
X = []
y = []
for job_id in range(len(jobs_data)):
    sentences, sentences_label = label_sentences(job_id)
    for sentence, sentence_label in zip(sentences, sentences_label):
        X.append(sentence)
        y.append(sentence_label)

# +
# Random shuffle data points
shuffle_index = list(range(len(X)))
random.Random(42).shuffle(shuffle_index)

X = [X[i] for i in shuffle_index]
y = [y[i] for i in shuffle_index]

# Split test/train set
train_split = 0.75
len_train = round(len(X) * train_split)
X_train = X[0:len_train]
y_train = y[0:len_train]
X_test = X[len_train:]
y_test = y[len_train:]
# -

print(len(X))
print(len(y_train))
print(len(y_test))

vectorizer = CountVectorizer(
    analyzer="word",
    token_pattern=r"(?u)\b\w+\b",
    ngram_range=(1, 2),
    stop_words='english'
)
X_train_vect = vectorizer.fit_transform(X_train)

model = MultinomialNB()
model = model.fit(X_train_vect, y_train)

X_test_vect = vectorizer.transform(X_test)
y_test_pred = model.predict(X_test_vect)

print(classification_report(y_test, y_test_pred))

# +
# LogisticRegression

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model = model.fit(X_train_vect, y_train)

X_test_vect = vectorizer.transform(X_test)
y_test_pred = model.predict(X_test_vect)

print(classification_report(y_test, y_test_pred))
# -
