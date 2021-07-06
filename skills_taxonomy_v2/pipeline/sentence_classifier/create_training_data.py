"""
Create training data for sentence classifier.
"""

import spacy
from tqdm import tqdm

import json
import re

def load_raw_data(jobs_data_file='inputs/karlis_ojo_manually_labelled/OJO_test_labelling_April2021_jobs.jsonl'):

    with open(jobs_data_file, 'r') as file:
        jobs_data = [json.loads(line) for line in file]

    return jobs_data

def mask_text(nlp, text, spacy_ner_types = ['DATE', 'MONEY', 'CARDINAL', 'TIME', 'ORDINAL', 'QUANTITY']):
    # Cleaning where you need to keep indices the same
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

def text_cleaning(text):
    # Cleaning where it doesnt matter if you mess up the indices
    text = re.sub(r'[#]+','NUMBER', text)
    text = re.sub('CNUMBER','C#', text) # Some situations you shouldn't have removed the numbers
    return text

def split_labelled_sentences(nlp, text, skills_annotations, skill_label_ids=[1, 5]):
    """
    Tag the sentences within one job advert text as having skills in (1) or not (0).
    - Get job advert text and corresponding annotations of skills within it
    - Clean text
    - Split up sentences and check if they contained any skill labels
    - Create list of sentences with corresponding list of tags
    """

    skill_spans = [(label['start_offset'], label['end_offset']) for label in skills_annotations if label['label'] in skill_label_ids]
    
    # Mask out numbers (good to do before sentence splitting since they can look like sentence boundaries) and remove *, but doesn't effect number of characters
    text = mask_text(nlp, text)
    
    # Split up sentences
    skill_span_sets = [set(range(s,e)) for s,e in skill_spans]
    doc = nlp(text)
    sentences = []
    sentences_label = []
    for sent in doc.sents:
        sentences.append(text_cleaning(sent.text))
        sentence_set = set(range(sent.start_char, sent.end_char))
        if any([entity_set.issubset(sentence_set) for entity_set in skill_span_sets]):
            sentences_label.append(1)
        else:
            sentences_label.append(0)

    return sentences, sentences_label

def create_training_data(nlp, jobs_data, sentence_train_threshold=15, skill_label_ids=[1, 5]):
    """
    Label all job adverts and add them all to a list of training data.
    sentence_train_threshold: A threshold of how big the sentence has to be in order to include it in the training/test data
    """

    # Create training dataset
    training_data = []
    for job_info in tqdm(jobs_data):
        text = job_info['text']
        skills_annotations = job_info['annotations']
        sentences, sentences_label = split_labelled_sentences(nlp, text, skills_annotations, skill_label_ids=skill_label_ids)
        for sentence, sentence_label in zip(sentences, sentences_label):
            if len(sentence) > sentence_train_threshold:
                training_data.append((sentence, sentence_label))

    return training_data

def save_training_data(training_data, output_file):
    with open(output_file, 'w') as file:
        for label in training_data:
            file.write(json.dumps(label))
            file.write('\n')

def load_training_data(input_file):
    with open(input_file, 'r') as file:
        training_data = [tuple(json.loads(line)) for line in file]
    return training_data

if __name__ == '__main__':
    
    jobs_data = load_raw_data(jobs_data_file='inputs/karlis_ojo_manually_labelled/OJO_test_labelling_April2021_jobs.jsonl')

    nlp = spacy.load("en_core_web_sm")

    training_data = create_training_data(nlp, jobs_data, sentence_train_threshold=15)

    save_training_data(training_data, 'outputs/sentence_classifier/data/training_data_April2021.json')
