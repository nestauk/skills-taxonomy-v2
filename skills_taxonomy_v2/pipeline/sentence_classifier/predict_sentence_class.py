"""
Predict whether sentences are skill sentences or not using all the sentences for a file of job adverts
"""

import json
import time
import yaml
import pickle 

from tqdm import tqdm
import spacy

from skills_taxonomy_v2.pipeline.sentence_classifier.sentence_classifier import SentenceClassifier
from skills_taxonomy_v2.pipeline.sentence_classifier.create_training_data import mask_text, text_cleaning

if __name__ == '__main__':

	nlp = spacy.load("en_core_web_sm")

	with open('inputs/TextKernel_sample/jobs_new.1.jsonl', 'r') as file:
		data = [json.loads(line) for line in file]

	# Load sentence classifier trained model
	config_name = '2021.07.09.small'
	file_name = 'outputs/sentence_classifier/models/2021_07_09_small.pkl'

	fname = f"skills_taxonomy_v2/config/sentence_classifier/{config_name}.yaml"
	with open(fname, "r") as f:
		config = yaml.load(f, Loader=yaml.FullLoader)
		
	sentence_train_threshold = 15
	sentence_max_len = 100

	# Loading the model
	sent_class = SentenceClassifier(
		bert_model_name=config['flows']['sentence_classifier_flow']['params']['bert_model_name'],
		multi_process=config['flows']['sentence_classifier_flow']['params']['multi_process']
	)
	sent_class.load_model(file_name)

	# Split the sentences
	t0 = time.time()
	sentences = []
	job_ids = []
	for job_info in tqdm(data):
		job_id = job_info['job_id']
		text = job_info['full_text']
		text = mask_text(nlp, text)
		# Split up sentences
		doc = nlp(text)
		for sent in doc.sents:
			sentence = text_cleaning(sent.text) 
		if len(sentence) in range(sentence_train_threshold, sentence_max_len):
			sentences.append(sentence)
			job_ids.append(job_id)

	print(f"It took {time.time()-t0} seconds to split the job adverts into sentences")

	# Predict
	t0 = time.time()
	all_sentences_preds = []
	sentences_vec = sent_class.transform(sentences)
	sentences_pred = sent_class.predict(sentences_vec)
	print(f"It took {time.time()-t0} seconds to predict skill sentences")

	# Save

	# Dump data to file
	with open(f"outputs/sentence_classifier/data/skill_sentences/{config_name}_jobsnew1_embeddings.pkl", 'wb') as file:
		pickle.dump(sentences_vec, file)
	with open(f"outputs/sentence_classifier/data/skill_sentences/{config_name}_jobsnew1_predictions.pkl", 'wb') as file:
		pickle.dump(sentences_pred, file)
	with open(f"outputs/sentence_classifier/data/skill_sentences/{config_name}_jobsnew1_sentences.pkl", "wb") as file:
		pickle.dump(sentences, file)
	with open(f"outputs/sentence_classifier/data/skill_sentences/{config_name}_jobsnew1_job_ids.pkl", "wb") as file:
		pickle.dump(job_ids, file)

	# Load
	# with open(f"outputs/sentence_classifier/data/skill_sentences/{config_name}_jobsnew1_sentences.pkl", "rb") as file:
	# 	sentences2 = pickle.load(file)

