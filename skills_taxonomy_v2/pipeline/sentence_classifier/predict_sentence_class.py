"""
Predict whether sentences are skill sentences or not using all the sentences for a file of job adverts.

If you are running this on a whole directory of files it will try to read any jsonl or jsonl.gz in the location,
it will also skip over a file if a 'full_text' key isn't found in the first element of it.
"""

import json
import time
import yaml
import pickle 
import os
from fnmatch import fnmatch
import gzip
import logging
from argparse import ArgumentParser

from tqdm import tqdm
import spacy
import boto3

from skills_taxonomy_v2.pipeline.sentence_classifier.sentence_classifier import SentenceClassifier
from skills_taxonomy_v2.pipeline.sentence_classifier.create_training_data import mask_text, text_cleaning

BUCKET_NAME = 'skills-taxonomy-v2'

logger = logging.getLogger(__name__)


def load_neccessary(line):
	neccessary_fields = ['full_text', 'job_id']
	line = json.loads(line)
	return {field: line.get(field, None) for field in neccessary_fields}

def load_local_data(file_name):
	"""
	Local locations - jsonl or jsonl.gz
	Just load what's needed (job id and full text)
	"""
	if fnmatch(file_name, "*.jsonl.gz"):
		data = []
		with gzip.open(file_name) as f:
			for line in f:
				data.append(load_neccessary(line))
	elif fnmatch(file_name, "*.jsonl"):
		with open(file_name, 'r') as file:
			data = [load_neccessary(line) for line in file]
	else:
		raise TypeError('Input file type not recognised')

	if not data[0].get('full_text'):
		raise ValueError('The loaded data has no full text field')

	return data

def load_s3_data(file_name, s3):
	"""
	Load from S3 locations - jsonl.gz
	file_name: S3 key
	"""
	if fnmatch(file_name, "*.jsonl.gz"):
		obj = s3.Object(BUCKET_NAME, file_name)
		data = []
		with gzip.GzipFile(fileobj=obj.get()["Body"]) as f:
			for line in f:
				data.append(load_neccessary(line))
	else:
		raise TypeError('Input file type not recognised')

	if not data[0].get('full_text'):
		raise ValueError('The loaded data has no full text field')

	return data

def load_model(config_name):

	# Load sentence classifier trained model and config it came with
	# Be careful here if you change output locations in sentence_classifier.py
	model_dir = f"outputs/sentence_classifier/models/{config_name.replace('.', '_')}.pkl"
	config_dir = f"skills_taxonomy_v2/config/sentence_classifier/{config_name}.yaml"

	with open(config_dir, "r") as f:
		config = yaml.load(f, Loader=yaml.FullLoader)

	# Loading the model
	sent_classifier = SentenceClassifier(
		bert_model_name=config['flows']['sentence_classifier_flow']['params']['bert_model_name'],
		multi_process=config['flows']['sentence_classifier_flow']['params']['multi_process']
	)
	sent_classifier.load_model(model_dir)

	return sent_classifier, config

def split_sentences(nlp, data, min_length=15, max_length=100):
	"""
	Split the sentences. Only include in output if sentence length
	is in a range.
	Output is two lists, a list of each sentence and a list of the job_ids they are from.
	"""
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
			if len(sentence) in range(min_length, max_length):
				sentences.append(sentence)
				job_ids.append(job_id)

	return sentences, job_ids


def predict_sentences(sent_classifier, sentences):

	# Predict
	sentences_vec = sent_classifier.transform(sentences)
	sentences_pred = sent_classifier.predict(sentences_vec)

	return sentences_pred, sentences_vec

def combine_output(job_ids, sentences, sentences_pred, sentences_vec=None):
	"""
	Combine the job_ids, sentences, sentences_pred and sentences_vec (optional since it is so large and possibly unneeded)
	in a dict, filtering out any sentences that were predicted as not-skills.
	Output (dict): {'job_id_1': [('sentence1', [1.5, 1.4 ...]), ('sentence1', [1.5, 1.4 ...])],
					'job_id_2': [('sentence1', [1.5, 1.4 ...]), ('sentence1', [1.5, 1.4 ...]}
	"""
	skill_sentences_dict = {}
	for job_id in set(job_ids):
		# Which values correspond to this job id
		if sentences_vec is None:
			sentence_list = [sentences[i] for i, e in enumerate(job_ids) if e==job_id and sentences_pred[i]==1]
		else:
			sentence_list = [(sentences[i], sentences_vec[i].tolist()) for i, e in enumerate(job_ids) if e==job_id and sentences_pred[i]==1]
		if len(sentence_list) != 0:
			skill_sentences_dict[job_id] = sentence_list
	return skill_sentences_dict
	

def save_outputs(skill_sentences_dict, output_file_dir):

	directory = os.path.dirname(output_file_dir)

	if not os.path.exists(directory):
		os.makedirs(directory)

	with open(output_file_dir, 'w') as file:
		json.dump(skill_sentences_dict, file)

def save_outputs_to_s3(s3, skill_sentences_dict, output_file_dir):
  
	obj = s3.Object(BUCKET_NAME, output_file_dir)

	obj.put(
		Body=json.dumps(skill_sentences_dict)
	)

def load_outputs(file_name):

	with open(file_name, "r") as file:
		output = json.load(file)

	return output

def get_output_name(data_path, input_dir, output_dir, model_config_name):

	# For naming the output, remove the input dir folder structure
	data_dir = os.path.relpath(data_path, input_dir)

	# Put the output in a folder with a similar naming structure to the input
	# Should work for .jsonl and .jsonl.gz
	output_file_dir = os.path.join(output_dir, data_dir.split('.json')[0] + '_' + model_config_name)
	output_file_dir = output_file_dir.replace('.', '_') + '.json'

	return output_file_dir

def get_local_data_paths(root):

	if os.path.isdir(root):
		# If data_dir to predict on is a directory
		# get all the names of jsonl or jsonl.gz files in this location
		pattern = "*.jsonl*"
		data_paths = []
		for path, subdirs, files in os.walk(root):
			for name in files:
				if fnmatch(name, pattern):
					data_paths.append(os.path.join(path, name))
	else:
		data_paths = [root]

	return data_paths

def get_s3_data_paths(bucket, root):
	pattern = "*.jsonl*"
	s3_keys = []
	for obj in bucket.objects.all():
		key = obj.key
		if root in key:
			if fnmatch(key, pattern):
				s3_keys.append(key)

	return s3_keys


def run_predict_sentence_class(input_dir, data_dir, model_config_name, output_dir, data_local=True):
	"""
	Given the input dir, get predictions and save out for every relevant file in the dir.
	data_local : True if the data is stored localle, False if you are reading from S3
	"""

	sent_classifier, _ = load_model(model_config_name)
	nlp = spacy.load("en_core_web_sm")

	root = os.path.join(input_dir, data_dir)

	if data_local:
		data_paths = get_local_data_paths(root)
	else:
		s3 = boto3.resource('s3')
		bucket = s3.Bucket(BUCKET_NAME)
		data_paths = get_s3_data_paths(bucket, root)

	# Make predictions and save output for data path(s)
	logger.info(f"Running predictions on {len(data_paths)} data files ...")
	for data_path in data_paths[0:2]:
		# Run predictions and save outputs iteratively
		try:
			logger.info(f"Loading data from {data_path} ...")
			if data_local:
				data = load_local_data(data_path)
			else:
				data = load_s3_data(data_path, s3)
			output_file_dir = get_output_name(data_path, input_dir, output_dir, model_config_name)
			sentences, job_ids = split_sentences(nlp, data[0:10], min_length=15, max_length=100)
			sentences_pred, _ = predict_sentences(sent_classifier, sentences)
			skill_sentences_dict = combine_output(job_ids, sentences, sentences_pred, sentences_vec=None)

			logger.info(f"Saving data to {output_file_dir} ...")
			if data_local:
				save_outputs(skill_sentences_dict, output_file_dir)
			else:
				save_outputs_to_s3(s3, skill_sentences_dict, output_file_dir)

		except ValueError:
			logger.info('Skipping this data file since there is no full text field in it')

def parse_arguments(parser):

	parser.add_argument(
		'--config_path',
		help='Path to config file',
		default='skills_taxonomy_v2/config/predict_skill_sentences/2021.07.19.local.sample.yaml'
	)

	return parser.parse_args()

if __name__ == '__main__':

	parser = ArgumentParser()
	args = parse_arguments(parser)

	with open(args.config_path, "r") as f:
		config = yaml.load(f, Loader=yaml.FullLoader)

	FLOW_ID = "predict_skill_sentences_flow"

	flow_config = config["flows"][FLOW_ID]
	params = flow_config["params"]

	run_predict_sentence_class(
		params['input_dir'],
		params['data_dir'],
		params['model_config_name'],
		params['output_dir'],
		data_local=params['data_local'])

	