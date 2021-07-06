"""
Usage
----------
python skills_taxonomy_v2/pipeline/sentence_classifier/sentence_classifier.py
	--config_path config/sentence_classifier/2021.07.06.yaml
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	f1_score,
	precision_score,
	recall_score,
	confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from transformers import BertTokenizer, BertModel
import tqdm as tqdm
import spacy

import json
import random
from collections import Counter
import re
from argparse import ArgumentParser
import pickle
import configparser
import pickle
import os
import yaml
	
from skills_taxonomy_v2.pipeline.sentence_classifier.create_training_data import load_training_data

class BertVectorizer():
	"""
	Use a pretrained transformers model to embed sentences.
	In this form so it can be used as a step in the pipeline.
	layer_type: which layer to output, 'last_hidden_state' or 'pooler_output'
	"""
	def __init__(
		self,
		bert_model_name='bert-base-uncased',
		layer_type='last_hidden_state'
		):
		self.bert_model_name = bert_model_name
		self.layer_type = layer_type

	def fit(self, *_):
		self.bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)
		self.bert_model = BertModel.from_pretrained(self.bert_model_name)
		return self

	def get_embedding(self, text):
		encoded_input = self.bert_tokenizer.encode(text, return_tensors="pt")
		encoded_input = encoded_input[:,:510]# could do something better?
		output = self.bert_model(encoded_input)
		embedded_x = output[self.layer_type]	
		if self.layer_type == 'last_hidden_state':
			embedded_x = embedded_x.mean(dim=1)

		return embedded_x.detach().numpy().flatten()

	def transform(self, texts):

		self.transformed_texts = [self.get_embedding(x) for x in tqdm.tqdm(texts)]
		return self.transformed_texts


class SentenceClassifier():
	"""
	A class the train/save/load/predict a classifier to predict whether
	a sentence contains skills or not.
	...
	Attributes
	----------
	test_size : float (default 0.25)
	split_random_seed : int (default 1)
	log_reg_max_iter: int (default 1000)
	Methods
	-------

	split_data(training_data)
		Split the training data (list of pairs of text-label) into test/train sets
	fit_transform(X)
		Load the pretrained BERT models and transform X
	transform(X)
		Transform X uses already loaded BERT model
	fit(X_vec, y)
		Fit the scaler and classifier to vectorized X
	predict(X_vec)
		Predict classes from already vectorized text
	predict_transform(X)
		Transform then predict classes from text
	evaluate(y, y_pred)
	save_model(file_name)
	load_model(file_name)
	"""
	def __init__(
		self,
		split_random_seed=1,
		test_size=0.25,
		log_reg_max_iter=1000,
		bert_model_name='bert-base-uncased',
		layer_type='last_hidden_state'
		):

		self.split_random_seed = split_random_seed
		self.test_size = test_size
		self.log_reg_max_iter = log_reg_max_iter
		self.bert_model_name = bert_model_name
		self.layer_type = layer_type
		

	def split_data(self, training_data, verbose=False):
		
		X = [t[0] for t in training_data]
		y = [t[1] for t in training_data]
		X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=self.test_size, random_state=self.split_random_seed)

		if verbose:
			print(f'Size of training data: {len(y_train)}')
			print(f'Size of test data: {len(y_test)}')
			print(f'Counter of training data classes: {Counter(y_train)}')
			print(f'Counter of training data classes: {Counter(y_test)}')
		return X_train, X_test, y_train, y_test

	def load_bert(self):
		self.bert_vectorizer = BertVectorizer(bert_model_name=self.bert_model_name, layer_type=self.layer_type)
		self.bert_vectorizer.fit()

	def fit_transform(self, X):
		
		# Load BERT models and transform X
		self.load_bert()
		X_vec = self.bert_vectorizer.transform(X)

		return X_vec

	def transform(self, X):
		return self.bert_vectorizer.transform(X)

	def fit(self, X_vec, y):
		# Fit classifier
		# Including the BertVectorizer in this means the outputted model is very big, so we won't include
		self.classifier = Pipeline([('scaler', MinMaxScaler()), ('classifier', LogisticRegression(max_iter=self.log_reg_max_iter, class_weight="balanced"))])
		self.classifier.fit(X_vec, y)

	def predict(self, X_vec):
		return self.classifier.predict(X_vec)

	def predict_transform(self, X):
		X_vec = self.transform(X)
		return self.predict(X_vec)

	def evaluate(self, y, y_pred, verbose=True):
		class_rep = classification_report(y, y_pred, output_dict=True)
		if verbose:
			print(classification_report(y, y_pred))
			print(confusion_matrix(y, y_pred))
		return class_rep

	def save_model(self, file_name):
		directory = os.path.dirname(file_name)

		if not os.path.exists(directory):
			os.makedirs(directory)

		with open(file_name, 'wb') as f:
			pickle.dump(self.classifier, f)

	def load_model(self, file_name):

		with open(file_name, 'rb') as f:
			self.classifier = pickle.load(f)

		# Load BERT models
		self.load_bert()

		return self.classifier


if __name__ == '__main__':

	# Later this can all go in a run.py and sentence_classifier_flow.py file

	# Load specific config file
	yaml_file_name = "2021.07.06"
	fname = os.path.join(
		"skills_taxonomy_v2", "config", "sentence_classifier", yaml_file_name + ".yaml"
	)
	with open(fname, "r") as f:
		config = yaml.load(f, Loader=yaml.FullLoader)

	# Get parameters
	FLOW_ID = "sentence_classifier_flow"
	flow_config = config["flows"][FLOW_ID]
	params = flow_config["params"]

	training_data_file = params['training_data_file']
	split_random_seed = params['split_random_seed']
	test_size = params['test_size']
	log_reg_max_iter = params['log_reg_max_iter']
	bert_model_name = params['bert_model_name']
	layer_type = params['layer_type']

	# Output file name
	output_dir = params['output_dir']
	file_name = os.path.join(output_dir, yaml_file_name.replace('.', '_'))

	# Run flow
	training_data = load_training_data(training_data_file)
	
	sent_class = SentenceClassifier(
		split_random_seed=split_random_seed,
		test_size=test_size,
		log_reg_max_iter=log_reg_max_iter,
		bert_model_name=bert_model_name,
		layer_type=layer_type
		)
	X_train, X_test, y_train, y_test = sent_class.split_data(training_data, verbose=True)
	
	X_train_vec = sent_class.fit_transform(X_train)
	sent_class.fit(X_train_vec, y_train)

	# Training evaluation
	y_train_pred = sent_class.predict(X_train_vec)
	class_rep_train = sent_class.evaluate(y_train, y_train_pred, verbose=True)

	# Test evaluation
	y_test_pred = sent_class.predict_transform(X_test)
	class_rep_test = sent_class.evaluate(y_test, y_test_pred, verbose=True)

	results = {'Train': class_rep_train, 'Test': class_rep_test}

	# Output
	sent_class.save_model(file_name + '.pkl')
	with open(file_name + '_results.txt', 'w') as file:
		json.dump(results, file)


	# # Loading a model
	# file_name = f'outputs/sentence_classifier/models/{job_id}.pkl'

	# sent_class = SentenceClassifier()
	# sent_class.load_model(file_name)
	
	# X_train, X_test, y_train, y_test = sent_class.split_data(training_data, verbose=True)

	# y_test_pred = sent_class.predict_transform(X_test)
	# class_rep_test = sent_class.evaluate(y_test, y_test_pred, verbose=True)
