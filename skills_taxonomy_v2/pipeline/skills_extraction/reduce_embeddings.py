"""
1. Load sample of embeddings (found in data_reduction_param_exploration.py)
2. Fit reducer class on 300k of them
3. Tranform each file of embeddings and output
"""

from argparse import ArgumentParser
import yaml
import random
from tqdm import tqdm
import logging
import os

import pandas as pd
import boto3
from sklearn import metrics
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.neighbors import NearestNeighbors

from skills_taxonomy_v2.getters.s3_data import get_s3_data_paths, save_to_s3, load_s3_data
from skills_taxonomy_v2 import BUCKET_NAME

logger = logging.getLogger(__name__)

def fit_reducer_class(
	s3,
	data_paths_list,
	fit_reducer_n,
	umap_n_neighbors,
	umap_min_dist,
	umap_random_state,
	umap_n_components
	):
	"""
	Load a random sample of the embeddings, take the first fit_reducer_n
	of them, and fit the reducer class with them.

	data_paths_list will be a list of random embeddings so no need
	to further randomise.
	"""

	logger.info("Load the sample of embeddings for fitting the reducer class to...")

	embeddings_sample = []
	for data_path in data_paths_list:
		embeddings_sample += load_s3_data(
			s3, BUCKET_NAME, data_path
		)
	embeddings_sample = embeddings_sample[0:fit_reducer_n]

	logger.info(f"Fitting reducer class to {len(embeddings_sample)} embeddings...")
	# Fit reducer class
	reducer_class = umap.UMAP(
		n_neighbors=umap_n_neighbors,
		min_dist=umap_min_dist,
		random_state=umap_random_state,
		n_components=umap_n_components,
	)
	reducer_class.fit(embeddings_sample)

	return reducer_class

def parse_arguments(parser):

	parser.add_argument(
		"--config_path",
		help="Path to config file",
		default="skills_taxonomy_v2/config/skills_extraction/2022.01.14.yaml",
	)

	return parser.parse_args()

if __name__ == "__main__":

	parser = ArgumentParser()
	args = parse_arguments(parser)

	with open(args.config_path, "r") as f:
		config = yaml.load(f, Loader=yaml.FullLoader)

	FLOW_ID = "reduce_embeddings"

	flow_config = config["flows"][FLOW_ID]
	params = flow_config["params"]

	sentence_embeddings_dir = params["sentence_embeddings_dir"]

	s3 = boto3.resource("s3")

	reducer_class = fit_reducer_class(
		s3,
		[params["embeddings_sample_0"], params["embeddings_sample_1"]],
		params["fit_reducer_n"],
		params["umap_n_neighbors"],
		params["umap_min_dist"],
		params["umap_random_state"],
		params["umap_n_components"],
		)

	sentence_embeddings_dirs = get_s3_data_paths(s3, BUCKET_NAME, sentence_embeddings_dir, file_types=["*.json"])

	logger.info(f"Loading original sentences")

	original_sentences = {}
	for embedding_dir in sentence_embeddings_dirs:
		if "original_sentences.json" in embedding_dir:
			original_sentences.update(load_s3_data(s3, BUCKET_NAME, embedding_dir))

	sent_thresh = params["sent_thresh"]
	output_dir = os.path.join(
	                params["output_dir"],
	                os.path.basename(args.config_path).split(".yaml")[0],
	            )
	logger.info(f"Will be saving outputs to {output_dir}")

	unique_sentences = set()

	words_list = []
	sentence_list = []
	jobid_list = []
	sentid_list = []
	sentence_embeddings_red = []

	output_count = 0
	logger.info(f"Reducing embeddings from {len(sentence_embeddings_dirs)/2} files")
	for embedding_dir in tqdm(sentence_embeddings_dirs):
		if "embeddings.json" in embedding_dir:
			sentence_embeddings = load_s3_data(s3, BUCKET_NAME, embedding_dir)
			# Only get the reduced embeddings if length is under the threshold and no repeats
			embedding_list = []
			for job_id, sent_id, words, embedding in sentence_embeddings:
				if words not in unique_sentences:
					original_sentence = original_sentences[str(sent_id)]
					if len(original_sentence) < sent_thresh:
						unique_sentences.add(words)
						embedding_list.append(embedding)
						words_list.append(words.replace(params["mask_seq"], "").split())
						sentence_list.append(original_sentence)
						jobid_list.append(job_id)
						sentid_list.append(sent_id)
			# Reduce the embeddings
			sentence_embeddings_red = sentence_embeddings_red + reducer_class.transform(embedding_list).tolist()
			if len(sentence_embeddings_red) > 500000:
				sentences_data =  {
					"description": words_list,
					"original sentence": sentence_list,
					"job id": jobid_list,
					"sentence id": sentid_list,
					"embedding": sentence_embeddings_red,
				}
				logger.info(f"Saving reduced embeddings from {len(sentences_data['sentence id'])} sentences")
				save_to_s3(
					s3,
					BUCKET_NAME,
					sentences_data,
					output_dir + f"sentences_data_{output_count}.json",
				)
				words_list = []
				sentence_list = []
				jobid_list = []
				sentid_list = []
				sentence_embeddings_red = []
				output_count = output_count + 1


	sentences_data =  {
		"description": words_list,
		"original sentence": sentence_list,
		"job id": jobid_list,
		"sentence id": sentid_list,
		"embedding": sentence_embeddings_red,
	}

	logger.info(f"Saving reduced embeddings from {len(sentences_data)} sentences")
	save_to_s3(
		s3,
		BUCKET_NAME,
		sentences_data,
		output_dir + f"sentences_data_{output_count}.json",
	)
