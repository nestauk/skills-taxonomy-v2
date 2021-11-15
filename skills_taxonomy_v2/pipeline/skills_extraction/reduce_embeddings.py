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

import pandas as pd
import boto3
from sklearn import metrics
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.neighbors import NearestNeighbors

from skills_taxonomy_v2.getters.s3_data import get_s3_data_paths, save_to_s3, load_s3_data
from skills_taxonomy_v2.pipeline.skills_extraction.extract_skills_utils import (
	load_sentences_embeddings,ExtractSkills
	)
from skills_taxonomy_v2 import BUCKET_NAME

logger = logging.getLogger(__name__)

def parse_arguments(parser):

	parser.add_argument(
		"--config_path",
		help="Path to config file",
		default="skills_taxonomy_v2/config/skills_extraction/2021.11.05.yaml",
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

	logger.info("Load the sample of embeddings for fitting the reducer class to...")

	embeddings_sample_0 = load_s3_data(
		s3, BUCKET_NAME, params["embeddings_sample_0"]
	)
	embeddings_sample_1 = load_s3_data(
		s3, BUCKET_NAME, params["embeddings_sample_1"]
	)
	embeddings_sample = embeddings_sample_0 + embeddings_sample_1
	embeddings_sample_300k = embeddings_sample[0:params["fit_reducer_n"]]

	logger.info(f"Fitting reducer class to {len(embeddings_sample_300k)} embeddings...")
	# Fit reducer class
	reducer_class = umap.UMAP(
		n_neighbors=params["umap_n_neighbors"],
		min_dist=params["umap_min_dist"],
		random_state=params["umap_random_state"],
		n_components=params["umap_n_components"],
	)
	reducer_class.fit(embeddings_sample_300k)

	sentence_embeddings_dirs = get_s3_data_paths(s3, BUCKET_NAME, sentence_embeddings_dir, file_types=["*.json"])

	# You want a sample of 1 million embeddings (which should be far more than we actually will need to use)
	# So get a random 2000 from each file

	# Load a sample of the embeddings from each file
	# when sentence len <250 and 
	# No repeats

	logger.info(f"Loading original sentences")

	original_sentences = {}
	for embedding_dir in sentence_embeddings_dirs:
		if "original_sentences.json" in embedding_dir:
			original_sentences.update(load_s3_data(s3, BUCKET_NAME, embedding_dir))

	sent_thresh = params["sent_thresh"]

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

				logger.info(f"Saving reduced embeddings from {len(sentences_data)} sentences")
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
