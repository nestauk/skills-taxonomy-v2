"""
In reduce_embeddings.py we don't filter out duplicated sentences.
However, for analysis with the TK data we need to make sure all the
job ids are used.

e.g. if the same sentence is used in two job adverts only one of them is brought
forward, and thus the analysis will miss out including the second job advert in the
counts.
"""
import random
from tqdm import tqdm
import logging
from collections import defaultdict

import pandas as pd
import boto3

from skills_taxonomy_v2.getters.s3_data import get_s3_data_paths, save_to_s3, load_s3_data
from skills_taxonomy_v2 import BUCKET_NAME

logger = logging.getLogger(__name__)

s3 = boto3.resource("s3")

sentence_embeddings_dir = 'outputs/skills_extraction/word_embeddings/data/2021.11.05'
sent_thresh = 250

sentence_embeddings_dirs = get_s3_data_paths(s3, BUCKET_NAME, sentence_embeddings_dir, file_types=["*.json"])

logger.info(f"Loading sentences from {len(sentence_embeddings_dirs)/2} files")
original_sentences = {}
problem_sentence_dir = []
for sentence_dir in tqdm(sentence_embeddings_dirs):
	if "original_sentences.json" in sentence_dir:
		try:
			original_sentences.update(load_s3_data(s3, BUCKET_NAME, sentence_dir))
		except:
			problem_sentence_dir.append(sentence_dir)

print(f"{len(problem_sentence_dir)} problem_sentence_dir")


words_id_list = []

logger.info(f"Loading embeddings from {len(sentence_embeddings_dirs)/2} files")
problem_embedding_dir = []
output_i = 0
count = 0
for embedding_dir in tqdm(sentence_embeddings_dirs):
	if "embeddings.json" in embedding_dir:
		try:
			sentence_embeddings = load_s3_data(s3, BUCKET_NAME, embedding_dir)
		except:
			problem_embedding_dir.append(embedding_dir)
		for job_id, sent_id, words, _ in sentence_embeddings:
			original_sentence = original_sentences[str(sent_id)]
			if len(original_sentence) < sent_thresh:
				words_id = hash(words)
				words_id_list.append([words_id, job_id, sent_id])
				count += 1
		if count%100==0:
			save_to_s3(
				s3, BUCKET_NAME,words_id_list,
				f"outputs/tk_data_analysis_new_method/2021.11.05_words_id_list_{output_i}.json",)
			output_i += 1

print(f"{len(problem_embedding_dir)} problem_sentence_dir")

logger.info(f"Saving remainder word ids")
save_to_s3(
        s3,
        BUCKET_NAME,
        words_id_list,
        f"outputs/tk_data_analysis_new_method/2021.11.05_words_id_list_{output_i}.json",
    )
