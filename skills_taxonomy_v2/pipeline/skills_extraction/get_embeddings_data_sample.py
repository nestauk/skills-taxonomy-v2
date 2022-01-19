"""
Get samples of the embeddings data.

This is useful because the embeddings data is quite large, so by first taking a sample we can
more quickly fit the reducer class in `reduce_embeddings.py`.

It also always for some pieces of analysis, including:
- Find a good sample number to train the reducer class on.
- Find a good number of dimensions to reduce the embeddings to.
"""

import yaml
import random
from tqdm import tqdm

import pandas as pd
import numpy as np
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

output_date = '2022.01.14'

sentence_embeddings_dir = f'outputs/skills_extraction/word_embeddings/data/{output_date}'

s3 = boto3.resource("s3")

sentence_embeddings_dirs = get_s3_data_paths(s3, BUCKET_NAME, sentence_embeddings_dir, file_types=["*.json"])

# You want a sample of 1 million embeddings (which should be far more than we actually will need to use)
# So get a random 2000 from each file

# Load a sample of the embeddings from each file
# when sentence len <250 and 
# No repeats

original_sentences = {}
for embedding_dir in sentence_embeddings_dirs:
    if "original_sentences.json" in embedding_dir:
        original_sentences.update(load_s3_data(s3, BUCKET_NAME, embedding_dir))

n_each_file = 2000
sent_thresh = 250

n_all_each_file = {}
n_in_sample_each_file = {}
unique_sentences = set()
embeddings_sample = []

count_too_long = 0
for embedding_dir in tqdm(sentence_embeddings_dirs):
	if "embeddings.json" in embedding_dir:
		sentence_embeddings = load_s3_data(s3, BUCKET_NAME, embedding_dir)
		n_all_each_file[embedding_dir] = len(sentence_embeddings)
		random.seed(42)
		sentence_embeddings_sample = random.sample(sentence_embeddings, min(len(sentence_embeddings), n_each_file))
		count = 0
		for _, sent_id, words, embedding in sentence_embeddings_sample:
			if words not in unique_sentences:
				original_sentence = original_sentences[str(sent_id)]
				if len(original_sentence) < sent_thresh:
					unique_sentences.add(words)
					embeddings_sample.append(embedding)
					count += 1
				else:
					count_too_long += 1
		n_in_sample_each_file[embedding_dir] = count

random.seed(42)
random.shuffle(embeddings_sample)

print(f"In total - there are {sum(n_all_each_file.values())} embeddings")
print(f"In the sample - there are {len(unique_sentences)} unique sentences with embeddings where the sentences is <{sent_thresh} characters long")
print(f"In the sample - there were {count_too_long} sentences which were too long to be included (>{sent_thresh} characters long)")

save_to_s3(
		s3, BUCKET_NAME, n_in_sample_each_file, f"outputs/skills_extraction/word_embeddings/data/{output_date}_n_in_sample_each_file.json",
	)
save_to_s3(
		s3, BUCKET_NAME, n_all_each_file, f"outputs/skills_extraction/word_embeddings/data/{output_date}_n_all_each_file.json",
	)

# Split into 3 files (otherwise a single one is too big to do much with)
n_files = 3
split_chunks = [round(n) for n in np.linspace(0, len(embeddings_sample), n_files+1)]
for file_i, start_index in enumerate(split_chunks):
	if file_i != n_files:
		end_index = split_chunks[file_i+1]
		print((start_index, end_index))
		save_to_s3(
			s3, BUCKET_NAME, embeddings_sample[start_index: end_index], f"outputs/skills_extraction/word_embeddings/data/{output_date}_sample_{file_i}.json",
		)

# The order is random anyway so no need to resample
save_to_s3(
    s3,
    BUCKET_NAME,
    embeddings_sample[0:300000],
    f"outputs/skills_extraction/word_embeddings/data/{output_date}_sample_300k.json",
)
