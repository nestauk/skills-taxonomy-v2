"""
Import the embeddings.
Find a good sample number to train the reducer class on.
Find a good number of dimensions to reduce the embeddings to.
"""

import yaml
import random
from tqdm import tqdm

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

sentence_embeddings_dir = 'outputs/skills_extraction/word_embeddings/data/2021.11.05'

s3 = boto3.resource("s3")

sentence_embeddings_dirs = get_s3_data_paths(s3, BUCKET_NAME, sentence_embeddings_dir, file_types=["*.json"])

# # Load just a sample
# n_files_sample = 10

# # If you want to just do this on a sample of the data:
# indivs = set([s.split('_embeddings.json')[0].split('_original_sentences.json')[0] for s in sentence_embeddings_dirs])
# indivs_sample = random.sample(indivs, n_files_sample)
# sentence_embeddings_dirs = [i+'_embeddings.json' for i in indivs_sample]
# sentence_embeddings_dirs += [i+'_original_sentences.json' for i in indivs_sample]

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
		n_in_sample_each_file[embedding_dir] = count

save_to_s3(
		s3, BUCKET_NAME, n_in_sample_each_file, "outputs/skills_extraction/word_embeddings/data/2021.11.05_n_in_sample_each_file.json",
	)
save_to_s3(
		s3, BUCKET_NAME, n_all_each_file, "outputs/skills_extraction/word_embeddings/data/2021.11.05_n_all_each_file.json",
	)

save_to_s3(
		s3, BUCKET_NAME, embeddings_sample[0:250000], "outputs/skills_extraction/word_embeddings/data/2021.11.05_sample_0.json",
	)
save_to_s3(
		s3, BUCKET_NAME, embeddings_sample[250000:500000], "outputs/skills_extraction/word_embeddings/data/2021.11.05_sample_1.json",
	)
save_to_s3(
		s3, BUCKET_NAME, embeddings_sample[500000:750000], "outputs/skills_extraction/word_embeddings/data/2021.11.05_sample_2.json",
	)
save_to_s3(
		s3, BUCKET_NAME, embeddings_sample[750000:], "outputs/skills_extraction/word_embeddings/data/2021.11.05_sample_3.json",
	)




# # It's easier to manipulate this dataset as a dataframe
# sentences_data = pd.DataFrame(sentences_data)


# # Filter to just include sentences under a length threshold

# sent_thresh = 250
# sentences_data_filt = sentences_data[sentences_data["length original"]<sent_thresh].reset_index()
# sentences_data_filt.head(2)
# len(sentences_data_filt)

