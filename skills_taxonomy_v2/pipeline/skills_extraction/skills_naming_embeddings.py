"""
For the skill naming we need the mean embedding for each skill.
"""
import pandas as pd
import numpy as np
import boto3
from tqdm import tqdm

from collections import defaultdict

from skills_taxonomy_v2.getters.s3_data import get_s3_data_paths, save_to_s3, load_s3_data
from skills_taxonomy_v2 import BUCKET_NAME

s3 = boto3.resource("s3")

# Load skills
# The sentences ID + cluster num
sentence_embs = load_s3_data(s3, BUCKET_NAME, "outputs/skills_extraction/extracted_skills/2021.11.05_sentences_skills_data.json")
sentence_embs = pd.DataFrame(sentence_embs)
sentence_embs = sentence_embs[sentence_embs["Cluster number predicted"] >= 0]

# Load embeddings
sentence_embeddings_dirs = get_s3_data_paths(
	s3, BUCKET_NAME, 'outputs/skills_extraction/word_embeddings/data/2021.11.05', file_types=["*embeddings.json"])

skill_embeddings = defaultdict(list)
for embedding_dir in tqdm(sentence_embeddings_dirs):
	sentence_embeddings = load_s3_data(s3, BUCKET_NAME, embedding_dir)
	sentence_embeddings_df = pd.DataFrame(sentence_embeddings)
	temp_merge = pd.merge(sentence_embs, sentence_embeddings_df, how="inner", left_on=['job id', 'sentence id'], right_on=[0,1])
	for skill_num, embeddings in temp_merge.groupby('Cluster number predicted'):
		skill_embeddings[skill_num].extend(embeddings[3].tolist())

# Get mean embedding for each skill number
print("Getting mean embeddings")
mean_skill_embeddings = {}
for skill_num, embeddings_list in skill_embeddings.items():
	mean_skill_embeddings[skill_num] = np.mean(embeddings_list, axis=0).tolist()

# Save out
print("Saving mean embeddings")
save_to_s3(s3, BUCKET_NAME, mean_skill_embeddings, 'outputs/skills_extraction/extracted_skills/2021.11.05_skill_mean_embeddings.json')