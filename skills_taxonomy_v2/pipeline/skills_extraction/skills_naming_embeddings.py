"""
For the skill naming we need the mean embedding for each skill.
"""
import pandas as pd
import numpy as np
import boto3
from tqdm import tqdm

from collections import defaultdict

from skills_taxonomy_v2.getters.s3_data import (
    get_s3_data_paths,
    save_to_s3,
    load_s3_data,
)
from skills_taxonomy_v2 import BUCKET_NAME

s3 = boto3.resource("s3")

skills_date = "2022.01.14"

# Load skills
# The sentences ID + cluster num
sentence_embs = load_s3_data(
    s3,
    BUCKET_NAME,
    f"outputs/skills_extraction/extracted_skills/{skills_date}_sentences_skills_data_lightweight.json",
)
sentence_embs_dict = defaultdict(dict)
for job_id, sent_id, skill_num in tqdm(sentence_embs):
    if skill_num >= 0:
        sentence_embs_dict[job_id].update({sent_id: skill_num})

# Load embeddings
sentence_embeddings_dirs = get_s3_data_paths(
    s3,
    BUCKET_NAME,
    f"outputs/skills_extraction/word_embeddings/data/{skills_date}",
    file_types=["*embeddings.json"],
)

skill_embeddings_sum = {}
for embedding_dir in tqdm(sentence_embeddings_dirs):
    sentence_embeddings = load_s3_data(s3, BUCKET_NAME, embedding_dir)
    for job_id, sent_id, _, embedding in sentence_embeddings:
        skill_num = sentence_embs_dict[job_id].get(sent_id)
        if skill_num != None:
            if skill_num in skill_embeddings_sum:
                skill_embeddings_sum[skill_num] = np.sum(
                    [skill_embeddings_sum[skill_num], np.array(embedding)], axis=0
                )
            else:
                skill_embeddings_sum[skill_num] = np.array(embedding)

# Get mean embedding for each skill number
print("Getting mean embeddings")
mean_skill_embeddings = {}
for skill_num, sum_embeddings in skill_embeddings_sum.items():
    mean_skill_embeddings[skill_num] = (sum_embeddings / len(sum_embeddings)).tolist()

# Save out
print("Saving mean embeddings")
save_to_s3(
    s3,
    BUCKET_NAME,
    mean_skill_embeddings,
    f"outputs/skills_extraction/extracted_skills/{skills_date}_skill_mean_embeddings.json",
)
