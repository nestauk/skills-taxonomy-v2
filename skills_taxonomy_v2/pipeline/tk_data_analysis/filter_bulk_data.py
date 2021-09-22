"""
Quick script to filter all bulk data to just save job ids which we used in our skill data sample.
"""

from tqdm import tqdm
import json

import numpy as np
import pandas as pd
import boto3

from skills_taxonomy_v2.getters.s3_data import load_s3_data
from skills_taxonomy_v2 import BUCKET_NAME

s3 = boto3.resource("s3")

if __name__ == '__main__':

	sentences_data_dir = 'outputs/skills_extraction/extracted_skills/2021.08.31_sentences_data.json'
	
	sentence_data = load_s3_data(s3, BUCKET_NAME, sentences_data_dir)
	sentence_data = pd.DataFrame(sentence_data)
	sentence_data = sentence_data[sentence_data['Cluster number']!=-1]

	# Job adverts in our sample
	skill_job_ads = list(sentence_data['job id'].unique())

	# Job titles
	job_titles = {}
	for file_name in tqdm(range(0,13)):
	    file_date_dict = load_s3_data(
	        s3,
	        BUCKET_NAME,
	        f'outputs/tk_data_analysis/metadata_job/{file_name}.json')
	    for job_id, f in file_date_dict.items():
	        if job_id in skill_job_ads:
	            job_titles[job_id] = f
	    
	print(len(job_titles))
	save_to_s3(
		s3,
		BUCKET_NAME,
		job_titles,
		os.path.join('outputs/tk_data_analysis/metadata_job/sample_filtered.json')
		)

	# Job dates
	job_dates = {}
	for file_name in tqdm(range(0,13)):
	    file_date_dict = load_s3_data(
	        s3,
	        BUCKET_NAME,
	        f'outputs/tk_data_analysis/metadata_date/{file_name}.json')
	    for job_id, f in file_date_dict.items():
	        if job_id in skill_job_ads:
	            job_dates[job_id] = f[0]
	    
	print(len(job_dates))
	save_to_s3(
		s3,
		BUCKET_NAME,
		job_dates,
		os.path.join('outputs/tk_data_analysis/metadata_date/sample_filtered.json')
		)
