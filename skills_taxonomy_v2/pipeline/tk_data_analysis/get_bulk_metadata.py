"""
The TextKernel data is stored in 686 separate files each with 100k job adverts.
In this script we extract some key metadata for each job advert to be stored in a single dictionary.

This will be useful for some analysis pieces.
"""

import boto3
import pandas as pd
from tqdm import tqdm

import json
import gzip
import os

from skills_taxonomy_v2 import BUCKET_NAME

from skills_taxonomy_v2.getters.s3_data import get_s3_data_paths, save_to_s3


s3 = boto3.resource("s3")

if __name__ == '__main__':

	tk_data_path = "inputs/data/textkernel-files/"
	output_dir = "outputs/tk_data_analysis/"

	all_tk_data_paths = get_s3_data_paths(
		s3, BUCKET_NAME, tk_data_path, file_types=["*.jsonl*"]
	)

	file_num = 0
	count_tk_files = 0
	job_id_file_dict = {}
	job_id_date_dict = {}
	job_id_meta_dict = {}
	job_id_job_dict = {}
	job_id_location_dict = {}
	for file_name in tqdm(all_tk_data_paths):
		obj = s3.Object(BUCKET_NAME, file_name)
		with gzip.GzipFile(fileobj=obj.get()["Body"]) as file:
			for line in file:
				d = json.loads(line)
				# Save out as little info as possible to make file smaller
				job_id_file_dict[d['job_id']] = file_name.split(tk_data_path)[1]
				job_id_date_dict[d['job_id']] = [
					d.get('date'),
					d.get('expiration_date')
					]
				job_id_meta_dict[d['job_id']] = [
					d.get('source_website'),
					d.get('language'),
				]
				job_id_job_dict[d['job_id']] = [
					d.get('job_title'),
					d.get('organization_industry').get('label'),
				]
				region = d.get('region')
				subregion = d.get('subregion')
				job_id_location_dict[d['job_id']] = [
					d.get('location_name')
					d.get('location_coordinates'),
					region.get('label') if region else None,
					subregion.get('label') if subregion else None,
				]
		count_tk_files += 1
		if count_tk_files == 50:
			print('Saving data ...')
			save_to_s3(s3, BUCKET_NAME, job_id_file_dict, os.path.join(output_dir, f'metadata_file/{file_num}.json'))
			save_to_s3(s3, BUCKET_NAME, job_id_date_dict, os.path.join(output_dir, f'metadata_date/{file_num}.json'))
			save_to_s3(s3, BUCKET_NAME, job_id_meta_dict, os.path.join(output_dir, f'metadata_meta/{file_num}.json'))
			save_to_s3(s3, BUCKET_NAME, job_id_job_dict, os.path.join(output_dir, f'metadata_job/{file_num}.json'))
			save_to_s3(s3, BUCKET_NAME, job_id_location_dict, os.path.join(output_dir, f'metadata_location/{file_num}.json'))
			file_num += 1
			count_tk_files = 0
			job_id_file_dict = {}
			job_id_date_dict = {}
			job_id_meta_dict = {}
			job_id_job_dict = {}
			job_id_location_dict = {}
