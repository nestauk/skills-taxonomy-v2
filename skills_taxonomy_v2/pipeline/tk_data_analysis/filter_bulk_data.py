"""
Quick script to filter all bulk data to just save job ids which we used in our skill data sample.
"""

from tqdm import tqdm
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import boto3

from skills_taxonomy_v2.getters.s3_data import load_s3_data, save_to_s3
from skills_taxonomy_v2 import BUCKET_NAME

s3 = boto3.resource("s3")

if __name__ == "__main__":

    # All 5 million job adverts in the sample
    sample_dict = load_s3_data(s3, BUCKET_NAME, "outputs/tk_sample_data/14.01.22.sample_file_locations.json")

    skill_job_ads = set([v for s in sample_dict.values() for v in s])

    # Takes up a fair amount of memory, so do separately
    job_dates = defaultdict(list)
    for file_name in tqdm(range(0, 14)):
        # Job dates
        date_dict = load_s3_data(
            s3, BUCKET_NAME, f"outputs/tk_data_analysis_new_method/metadata_date/{file_name}.json"
        )        
        for job_id, date_list in date_dict.items():
            if job_id in skill_job_ads:
                job_dates[job_id].append(date_list)

    print(len(job_dates))
    save_to_s3(
        s3,
        BUCKET_NAME,
        job_dates,
        "outputs/tk_data_analysis_new_method/metadata_date/14.01.22/sample_filtered.json",
    )

    job_locations = defaultdict(list)
    for file_name in tqdm(range(0, 14)):
        # Job regions
        region_dict = load_s3_data(
            s3, BUCKET_NAME, f"outputs/tk_data_analysis_new_method/metadata_location/{file_name}.json"
        )        
        for job_id, region_list in region_dict.items():
            if job_id in skill_job_ads:
                job_locations[job_id].append(region_list)
    print(len(job_locations))
    save_to_s3(
        s3,
        BUCKET_NAME,
        job_locations,
        "outputs/tk_data_analysis_new_method/metadata_location/14.01.22/sample_filtered.json",
    )

    job_titles = defaultdict(list)
    for file_name in tqdm(range(0, 14)):
        # Job titles
        titles_dict = load_s3_data(
            s3, BUCKET_NAME, f"outputs/tk_data_analysis_new_method/metadata_job/{file_name}.json"
        )        
        for job_id, titles_list in titles_dict.items():
            if job_id in skill_job_ads:
                job_titles[job_id].append(titles_list)

    print(len(job_titles))
    save_to_s3(
        s3,
        BUCKET_NAME,
        job_titles,
        "outputs/tk_data_analysis_new_method/metadata_job/14.01.22/sample_filtered.json",
    )
