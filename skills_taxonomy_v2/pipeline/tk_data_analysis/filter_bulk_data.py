"""
Quick script to filter all bulk data to just save job ids which we used in our skill data sample.
"""

from tqdm import tqdm
import json
import os

import numpy as np
import pandas as pd
import boto3

from skills_taxonomy_v2.getters.s3_data import load_s3_data, save_to_s3
from skills_taxonomy_v2 import BUCKET_NAME

s3 = boto3.resource("s3")

if __name__ == "__main__":

    # All 5 million job adverts in the original sample
    original_sample = load_s3_data(s3, BUCKET_NAME, "outputs/tk_sample_data/sample_file_locations.json")
    skill_job_ads = set([v for s in original_sample.values() for v in s])

    # Job titles
    job_titles = {}
    for file_name in tqdm(range(0, 13)):
        file_dict = load_s3_data(
            s3, BUCKET_NAME, f"outputs/tk_data_analysis/metadata_job/{file_name}.json"
        )
        job_titles.update(
            {job_id: f for job_id, f in file_dict.items() if job_id in skill_job_ads}
        )

    print(len(job_titles))
    save_to_s3(
        s3,
        BUCKET_NAME,
        job_titles,
        "outputs/tk_data_analysis/metadata_job/sample_filtered_2021.11.05.json",
    )

    # Job dates
    job_dates = {}
    for file_name in tqdm(range(0, 13)):
        file_date_dict = load_s3_data(
            s3, BUCKET_NAME, f"outputs/tk_data_analysis/metadata_date/{file_name}.json"
        )
        job_dates.update(
            {
                job_id: f[0]
                for job_id, f in file_date_dict.items()
                if job_id in skill_job_ads
            }
        )

    print(len(job_dates))
    save_to_s3(
        s3,
        BUCKET_NAME,
        job_dates,
        "outputs/tk_data_analysis/metadata_date/sample_filtered_2021.11.05.json",
    )

    # Job location
    job_locations = {}
    for file_name in tqdm(range(0, 13)):
        file_date_dict = load_s3_data(
            s3, BUCKET_NAME, f"outputs/tk_data_analysis/metadata_location/{file_name}.json"
        )
        job_locations.update(
            {
                job_id: f[0]
                for job_id, f in file_date_dict.items()
                if job_id in skill_job_ads
            }
        )

    print(len(job_locations))
    save_to_s3(
        s3,
        BUCKET_NAME,
        job_locations,
        "outputs/tk_data_analysis/metadata_location/sample_filtered_2021.11.05.json",
    )
