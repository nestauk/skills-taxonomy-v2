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

    # All 5 million job adverts in the original sample
    original_sample = load_s3_data(s3, BUCKET_NAME, "outputs/tk_sample_data/sample_file_locations.json")
   
    job_titles = []
    job_dates = []
    job_locations = []
    for file_name, job_ids in tqdm(original_sample.items()):
        job_ids = set(job_ids)
        job_ad_data = load_s3_data(s3, BUCKET_NAME, f'inputs/data/textkernel-files/{file_name}')
        for job_ad in job_ad_data:
            if job_ad['job_id'] in job_ids:
                job_dates.append(
                    (job_ad['job_id'], job_ad['date'])
                )
                job_titles.append(
                    (job_ad['job_id'], job_ad['job_title'])
                )
                job_locations.append(
                    (job_ad['job_id'], (job_ad.get("region"), job_ad.get("subregion"), job_ad.get("location_name"), job_ad.get("location_coordinates")))
                )

    save_to_s3(
        s3,
        BUCKET_NAME,
        job_dates,
        "outputs/tk_data_analysis/metadata_date/sample_filtered_2021.11.05_new_method.json",
    )
    save_to_s3(
        s3,
        BUCKET_NAME,
        job_titles,
        "outputs/tk_data_analysis/metadata_job/sample_filtered_2021.11.05_new_method.json",
    )
    save_to_s3(
        s3,
        BUCKET_NAME,
        job_locations,
        "outputs/tk_data_analysis/metadata_location/sample_filtered_2021.11.05_new_method.json",
    )

    # skill_job_ads = [v for s in original_sample.values() for v in s]

    # # Job titles
    # job_titles = {}
    # for file_name in tqdm(range(0, 13)):
    #     file_dict = load_s3_data(
    #         s3, BUCKET_NAME, f"outputs/tk_data_analysis/metadata_job/{file_name}.json"
    #     )
    #     job_titles.update(
    #         {job_id: f for job_id, f in file_dict.items() if job_id in skill_job_ads}
    #     )

    # print(len(job_titles))
    # save_to_s3(
    #     s3,
    #     BUCKET_NAME,
    #     job_titles,
    #     "outputs/tk_data_analysis/metadata_job/sample_filtered_2021.11.05.json",
    # )

    # # Job dates
    # job_dates = {}
    # for file_name in tqdm(range(0, 13)):
    #     file_date_dict = load_s3_data(
    #         s3, BUCKET_NAME, f"outputs/tk_data_analysis/metadata_date/{file_name}.json"
    #     )
    #     job_dates.update(
    #         {
    #             job_id: f[0]
    #             for job_id, f in file_date_dict.items()
    #             if job_id in skill_job_ads
    #         }
    #     )

    # print(len(job_dates))
    # save_to_s3(
    #     s3,
    #     BUCKET_NAME,
    #     job_dates,
    #     "outputs/tk_data_analysis/metadata_date/sample_filtered_2021.11.05.json",
    # )

    # # Job location
    # job_locations = {}
    # for file_name in tqdm(range(0, 13)):
    #     file_date_dict = load_s3_data(
    #         s3, BUCKET_NAME, f"outputs/tk_data_analysis/metadata_location/{file_name}.json"
    #     )
    #     job_locations.update(
    #         {
    #             job_id: f[0]
    #             for job_id, f in file_date_dict.items()
    #             if job_id in skill_job_ads
    #         }
    #     )

    # print(len(job_locations))
    # save_to_s3(
    #     s3,
    #     BUCKET_NAME,
    #     job_locations,
    #     "outputs/tk_data_analysis/metadata_location/sample_filtered_2021.11.05.json",
    # )

    # # All TK dates counts
    # all_tk_dates = {}
    # for file_name in tqdm(range(0, 13)):
    #     file_date_dict = load_s3_data(
    #         s3, BUCKET_NAME, f"outputs/tk_data_analysis/metadata_date/{file_name}.json"
    #     )
    #     all_tk_dates.update({k: f[0] for k, f in file_date_dict.items()})

    # print(len(all_tk_dates))

    # job_ads_date_count = defaultdict(int)
    # for k, v in tqdm(all_tk_dates.items()):
    #     if v:
    #         date = v[0:7]
    #         job_ads_date_count[date] += 1
    #     else:
    #         job_ads_date_count["No date given"] += 1

    # save_to_s3(
    #     s3,
    #     BUCKET_NAME,
    #     job_ads_date_count,
    #     "outputs/tk_data_analysis/metadata_date/all_tk_dates_counts.json",
    # )

    # # All TK locations counts
    # all_tk_locations = {}
    # for file_name in tqdm(range(0, 13)):
    #     file_date_dict = load_s3_data(
    #         s3, BUCKET_NAME, f"outputs/tk_data_analysis/metadata_location/{file_name}.json"
    #     )
    #     all_tk_locations.update({k: f[0] for k, f in file_date_dict.items()})

    # print(len(all_tk_locations))

    # job_ads_location_count = defaultdict(int)
    # for k, v in tqdm(all_tk_locations.items()):
    #     if v:
    #         date = v[0:7]
    #         job_ads_location_count[date] += 1
    #     else:
    #         job_ads_location_count["No date given"] += 1

    # save_to_s3(
    #     s3,
    #     BUCKET_NAME,
    #     job_ads_location_count,
    #     "outputs/tk_data_analysis/metadata_location/all_tk_location_counts.json",
    # )

