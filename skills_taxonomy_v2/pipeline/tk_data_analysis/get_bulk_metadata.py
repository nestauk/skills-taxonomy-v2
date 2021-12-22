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
from collections import defaultdict

from skills_taxonomy_v2 import BUCKET_NAME

from skills_taxonomy_v2.getters.s3_data import get_s3_data_paths, save_to_s3


s3 = boto3.resource("s3")

if __name__ == "__main__":

    tk_data_path = "inputs/data/textkernel-files/"
    output_dir = "outputs/tk_data_analysis_new_method/"

    all_tk_data_paths = get_s3_data_paths(
        s3, BUCKET_NAME, tk_data_path, file_types=["*.jsonl*"]
    )

    file_num = 0
    count_tk_files = 0
    job_id_file_list = []
    job_id_date_list = []
    job_id_meta_list = []
    job_id_job_list = []
    job_id_location_list = []
    all_tk_date_count = defaultdict(int)
    all_tk_region_count = defaultdict(int)
    all_tk_subregion_count = defaultdict(int)

    for file_name in tqdm(all_tk_data_paths):
        data = load_s3_data(s3, BUCKET_NAME, file_name)
        for d in data:
            # Save out as little info as possible to make file smaller
            job_id_file_list.append((d["job_id"], file_name.split(tk_data_path)[1]))
            job_id_date_list.append((d["job_id"], [
                d.get("date"),
                d.get("expiration_date"),
            ]))
            if d.get("date"):
                all_tk_date_count[d.get("date")] += 1
            else:
                all_tk_date_count["Not given"] += 1

            job_id_meta_list.append((d["job_id"], [
                d.get("source_website"),
                d.get("language"),
            ]))
            organization_industry = d.get("organization_industry")
            job_id_job_list.append((d["job_id"], [
                d.get("job_title"),
                organization_industry.get("label")
                if organization_industry
                else None,
            ]))
            region = d.get("region")
            subregion = d.get("subregion")
            job_id_location_list.append((d["job_id"], [
                d.get("location_name"),
                d.get("location_coordinates"),
                region.get("label") if region else None,
                subregion.get("label") if subregion else None,
            ]))
            if region:
                all_tk_region_count[region.get("label")] += 1
            else:
                all_tk_region_count["Not given"] += 1
            if subregion:
                all_tk_subregion_count[subregion.get("label")] += 1
            else:
                all_tk_subregion_count["Not given"] += 1

        count_tk_files += 1
        if count_tk_files == 50:
            print("Saving data ...")
            save_to_s3(
                s3,
                BUCKET_NAME,
                job_id_file_list,
                os.path.join(output_dir, f"metadata_file/{file_num}.json"),
            )
            save_to_s3(
                s3,
                BUCKET_NAME,
                job_id_date_list,
                os.path.join(output_dir, f"metadata_date/{file_num}.json"),
            )
            save_to_s3(
                s3,
                BUCKET_NAME,
                job_id_meta_list,
                os.path.join(output_dir, f"metadata_meta/{file_num}.json"),
            )
            save_to_s3(
                s3,
                BUCKET_NAME,
                job_id_job_list,
                os.path.join(output_dir, f"metadata_job/{file_num}.json"),
            )
            save_to_s3(
                s3,
                BUCKET_NAME,
                job_id_location_list,
                os.path.join(output_dir, f"metadata_location/{file_num}.json"),
            )
            file_num += 1
            count_tk_files = 0
            job_id_file_list = []
            job_id_date_list = []
            job_id_meta_list = []
            job_id_job_list = []
            job_id_location_list = []
    
    save_to_s3(
        s3,
        BUCKET_NAME,
        all_tk_date_count,
        os.path.join(output_dir, f"metadata_date/all_tk_date_count.json"),
    )
    save_to_s3(
        s3,
        BUCKET_NAME,
        all_tk_region_count,
        os.path.join(output_dir, f"metadata_location/all_tk_region_count.json"),
    )
    save_to_s3(
        s3,
        BUCKET_NAME,
        all_tk_subregion_count,
        os.path.join(output_dir, f"metadata_location/all_tk_subregion_count.json"),
    )

