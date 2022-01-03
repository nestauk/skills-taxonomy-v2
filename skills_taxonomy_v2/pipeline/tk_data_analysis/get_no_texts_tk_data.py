"""
Check all the files for 2020/21 - do they have a text field?
Save out every job id that ever has no text field
"""

from tqdm import tqdm
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import boto3

from skills_taxonomy_v2.getters.s3_data import get_s3_data_paths, load_s3_data, save_to_s3
from skills_taxonomy_v2 import BUCKET_NAME

s3 = boto3.resource("s3")
 
# 1. Get counts when there is the full text field
# 2. Get job ids when there is no full text field

tk_data_path = "inputs/data/textkernel-files/"

all_tk_data_paths = get_s3_data_paths(
    s3, BUCKET_NAME, tk_data_path, file_types=["*.jsonl*"]
)

job_dates_fulltext_count = defaultdict(int)
job_region_fulltext_count = defaultdict(int)
job_dates_nofulltext_count = defaultdict(int)
job_dates_all_length = defaultdict(list)
no_text_job_ids = []
for file_name in tqdm(all_tk_data_paths):
    data = load_s3_data(s3, BUCKET_NAME, file_name)
    for d in data:
        if d.get('full_text'):
            if d.get("date"):
                job_dates_fulltext_count[d.get("date")] += 1
                job_dates_all_length[d.get("date")].append(len(d.get('full_text')))
            else:
                job_dates_fulltext_count["Not given"] += 1
                job_dates_all_length["Not given"].append(len(d.get('full_text')))
            region = d.get("region")
            if region:
                region_label = region.get("label")
                if region_label:
                    job_region_fulltext_count[region_label] += 1
        else:
            no_text_job_ids.append(d.get('job_id'))
            if d.get("date"):
                job_dates_nofulltext_count[d.get("date")] += 1
                job_dates_all_length[d.get("date")].append(0)
            else:
                job_dates_nofulltext_count["Not given"] += 1
                job_dates_all_length["Not given"].append(0)

save_to_s3(
        s3,
        BUCKET_NAME,
        job_dates_fulltext_count,
        f"outputs/tk_data_analysis_new_method/metadata_date/tk_dates_count_got_full_text.json",
    )
save_to_s3(
        s3,
        BUCKET_NAME,
        job_region_fulltext_count,
        f"outputs/tk_data_analysis_new_method/metadata_location/tk_regions_count_got_full_text.json",
    )
save_to_s3(
        s3,
        BUCKET_NAME,
        no_text_job_ids,
        "outputs/tk_data_analysis_new_method/all_tk_no_full_text.json",
    )
save_to_s3(
        s3,
        BUCKET_NAME,
        job_dates_nofulltext_count,
        f"outputs/tk_data_analysis_new_method/metadata_date/tk_dates_count_no_full_text.json",
    )
save_to_s3(
        s3,
        BUCKET_NAME,
        job_dates_all_length,
        f"outputs/tk_data_analysis_new_method/metadata_date/tk_dates_all_length.json",
    )
