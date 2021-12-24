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

tk_data_path = "inputs/data/textkernel-files/"

all_tk_data_paths = get_s3_data_paths(
    s3, BUCKET_NAME, tk_data_path, file_types=["*.jsonl*"]
)

no_text_job_ids = []
for file_name in tqdm(all_tk_data_paths):
    data = load_s3_data(s3, BUCKET_NAME, file_name)
    for d in data:
    	if not d.get('full_text'):
    		no_text_job_ids.append(d.get('job_id'))

save_to_s3(
        s3,
        BUCKET_NAME,
        no_text_job_ids,
        "outputs/tk_data_analysis_new_method/all_tk_no_full_text.json",
    )