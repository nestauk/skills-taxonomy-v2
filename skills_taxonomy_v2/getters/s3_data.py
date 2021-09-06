"""
Loading and saving to S3
"""

import json
import pickle
import gzip
from fnmatch import fnmatch
import os
import logging

import pandas as pd
import boto3

logger = logging.getLogger(__name__)


def get_s3_resource():
    s3 = boto3.resource("s3")
    return s3


def get_s3_data_paths(s3, bucket_name, root, file_types=["*.jsonl"]):
    """
    Get all paths to particular file types in a S3 root location

    s3: S3 boto3 resource
    bucket_name: The S3 bucket name
    root: The root folder to look for files in
    file_types: List of file types to look for, or one
    """
    if isinstance(file_types, str):
        file_types = [file_types]

    bucket = s3.Bucket(bucket_name)

    s3_keys = []
    for obj in bucket.objects.all():
        key = obj.key
        if root in key:
            if any([fnmatch(key, pattern) for pattern in file_types]):
                s3_keys.append(key)

    return s3_keys


def save_to_s3(s3, bucket_name, output_var, output_file_dir):

    obj = s3.Object(bucket_name, output_file_dir)

    if fnmatch(output_file_dir, "*.pkl") or fnmatch(output_file_dir, "*.pickle"):
        byte_obj = pickle.dumps(output_var) 
    elif fnmatch(output_file_dir, "*.gz"):
        byte_obj = gzip.compress(json.dumps(output_var))
    else:
        byte_obj = json.dumps(output_var)
    obj.put(Body=byte_obj)

    logger.info(f"Saved to s3://{bucket_name} + {output_file_dir} ...")


def load_s3_data(s3, bucket_name, file_name):
    """
    Load data from S3 location.

    s3: S3 boto3 resource
    bucket_name: The S3 bucket name
    file_name: S3 key to load
    """
    obj = s3.Object(bucket_name, file_name)
    if fnmatch(file_name, "*.jsonl.gz"):
        with gzip.GzipFile(fileobj=obj.get()["Body"]) as file:
            return [json.loads(line) for line in file]
    elif fnmatch(file_name, "*.jsonl"):
        file = obj.get()["Body"].read().decode()
        return [json.loads(line) for line in file]
    elif fnmatch(file_name, "*.json.gz"):
        with gzip.GzipFile(fileobj=obj.get()["Body"]) as file:
            return json.loads(file)
    elif fnmatch(file_name, "*.json"):
        file = obj.get()["Body"].read().decode()
        return json.loads(file)
    elif fnmatch(file_name, "*.csv"):
        return pd.read_csv(os.path.join("s3://" + bucket_name, file_name))
    elif fnmatch(file_name, "*.pkl") or fnmatch(file_name, "*.pickle"):
        file = obj.get()["Body"].read().decode()
        return pickle.loads(file)
    else:
        print(
            'Function not supported for file type other than "*.jsonl.gz", "*.jsonl", or "*.json"'
        )
