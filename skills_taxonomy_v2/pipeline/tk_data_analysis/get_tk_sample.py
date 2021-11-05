"""
Take a sample of the TK job adverts to be used in the pipeline.

Output is a dict of each tk file name and a list of the job ids
within it which are included in the sample.
e.g. {"historical/...0.json": ['6001f8701aeb4072a8eb0cca85535208', ...]}
"""

from skills_taxonomy_v2.getters.s3_data import (
    load_s3_data,
    get_s3_data_paths,
    save_to_s3,
)

from argparse import ArgumentParser
from collections import defaultdict
import random
import os
import yaml

from tqdm import tqdm
import boto3

from skills_taxonomy_v2 import BUCKET_NAME

s3 = boto3.resource("s3")


def parse_arguments(parser):

    parser.add_argument(
        "--config_path",
        help="Path to config file",
        default="skills_taxonomy_v2/config/tk_data_sample/2021.10.25.yaml",
    )

    return parser.parse_args()


if __name__ == "__main__":

    parser = ArgumentParser()
    args = parse_arguments(parser)

    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    FLOW_ID = "get_tk_sample"
    flow_config = config["flows"][FLOW_ID]
    params = flow_config["params"]

    tk_metadata_dir = params["tk_metadata_dir"]

    # Get all the job ids in the job id:file location dictionary
    # from tk metadata (outputted from "get_bulk_metadata.py")
    tk_metadata_paths = get_s3_data_paths(
        s3, BUCKET_NAME, tk_metadata_dir, file_types=["*.json"]
    )

    # There is some duplication in job id, so use unique set
    job_ids = set()
    for tk_metadata_path in tqdm(tk_metadata_paths):
        file_dict = load_s3_data(s3, BUCKET_NAME, tk_metadata_path)
        job_ids.update(set(file_dict.keys()))

    # Take a random sample
    random.seed(params["random_seed"])
    job_ids_sample = random.sample(job_ids, params["sample_size"])

    del job_ids

    # It's quicker to query a set than a list
    job_ids_sample_set = set(job_ids_sample)

    # Output a dict of the job ids sampled from each file.
    # The nuance is that multiple files may have the same job id,
    # so only include the first one when iterating through the files
    # randomly

    sample_locs = defaultdict(list)
    job_ids_seen = set()
    random.seed(params["random_seed"])
    random.shuffle(tk_metadata_paths)

    for tk_metadata_path in tqdm(tk_metadata_paths):
        file_dict = load_s3_data(s3, BUCKET_NAME, tk_metadata_path)
        for job_id, file_name in file_dict.items():
            if (job_id in job_ids_sample_set) and (job_id not in job_ids_seen):
                sample_locs[file_name].append(job_id)
                job_ids_seen.add(job_id)

    print(sum([len(v) for v in sample_locs.values()]) == params["sample_size"])

    save_to_s3(
        s3,
        BUCKET_NAME,
        sample_locs,
        os.path.join(params["output_dir"], "sample_file_locations.json"),
    )
