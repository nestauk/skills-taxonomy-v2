"""
Take a sample of the TK job adverts to be used in the pipeline.

Output is a dict of each tk file name and a list of the job ids
within it which are included in the sample.
e.g. {"historical/...0.json": ['6001f8701aeb4072a8eb0cca85535208', ...]}

Note: this script isn't the most efficient since we wanted to utilise parts of the sample
already processed through the pipeline, so needed to adapt the original sample
rather than re-sampling everything without the expired files
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


def save_final_sample_list(s3, BUCKET_NAME):
    """
    Combine the original sample without the jobs_expired files, 
    with the replacements samples
    to have a dict of {file_name: list of job ids} for the final sample
    """
        
    # The 5 million sample (inc expired)
    sample_locs = load_s3_data(
        s3, BUCKET_NAME, "outputs/tk_sample_data/sample_file_locations.json"
    )

    final_sample_dict = defaultdict(list)
    for file_name, job_ids in sample_locs.items():
        if "jobs_expired" not in file_name:
            for job_id in job_ids:
                final_sample_dict[file_name].append(job_id)

    sample_locs_reboot = load_s3_data(
        s3, BUCKET_NAME, "outputs/tk_sample_data/sample_file_locations_expired_replacements.json"
        )

    for file_name, job_ids in sample_locs_reboot.items():
        for job_id in job_ids:
            final_sample_dict[file_name].append(job_id)

    save_to_s3(
            s3,
            BUCKET_NAME,
            final_sample_dict,
            "outputs/tk_sample_data/14.01.22.sample_file_locations.json",
        )

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

    # Now with the job ids from the expired files, create another dict to get these
    # job ids from not-expired files:
    job_ids_expired = []
    for sample_loc, job_ids in tqdm(sample_locs.items()):
        if "jobs_expired" in sample_loc:
            job_ids_expired += job_ids

    job_ids_expired = set(job_ids_expired)

    print(len(job_ids_expired))

    # Get the same number of expired job ids from these location of files
    expired_locs_nums = defaultdict(int)
    for k, job_ids in sample_locs.items():
        if "jobs_expired" in k:
            expired_locs_nums["/".join(k.split('/')[0:3])] += len(job_ids)


    def get_job_ids(expired_loc, num_job_ids, sample_job_ids):
        job_ids_replacement = set()
        for tk_metadata_path in tqdm(tk_metadata_paths):
            file_dict = load_s3_data(s3, BUCKET_NAME, tk_metadata_path)
            for job_id, file_name in file_dict.items():
                if "jobs_expired" not in file_name:
                    file_loc = "/".join(file_name.split('/')[0:3])
                    if file_loc == expired_loc:
                        job_ids_replacement.add(job_id)
        # Remove any which are in the sample already
        job_ids_replacement_unique = job_ids_replacement.difference(sample_job_ids)
        random.seed(params["random_seed"])
        job_ids_sample_replacement = set(random.sample(job_ids_replacement_unique, num_job_ids))
        return job_ids_sample_replacement

    sample_job_ids = []
    for sample_loc, job_ids in tqdm(sample_locs.items()):
        if "jobs_expired" not in sample_loc:
            sample_job_ids += job_ids

    sample_job_ids = set(sample_job_ids)
    job_ids_sample_replacement = set()
    for expired_loc, num_job_ids in expired_locs_nums.items():
        print(expired_loc)
        job_ids_sample_replacement.update(get_job_ids(expired_loc, num_job_ids, sample_job_ids))
        sample_job_ids.update(job_ids_sample_replacement)

    job_ids_seen = set()
    sample_locs_reboot = defaultdict(list)
    for tk_metadata_path in tqdm(tk_metadata_paths):
        file_dict = load_s3_data(s3, BUCKET_NAME, tk_metadata_path)
        for job_id, file_name in file_dict.items():
            if "jobs_expired" not in file_name:
                if (job_id in job_ids_sample_replacement) and (job_id not in job_ids_seen):
                    sample_locs_reboot[file_name].append(job_id)
                    job_ids_seen.add(job_id)


    print(sum([len(v) for v in sample_locs_reboot.values()]) == len(job_ids_expired))
    print(len(sample_job_ids))
    print(len(set(sample_job_ids)))

    save_to_s3(
            s3,
            BUCKET_NAME,
            sample_locs_reboot,
            os.path.join(params["output_dir"], "sample_file_locations_expired_replacements.json"),
        )

    # Get and save a final combined list of the sampled job adverts 
    save_final_sample_list(s3, BUCKET_NAME)
