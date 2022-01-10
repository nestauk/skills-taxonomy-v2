"""
We found that 687,715 of the 5 million job adverts were assigned to the expired files
in `get_tk_sample.py`. This means that when the full text is searched for, it isn't available.

Thus, for these job adverts we searched for them again (also in `get_tk_sample.py`) but in 
the not-expired files. We found them all.

For these files, we will supplement the data exported in predict_sentence_class.py with 
these extra skill sentences.


I copied 
s3://skills-taxonomy-v2/outputs/sentence_classifier/data/skill_sentences/2021.10.27/textkernel-files/
into 
s3://skills-taxonomy-v2/outputs/sentence_classifier/data/skill_sentences/2022.01.04/
before this was run (so we don't edit original).

"""

import json
import time
import yaml
import pickle
import os
from fnmatch import fnmatch
import gzip
import logging
from argparse import ArgumentParser
from multiprocessing import Pool
from functools import partial
import itertools
import random
from collections import defaultdict

from tqdm import tqdm
import spacy
import boto3

from skills_taxonomy_v2.pipeline.sentence_classifier.sentence_classifier import (
    SentenceClassifier,
)
from skills_taxonomy_v2.pipeline.sentence_classifier.utils import split_sentence, make_chunks, split_sentence_over_chunk
from skills_taxonomy_v2 import PROJECT_DIR, BUCKET_NAME
from skills_taxonomy_v2.getters.s3_data import load_s3_data, save_to_s3
from skills_taxonomy_v2.pipeline.sentence_classifier.predict_sentence_class import load_model, predict_sentences, get_output_name

logger = logging.getLogger(__name__)


def run_predict(
        s3, data_path, job_ids, sent_classifier, done_job_ids
    ):
    """
    Get predictions from the sample of job adverts already found.
    Don't re-run any of the job ids in done_job_ids.
    """
  
    # Run predictions and save outputs iteratively
    logger.info(f"Loading data from {data_path} ...")
    data = load_s3_data(s3, BUCKET_NAME, data_path)

    job_ids_set = set(job_ids).difference(done_job_ids)

    if len(job_ids_set) == 0:
        logger.info(f"Already processed ...")

    # If the length is 0 you have already processed this file.
    if len(job_ids_set) != 0:

        # Sample of job adverts used for this file
        neccessary_fields = ["full_text", "job_id"]
        data = [
            {field: job_ad.get(field, None) for field in neccessary_fields}
            for job_ad in data
            if job_ad["job_id"] in job_ids_set
        ]

        if data:
            logger.info(f"Splitting sentences ...")
            start_time = time.time()
            with Pool(4) as pool:  # 4 cpus
                chunks = make_chunks(data, 1000)  # chunks of 1000s sentences
                partial_split_sentence = partial(split_sentence_over_chunk, min_length=30)
                # NB the output will be a list of lists, so make sure to flatten after this!
                split_sentence_pool_output = pool.map(partial_split_sentence, chunks)
            logger.info(f"Splitting sentences took {time.time() - start_time} seconds")

            # Flatten and process output into one list of sentences for all documents
            start_time = time.time()
            sentences = []
            job_ids = []
            for chunk_split_sentence_pool_output in split_sentence_pool_output:
                for job_id, s in chunk_split_sentence_pool_output:
                    if s:
                        sentences += s
                        job_ids += [job_id] * len(s)
            logger.info(f"Processing sentences took {time.time() - start_time} seconds")

            if sentences:
                logger.info(f"Transforming skill sentences ...")
                sentences_vec = sent_classifier.transform(sentences)
                pool_sentences_vec = [
                    (vec_ix, [vec]) for vec_ix, vec in enumerate(sentences_vec)
                ]

                logger.info(f"Chunking up sentences ...")
                start_time = time.time()
                # Manually chunk up the data to predict multiple in a pool
                # This is because predict can't deal with massive vectors
                pool_sentences_vecs = []
                pool_sentences_vec = []
                for vec_ix, vec in enumerate(sentences_vec):
                    pool_sentences_vec.append((vec_ix, vec))
                    if len(pool_sentences_vec) > 1000:
                        pool_sentences_vecs.append(pool_sentences_vec)
                        pool_sentences_vec = []
                if len(pool_sentences_vec) != 0:
                    # Add the final chunk if not empty
                    pool_sentences_vecs.append(pool_sentences_vec)
                logger.info(
                    f"Chunking up sentences into {len(pool_sentences_vecs)} chunks took {time.time() - start_time} seconds"
                )

                logger.info(f"Predicting skill sentences ...")
                start_time = time.time()
                with Pool(4) as pool:  # 4 cpus
                    partial_predict_sentences = partial(
                        predict_sentences, sent_classifier=sent_classifier
                    )
                    predict_sentences_pool_output = pool.map(
                        partial_predict_sentences, pool_sentences_vecs
                    )
                logger.info(
                    f"Predicting on {len(sentences)} sentences took {time.time() - start_time} seconds"
                )

                # Process output into one list of sentences for all documents
                logger.info(f"Combining data for output ...")
                start_time = time.time()
                skill_sentences_dict = defaultdict(list)
                for chunk_output in predict_sentences_pool_output:
                    for (sent_ix, pred) in chunk_output:
                        if pred == 1:
                            job_id = job_ids[sent_ix]
                            sentence = sentences[sent_ix]
                            skill_sentences_dict[job_id].append(sentence)
                logger.info(f"Combining output took {time.time() - start_time} seconds")

                return skill_sentences_dict

def parse_arguments(parser):

    parser.add_argument(
        "--config_path",
        help="Path to config file",
        default="skills_taxonomy_v2/config/predict_skill_sentences/2021.10.27.yaml",
    )

    args, unknown = parser.parse_known_args()

    return args


if __name__ == "__main__":

    parser = ArgumentParser()
    args = parse_arguments(parser)

    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    FLOW_ID = "predict_skill_sentences_flow"

    flow_config = config["flows"][FLOW_ID]
    params = flow_config["params"]

    s3 = boto3.resource("s3")

    # The expired replacements
    sample_dict_additional = load_s3_data(
        s3, BUCKET_NAME, "outputs/tk_sample_data/sample_file_locations_expired_replacements.json"
    )
    # sample_dict_additional = [(k, v) for k, v in sample_dict_additional.items()]

    # CAN DELETE   
    manual_to_do = ['semiannual/2020/2020-10-02/jobs_new.13.jsonl.gz',
        'semiannual/2020/2020-10-02/jobs_new.24.jsonl.gz',
        'semiannual/2020/2020-10-02/jobs_new.25.jsonl.gz',
        'semiannual/2020/2020-10-02/jobs_new.26.jsonl.gz',
        'semiannual/2020/2020-10-02/jobs_new.27.jsonl.gz',
        'semiannual/2020/2020-10-02/jobs_new.28.jsonl.gz',
        'semiannual/2020/2020-10-02/jobs_new.32.jsonl.gz',
        'semiannual/2020/2020-10-02/jobs_new.33.jsonl.gz',
        'semiannual/2020/2020-10-02/jobs_new.34.jsonl.gz',
        'semiannual/2021/2021-04-01/jobs_new.16.jsonl.gz',
        'semiannual/2021/2021-04-01/jobs_new.17.jsonl.gz',
        'semiannual/2021/2021-04-01/jobs_new.18.jsonl.gz',
        'semiannual/2021/2021-04-01/jobs_new.15.jsonl.gz',
        'semiannual/2021/2021-04-01/jobs_new.12.jsonl.gz',
        'semiannual/2021/2021-04-01/jobs_new.14.jsonl.gz',
        'semiannual/2021/2021-04-01/jobs_new.13.jsonl.gz',
        'semiannual/2021/2021-04-01/jobs_new.11.jsonl.gz',
        'semiannual/2021/2021-04-01/jobs_new.0.jsonl.gz',
        'semiannual/2021/2021-04-01/jobs_new.10.jsonl.gz',
        'semiannual/2021/2021-04-01/jobs_new.1.jsonl.gz']
    sample_dict_additional = {k:v for k,v in sample_dict_additional.items() if k in manual_to_do}
    

    # Chunk up the files since some have a lot of job ids in
    max_size = 2000

    def chunks(l, n):
        chunked = []
        for i in range(0, len(l), n):
            chunked.append(l[i : i + n])
        return chunked

    sample_dict_additional_new = []
    for file_name, job_id_list in sample_dict_additional.items():
        chunked_job_id_list = chunks(job_id_list, max_size)
        for chunk_job_id_list in chunked_job_id_list:
            sample_dict_additional_new.append((file_name, chunk_job_id_list))
    sample_dict_additional = sample_dict_additional_new

    # Reverse since the first ones are likely to already be processed
    sample_dict_additional.reverse()
        
    sent_classifier, _ = load_model(params["model_config_name"], multi_process=True, batch_size=32)

    edit_dir = "outputs/sentence_classifier/data/skill_sentences/2022.01.04/"

    # Make predictions and save output for data path(s)
    logger.info(f"Running predictions on {len(sample_dict_additional)} data files ...")
    for data_subpath, job_ids in tqdm(sample_dict_additional):
        data_path = os.path.join(params["input_dir"], params["data_dir"], data_subpath)

        # Open up the original skill sentences for this file
        output_file_dir = get_output_name(data_path, "inputs/data/", edit_dir, params["model_config_name"])
        skill_sentences_dict_enhanced = load_s3_data(s3, BUCKET_NAME, output_file_dir)
        done_job_ids = set(skill_sentences_dict_enhanced.keys())

        skill_sentences_dict = run_predict(s3, data_path, job_ids, sent_classifier, done_job_ids)
        if skill_sentences_dict:
            #  Append file with new job ids
            logger.info(f"Original number job adverts with skill sentences was {len(skill_sentences_dict_enhanced)}")
            skill_sentences_dict_enhanced.update(skill_sentences_dict)
            logger.info(f"Now number job adverts with skill sentences is {len(skill_sentences_dict_enhanced)}")
            logger.info(f"Saving data to {output_file_dir} ...")
            save_to_s3(s3, BUCKET_NAME, skill_sentences_dict_enhanced, output_file_dir)

