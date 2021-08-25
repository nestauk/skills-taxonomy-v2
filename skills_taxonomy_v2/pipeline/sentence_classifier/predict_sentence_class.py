"""
Predict whether sentences are skill sentences or not using all the sentences for a file of job adverts.

If you are running this on a whole directory of files it will try to read any jsonl or jsonl.gz in the location,
it will also skip over a file if a 'full_text' key isn't found in the first element of it.

Output will be a dict of the form: {'job_id_1': [('sentence1', [1.5, 1.4 ...]), ('sentence1', [1.5, 1.4 ...])],
                                    'job_id_2': [('sentence1', [1.5, 1.4 ...]), ('sentence1', [1.5, 1.4 ...]}
Where any sentences that were predicted as not-skills are filtered out.

Usage:

python skills_taxonomy_v2/pipeline/sentence_classifier/predict_sentence_class.py --config_path 'skills_taxonomy_v2/config/predict_skill_sentences/2021.08.16.local.sample.yaml'

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

from skills_taxonomy_v2.pipeline.sentence_classifier.utils import (
    split_sentence,
)

from skills_taxonomy_v2 import PROJECT_DIR, BUCKET_NAME


logger = logging.getLogger(__name__)


def load_neccessary(line):
    neccessary_fields = ["full_text", "job_id"]
    line = json.loads(line)
    return {field: line.get(field, None) for field in neccessary_fields}


def load_local_data(file_name):
    """
    Local locations - jsonl or jsonl.gz
    Just load what's needed (job id and full text)
    """
    if fnmatch(file_name, "*.jsonl.gz"):
        with gzip.open(file_name) as file:
            data = [load_neccessary(line) for line in file]
    elif fnmatch(file_name, "*.jsonl"):
        with open(file_name, "r") as file:
            data = [load_neccessary(line) for line in file]
    else:
        data = None

    return data


def load_s3_data(file_name, s3):
    """
    Load from S3 locations - jsonl.gz
    file_name: S3 key
    """
    if fnmatch(file_name, "*.jsonl.gz"):
        obj = s3.Object(BUCKET_NAME, file_name)
        with gzip.GzipFile(fileobj=obj.get()["Body"]) as file:
            data = [load_neccessary(line) for line in file]
    else:
        data = None

    return data


def load_model(config_name):

    # Load sentence classifier trained model and config it came with
    # Be careful here if you change output locations in sentence_classifier.py
    model_dir = f"{config_name.replace('.', '_')}.pkl"
    config_dir = f"skills_taxonomy_v2/config/sentence_classifier/{config_name}.yaml"

    with open(config_dir, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Loading the model
    sent_classifier = SentenceClassifier(
        bert_model_name=config["flows"]["sentence_classifier_flow"]["params"][
            "bert_model_name"
        ],
        multi_process=config["flows"]["sentence_classifier_flow"]["params"][
            "multi_process"
        ],
        max_depth=config["flows"]["sentence_classifier_flow"]["params"]["max_depth"],
        min_child_weight=config["flows"]["sentence_classifier_flow"]["params"][
            "min_child_weight"
        ],
        gamma=config["flows"]["sentence_classifier_flow"]["params"]["gamma"],
        colsample_bytree=config["flows"]["sentence_classifier_flow"]["params"][
            "colsample_bytree"
        ],
        subsample=config["flows"]["sentence_classifier_flow"]["params"]["subsample"],
        reg_alpha=config["flows"]["sentence_classifier_flow"]["params"]["reg_alpha"],
        max_iter=config["flows"]["sentence_classifier_flow"]["params"]["max_iter"],
        solver=config["flows"]["sentence_classifier_flow"]["params"]["solver"],
        penalty=config["flows"]["sentence_classifier_flow"]["params"]["penalty"],
        class_weight=config["flows"]["sentence_classifier_flow"]["params"][
            "class_weight"
        ],
        C=config["flows"]["sentence_classifier_flow"]["params"]["C"],
        probability_threshold=config["flows"]["sentence_classifier_flow"]["params"][
            "probability_threshold"
        ],
    )

    sent_classifier.load_model(model_dir)

    return sent_classifier, config


def predict_sentences(sent_classifier, sentences):

    # Predict
    sentences_vec = sent_classifier.transform(sentences)
    sentences_pred = sent_classifier.predict(sentences_vec)

    return sentences_pred, sentences_vec


def save_outputs(skill_sentences_dict, output_file_dir):

    directory = os.path.dirname(output_file_dir)

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(output_file_dir, "w") as file:
        json.dump(skill_sentences_dict, file)


def save_outputs_to_s3(s3, skill_sentences_dict, output_file_dir):

    obj = s3.Object(BUCKET_NAME, output_file_dir)

    obj.put(Body=json.dumps(skill_sentences_dict))


def load_outputs(file_name):

    with open(file_name, "r") as file:
        output = json.load(file)

    return output


def get_output_name(data_path, input_dir, output_dir, model_config_name):

    # For naming the output, remove the input dir folder structure
    data_dir = os.path.relpath(data_path, input_dir)

    # Put the output in a folder with a similar naming structure to the input
    # Should work for .jsonl and .jsonl.gz
    output_file_dir = os.path.join(
        output_dir, data_dir.split(".json")[0] + "_" + model_config_name + ".json"
    )

    return output_file_dir


def get_local_data_paths(root, pattern="*.jsonl*"):

    if os.path.isdir(root):
        # If data_dir to predict on is a directory
        # get all the names of jsonl or jsonl.gz files in this location
        data_paths = []
        for path, subdirs, files in os.walk(root):
            for name in files:
                if fnmatch(name, pattern):
                    data_paths.append(os.path.join(path, name))
    else:
        data_paths = [root]

    return data_paths


def get_s3_data_paths(bucket, root, pattern="*.jsonl*"):
    s3_keys = []
    for obj in bucket.objects.all():
        key = obj.key
        if root in key:
            if fnmatch(key, pattern):
                s3_keys.append(key)

    return s3_keys


def run_predict_sentence_class(
    input_dir,
    data_dir,
    model_config_name,
    output_dir,
    data_local=True,
    sample_data_paths=True,
    random_seed=42,
    sample_size=100,
):
    """
    Given the input dir, get predictions and save out for every relevant file in the dir.
    data_local : True if the data is stored localle, False if you are reading from S3
    """

    sent_classifier, _ = load_model(model_config_name)
    nlp = spacy.load("en_core_web_sm")

    root = os.path.join(input_dir, data_dir)

    if data_local:
        data_paths = get_local_data_paths(root)
    else:
        s3 = boto3.resource("s3")
        bucket = s3.Bucket(BUCKET_NAME)
        data_paths = get_s3_data_paths(bucket, root)

    if sample_data_paths:
        random.seed(random_seed)
        data_paths = random.sample(data_paths, sample_size)

    # Make predictions and save output for data path(s)
    logger.info(f"Running predictions on {len(data_paths)} data files ...")
    for data_path in data_paths:
        # Run predictions and save outputs iteratively
        logger.info(f"Loading data from {data_path} ...")
        if data_local:
            data = load_local_data(data_path)
        else:
            data = load_s3_data(data_path, s3)

        if data:
            output_file_dir = get_output_name(
                data_path, input_dir, output_dir, model_config_name
            )
            logger.info(f"Splitting sentences ...")
            start_time = time.time()
            with Pool(4) as pool:  # 4 cpus
                partial_split_sentence = partial(split_sentence, nlp=nlp, min_length=30)
                split_sentence_pool_output = pool.map(partial_split_sentence, data)
            logger.info(f"Splitting sentences took {time.time() - start_time} seconds")

            # Process output into one list of sentences for all documents
            sentences = []
            job_ids = []
            for i, (job_id, s) in enumerate(split_sentence_pool_output):
                if s:
                    sentences += s
                    job_ids += [job_id] * len(s)

            if sentences:
                logger.info(f"Predicting skill sentences ...")
                start_time = time.time()
                sentences_pred, _ = predict_sentences(sent_classifier, sentences)
                logger.info(
                    f"Predicting on {len(sentences)} sentences took {time.time() - start_time} seconds"
                )

                logger.info(f"Combining data for output ...")
                start_time = time.time()
                skill_sentences_dict = defaultdict(list)
                for job_id, sentence, pred in list(
                    zip(job_ids, sentences, sentences_pred)
                ):
                    if pred == 1:
                        skill_sentences_dict[job_id].append(sentence)
                logger.info(f"Combining output took {time.time() - start_time} seconds")

                logger.info(f"Saving data to {output_file_dir} ...")
                if data_local:
                    save_outputs(skill_sentences_dict, output_file_dir)
                else:
                    save_outputs_to_s3(s3, skill_sentences_dict, output_file_dir)


def parse_arguments(parser):

    parser.add_argument(
        "--config_path",
        help="Path to config file",
        default="skills_taxonomy_v2/config/predict_skill_sentences/2021.08.16.local.sample.yaml",
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

    if not params["sample_data_paths"]:
        # If you don't want to sample the data you can set these to None
        params["random_seed"] = None
        params["sample_size"] = None

    if params["data_local"]:
        input_dir = os.path.join(PROJECT_DIR, params["input_dir"])
        output_dir = os.path.join(PROJECT_DIR, params["output_dir"])
    else:
        # If we are pulling the data from S3 we don't want the paths to join with our local project_dir
        input_dir = params["input_dir"]
        output_dir = params["output_dir"]

    run_predict_sentence_class(
        input_dir,
        params["data_dir"],
        params["model_config_name"],
        output_dir,
        data_local=params["data_local"],
        sample_data_paths=params["sample_data_paths"],
        random_seed=params["random_seed"],
        sample_size=params["sample_size"],
    )
