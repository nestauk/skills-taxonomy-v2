"""
python skills_taxonomy_v2/pipeline/sentence_classifier/predict_sentence_class_extras/flow.py --environment=conda --datastore=s3 run --max-num-splits 300
"""
from typing import List

from metaflow import FlowSpec, step, conda, batch, project, Parameter, current, S3, conda_base, pip

import logging
import json

try:  # Hack for type-hints on attributes
    from pandas import DataFrame
except ImportError:
    pass

from skills_taxonomy_v2 import BUCKET_NAME
from skills_taxonomy_v2.getters.s3_data import load_s3_data, save_to_s3

logger = logging.getLogger(__name__)

@project(name="predict_sentence_class_extras")
class PredictSkillSentsExtra(FlowSpec):

    @pip(libraries={"pyyaml": "5.4.1", "boto3":"1.18.0"})
    @step
    def start(self):
        """Load new job ids to get skill sentence predictions from"""
        # import os
        # os.system('pip install yaml') 

        import yaml
        import boto3

        s3 = boto3.resource("s3")

        config_path = "skills_taxonomy_v2/config/predict_skill_sentences/2021.10.27.yaml"
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        FLOW_ID = "predict_skill_sentences_flow"

        flow_config = config["flows"][FLOW_ID]
        self.params = flow_config["params"]

        # The expired replacements
        self.sample_dict_additional = load_s3_data(
            s3, BUCKET_NAME, "outputs/tk_sample_data/sample_file_locations_expired_replacements.json"
        )
        # Turn dict to list of tuples for batching
        self.sample_dict_additional = [(k, v) for k, v in self.sample_dict_additional.items()]

        # self.sample_dict_additional = []
        # for k, job_id_list in self.sample_dict_additional.items():
        #     for job_id in job_id_list:
        #         self.sample_dict_additional.append((k, job_id))

        # Reverse since the first ones are likely to already be processed
        self.sample_dict_additional.reverse()

        # self.sample_dict_additional = self.sample_dict_additional[0:2] ## REMOVE THIS
        files_left = [
            'historical/2019/2019-11-14/jobs.38.jsonl.gz',
            'semiannual/2020/2020-10-02/jobs_new.13.jsonl.gz',
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
            'semiannual/2021/2021-04-01/jobs_new.18.jsonl.gz'
        ]
        self.sample_dict_additional = [i for i in self.sample_dict_additional if i[0] in files_left]## REMOVE THIS

        self.edit_dir = "outputs/sentence_classifier/data/skill_sentences/2022.01.04/"

        print(f"Running predictions on {len(self.sample_dict_additional)} data files ...")
        # Get batching ready
        self.next(self.process_sentences, foreach="sample_dict_additional")

    @batch(
        queue="job-queue-GPU-nesta-metaflow",
        image="metaflow-pytorch",
        # Queue gives p3.2xlarge, with:
        gpu=1,
        memory=61000,
        cpu=8,
    )
    @conda(
        libraries={
            "boto3": "1.18.0",
            "spacy": "3.0.0",
            "py-xgboost": "1.4.0",

        },
        python="3.8",
    )
    @pip(libraries={"pyyaml": "5.4.1", "sentence-transformers":"1.2.0"})
    @step
    def process_sentences(self):

        # In batch there will be the error "OSError: mysql_config not found", so do:
        # import subprocess
        # subprocess.run(["sudo apt-get install libmysqlclient-dev"])

        import os
        import json
        import boto3
        import yaml

        # In batch there will be the error "OSError: mysql_config not found", so do:
        # os.system('sudo apt-get install libmysqlclient-dev') 

        from skills_taxonomy_v2.pipeline.sentence_classifier.predict_sentence_class_extras.utils import run_predict
        from skills_taxonomy_v2.pipeline.sentence_classifier.predict_sentence_class import get_output_name, load_model

        s3 = boto3.resource("s3")

        print(f"Loading model ...")
        self.sent_classifier, _ = load_model(self.params["model_config_name"], multi_process=True, batch_size=32)

        data_subpath = self.input[0]
        job_ids = self.input[1]

        data_path = os.path.join(self.params["input_dir"], self.params["data_dir"], data_subpath)

        # Open up the original skill sentences for this file
        self.output_file_dir = get_output_name(data_path, "inputs/data/", self.edit_dir, self.params["model_config_name"])
        print(f"Loading {self.output_file_dir} ...")
        skill_sentences_dict_enhanced = load_s3_data(s3, BUCKET_NAME, self.output_file_dir)
        done_job_ids = set(skill_sentences_dict_enhanced.keys())
        job_ids_set = set(job_ids).difference(done_job_ids)
        print(f"Processing for {len(job_ids_set)} job ids ...")

        # job_ids_set = set(list(job_ids_set)[0:10]) ## REMOVE THIS

        if len(job_ids_set) != 0:
            skill_sentences_dict = run_predict(s3, data_path, job_ids_set, self.sent_classifier)
            if skill_sentences_dict:
                #  Append file with new job ids
                print(f"Original number job adverts with skill sentences was {len(skill_sentences_dict_enhanced)}")
                skill_sentences_dict_enhanced.update(skill_sentences_dict)
                print(f"Now number job adverts with skill sentences is {len(skill_sentences_dict_enhanced)}")
                print(f"Saving data to {self.output_file_dir} ...")
                save_to_s3(s3, BUCKET_NAME, skill_sentences_dict_enhanced, self.output_file_dir)
                del skill_sentences_dict_enhanced

        self.next(self.join)

    @step
    def join(self, inputs):
        self.files_saved = [i.output_file_dir for i in inputs]
        print(f"{len(self.files_saved)} files processed")

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    PredictSkillSentsExtra()