"""
python skills_taxonomy_v2/pipeline/skills_extraction/metaflow/flow.py --environment=conda --datastore=s3 run
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
from skills_taxonomy_v2.getters.s3_data import load_s3_data, save_to_s3, get_s3_data_paths
from skills_taxonomy_v2.pipeline.skills_extraction.metaflow.utils import get_custom_stopwords_list

logger = logging.getLogger(__name__)

@project(name="skill_sentence_embeddings")
class SkillsSentenceEmbeddings(FlowSpec):

    @pip(libraries={"pyyaml": "5.4.1", "boto3":"1.18.0"})
    @step
    def start(self):

        import os
        import random

        import yaml
        import boto3
        from toolz.itertoolz import partition

        s3 = boto3.resource("s3")

        config_path = "skills_taxonomy_v2/config/skills_extraction/2022.01.14.yaml"
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        FLOW_ID = "word_embeddings_flow"

        flow_config = config["flows"][FLOW_ID]
        self.params = flow_config["params"]
        self.skill_sentences_dir = self.params["skill_sentences_dir"]
        self.token_len_threshold = self.params["token_len_threshold"]
        # Use config file name as a date stamp for the output dir
        self.output_dir = os.path.join(
            self.params["output_dir"], os.path.basename(config_path).replace(".yaml", "")
        )

        self.custom_stopwords = get_custom_stopwords_list()

        # Get data paths in the location
        data_paths = get_s3_data_paths(
            s3, BUCKET_NAME, self.skill_sentences_dir, file_types=["*.json"]
        )
        # Random shuffle, since the later ones might be bigger than the earlier ones
        # so chunks will be unequal
        random.seed(42)
        random.shuffle(data_paths)

        # Chunk up for Batch
        self.data_paths_chunked = list(partition(50, data_paths))

        # For loop through each data path
        print(f"Running predictions on {len(data_paths)} data files in {len(self.data_paths_chunked)} batches ...")

        # Get batching ready
        self.next(self.embed_sentences, foreach="data_paths_chunked")

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
            "spacy": "3.0.0"
        },
        python="3.8",
    )
    # @pip(libraries={"pyyaml": "5.4.1", "sentence-transformers":"1.2.0"})#, "nltk": "3.6.2"})
    @step
    def embed_sentences(self):

        import os
        import json
        import time

        import boto3
        import yaml
        import nltk
        from nltk.corpus import stopwords
        import numpy as np
        import spacy
        from torch import cuda

        from sentence_transformers import SentenceTransformer
        from skills_taxonomy_v2.pipeline.skills_extraction.get_sentence_embeddings_utils import process_sentence_mask
        from skills_taxonomy_v2.pipeline.sentence_classifier.utils import verb_features

        s3 = boto3.resource("s3")

        print(f"Loading SentenceTransformer model ...")
        bert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        bert_model.max_seq_length = 512
        
        os.system('python -m spacy download en_core_web_sm') 
        nlp = spacy.load("en_core_web_sm")

        print(f"Running sentence embeddings for {len(self.input)} files")
        for data_path_i, data_path in enumerate(self.input):
            print(
                f"Loading data for {data_path} ({data_path_i} of {len(self.input)}) ..."
            )
            data = load_s3_data(s3, BUCKET_NAME, data_path)

            print(f"Processing {len(data)} job adverts...")
            start_time = time.time()
            # For each sentence mask out stop words, proper nouns etc.
            masked_sentences = []
            sentence_job_ids = []
            sentence_hashes = []
            original_sentences = {}

            for job_id, sentences in data.items():
                for sentence in sentences:
                    masked_sentence = process_sentence_mask(
                        sentence,
                        nlp,
                        bert_model,
                        self.token_len_threshold,
                        stopwords=stopwords.words(),
                        custom_stopwords=self.custom_stopwords,
                    )
                    if masked_sentence.replace("[MASK]", "").replace(" ", ""):
                        # Don't include sentence if it only consists of masked words
                        masked_sentences.append(masked_sentence)
                        sentence_job_ids.append(job_id)
                        # Keep a record of the original sentence via a hashed id
                        original_sentence_id = hash(sentence)
                        sentence_hashes.append(original_sentence_id)
                        original_sentences[original_sentence_id] = sentence
            del data
            print(f"Processing sentences took {time.time() - start_time} seconds")

            print(f"Getting embeddings for {len(masked_sentences)} sentences...")
            start_time = time.time()
            # Find sentence embeddings in bulk for all masked sentences

            X_vec = bert_model.encode(masked_sentences, show_progress_bar=True)
            masked_sentence_embeddings = np.hstack((X_vec, verb_features(masked_sentences)))

            output_tuple_list = [
                (job_id, sent_id, sent, emb.tolist())
                for job_id, sent_id, sent, emb in zip(
                    sentence_job_ids,
                    sentence_hashes,
                    masked_sentences,
                    masked_sentence_embeddings,
                )
            ]
            del sentence_job_ids
            del sentence_hashes
            del masked_sentences
            del masked_sentence_embeddings
            print(f"Getting embeddings took {time.time() - start_time} seconds")
            
            # Save the output in a folder with a similar naming structure to the input
            data_dir = os.path.relpath(data_path, self.skill_sentences_dir)
            self.output_file_dir = os.path.join(
                self.output_dir, data_dir.split(".json")[0] + "_embeddings.json"
            )
            save_to_s3(s3, BUCKET_NAME, output_tuple_list, self.output_file_dir)

            sent_id_dir = os.path.join(
                self.output_dir, data_dir.split(".json")[0] + "_original_sentences.json"
            )
            save_to_s3(s3, BUCKET_NAME, original_sentences, sent_id_dir)

            print(f"Saved output to {self.output_file_dir}")
            del output_tuple_list
            del original_sentences

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
    SkillsSentenceEmbeddings()
