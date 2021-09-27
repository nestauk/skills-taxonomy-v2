# File: pipeline/sentence_classifier.py

"""Module for BertVectorizer and SentenceClassifier class.

Usage:
python skills_taxonomy_v2/pipeline/sentence_classifier/sentence_classifier.py --yaml_file_name 2021.08.16

"""
# ---------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sentence_transformers import SentenceTransformer
import tqdm as tqdm
import spacy
import numpy as np
from s3fs.core import S3FileSystem

import json
import random
from collections import Counter
import re
from argparse import ArgumentParser
import pickle
import os
import yaml
import time

from skills_taxonomy_v2.pipeline.sentence_classifier.utils import (
    text_cleaning,
    load_training_data_from_s3,
    verb_features,
)

from skills_taxonomy_v2 import PROJECT_DIR, BUCKET_NAME

# ---------------------------------------------------------------------------------
# %%
os.environ["TOKENIZERS_PARALLELISM"] = "false"

S3_PATH = "outputs/sentence_classifier/models/"

# %%


class BertVectorizer:
    """
    Use a pretrained transformers model to embed sentences.
    In this form so it can be used as a step in the pipeline.
    """

    def __init__(
        self,
        bert_model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
        multi_process=True,
    ):
        self.bert_model_name = bert_model_name
        self.multi_process = multi_process

    def fit(self, *_):
        self.bert_model = SentenceTransformer(self.bert_model_name)
        self.bert_model.max_seq_length = 512
        return self

    def transform(self, texts):
        print(f"Getting embeddings for {len(texts)} texts ...")
        t0 = time.time()
        if self.multi_process:
            print(".. with multiprocessing")
            pool = self.bert_model.start_multi_process_pool()
            self.embedded_x = self.bert_model.encode_multi_process(texts, pool)
            self.bert_model.stop_multi_process_pool(pool)
        else:
            self.embedded_x = self.bert_model.encode(texts, show_progress_bar=True)
        print(f"Took {time.time() - t0} seconds")
        return self.embedded_x


# %%
class SentenceClassifier:
    """
    A class the train/save/load/predict a classifier to predict whether
    a sentence contains skills or not.
    ...
    Attributes
    ----------
    test_size : float (default 0.15)
    split_random_seed : int (default 22)
    log_reg_max_iter: int (default 1000)
    Methods
    -------
    preprocess_text(training_data)
            Clean sentences to mask numbers, remove punctuation, ignore small sentences,
            lower
    split_data(training_data)
            Split the training data (list of pairs of text-label) into test/train sets
    fit_transform(X)
            Load the pretrained BERT model and transform X. Stack verb features. 
    transform(X)
            Transform X uses already loaded BERT model. Stack verb features. 
    fit(X_vec, y)
            Fit the classifier to vectorized X
    predict(X_vec)
            Predict classes from already vectorized text
    predict_transform(X)
            Transform then predict classes from text
    evaluate(y, y_pred)
    save_model(file_name)
    load_model(file_name)
    """

    def __init__(
        self,
        split_random_seed=22,
        test_size=0.15,
        bert_model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
        multi_process=True,
        max_depth=7,
        min_child_weight=1,
        gamma=0.0,
        colsample_bytree=0.8,
        subsample=0.8,
        reg_alpha=0.001,
        max_iter=1000,
        solver="liblinear",
        penalty="l2",
        class_weight="balanced",
        C=1.0,
        probability_threshold=0.67,
    ):
        self.split_random_seed = split_random_seed
        self.test_size = test_size
        self.bert_model_name = bert_model_name
        self.bert_model_name = bert_model_name
        self.multi_process = multi_process
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.colsample_bytree = colsample_bytree
        self.subsample = subsample
        self.reg_alpha = reg_alpha
        self.max_iter = max_iter
        self.solver = solver
        self.penalty = penalty
        self.class_weight = class_weight
        self.C = C
        self.probability_threshold = probability_threshold

    def preprocess_text(self, training_data):

        print(
            f"before removing short sents, the number of +ve and -ve labels are: {Counter([label[1] for label in training_data])}"
        )
        clean_training_data = []
        for sent in training_data:
            clean_sent = text_cleaning(sent[0])
            if len(clean_sent) > 30:
                clean_training_data.append((clean_sent, sent[1]))
        print(
            f"after removing short sents, the number of +ve and -ve labels are: {Counter([label[1] for label in clean_training_data])}"
        )

        return clean_training_data

    def split_data(self, training_data, verbose=False):

        training_data = self.preprocess_text(training_data)  # preprocess sentences
        X = [t[0] for t in training_data]
        y = [t[1] for t in training_data]
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            stratify=y,
            test_size=self.test_size,
            random_state=self.split_random_seed,
        )

        if verbose:
            print(f"Size of training data: {len(y_train)}")
            print(f"Size of test data: {len(y_test)}")
            print(f"Counter of training data classes: {Counter(y_train)}")
            print(f"Counter of test data classes: {Counter(y_test)}")
        return X_train, X_test, y_train, y_test

    def load_bert(self):
        self.bert_vectorizer = BertVectorizer(
            bert_model_name=self.bert_model_name, multi_process=self.multi_process
        )
        self.bert_vectorizer.fit()

    def fit_transform(self, X):

        # Load BERT models and transform X
        self.load_bert()
        X_vec = self.bert_vectorizer.transform(X)
        # add verb features
        X_stack = np.hstack((X_vec, verb_features(X)))
        print("stacked!")

        return X_stack

    def transform(self, X):
        X_vec = self.bert_vectorizer.transform(X)
        return np.hstack((X_vec, verb_features(X)))

    def fit(self, X_stack, y):

        xgb = XGBClassifier(
            eval_metric="mlogloss",
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            colsample_bytree=self.colsample_bytree,
            subsample=self.subsample,
            reg_alpha=self.reg_alpha,
            use_label_encoder=False,
        )
        xgb.fit(X_stack, y)
        lr = LogisticRegression(
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
        )
        lr.fit(X_stack, y)

        self.classifier = Pipeline(
            [
                ("scaler", MinMaxScaler()),
                (
                    "classifier",
                    VotingClassifier(
                        estimators=[("xgb", xgb), ("lr", lr)], voting="soft"
                    ),
                ),
            ]
        )
        self.classifier.fit(X_stack, y)

    def predict(self, X_stack):
        probs = self.classifier.predict_proba(X_stack)
        return [
            int(np.where(prob[1] >= self.probability_threshold, 1, 0)) for prob in probs
        ]

    def predict_transform(self, X):
        X_stack = self.transform(X)
        return self.predict(X_stack)

    def evaluate(self, y, y_pred, verbose=True):
        class_rep = classification_report(y, y_pred, output_dict=True)
        if verbose:
            print(classification_report(y, y_pred))
            print(confusion_matrix(y, y_pred))
        return class_rep

    def save_model(self, file_name):
        s3_file = S3FileSystem()
        bucket = BUCKET_NAME
        key = S3_PATH + file_name
        pickle.dump(self.classifier, s3_file.open(f"s3://{bucket}/{key}", "wb"))

    def load_model(self, file_name):
        s3_file = S3FileSystem()
        bucket = BUCKET_NAME
        key = S3_PATH + file_name
        self.classifier = pickle.load(s3_file.open("{}/{}".format(BUCKET_NAME, key)))

        # Load BERT models
        self.load_bert()

        return self.classifier


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--yaml_file_name",
        help="Name of config file from skills_taxonomy_v2/config/sentence_classifier to be used",
        default="2021.08.16",
    )

    args, unknown = parser.parse_known_args()

    # Load specific config file
    yaml_file_name = args.yaml_file_name
    fname = os.path.join(
        "skills_taxonomy_v2", "config", "sentence_classifier", yaml_file_name + ".yaml"
    )
    with open(fname, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Get parameters
    FLOW_ID = "sentence_classifier_flow"
    flow_config = config["flows"][FLOW_ID]
    params = flow_config["params"]

    # params for training data
    training_data = params["training_data_file_name"]

    # params for model
    split_random_seed = params["split_random_seed"]
    test_size = params["test_size"]
    bert_model_name = params["bert_model_name"]
    multi_process = params["multi_process"]

    max_depth = params["max_depth"]
    min_child_weight = params["min_child_weight"]
    gamma = params["gamma"]
    colsample_bytree = params["colsample_bytree"]
    subsample = params["subsample"]
    reg_alpha = params["reg_alpha"]
    max_iter = params["max_iter"]
    solver = params["solver"]
    penalty = params["penalty"]
    class_weight = params["class_weight"]
    C = params["C"]
    probability_threshold = params["probability_threshold"]

    # Output file name
    file_name = yaml_file_name.replace(".", "_")

    # Run flow
    training_data = load_training_data_from_s3(training_data)

    sent_class = SentenceClassifier(
        split_random_seed=split_random_seed,
        test_size=test_size,
        bert_model_name=bert_model_name,
        multi_process=multi_process,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        gamma=gamma,
        colsample_bytree=colsample_bytree,
        subsample=subsample,
        reg_alpha=reg_alpha,
        max_iter=max_iter,
        solver=solver,
        penalty=penalty,
        C=C,
        probability_threshold=probability_threshold,
    )

    X_train, X_test, y_train, y_test = sent_class.split_data(
        training_data, verbose=True
    )

    X_train_vec = sent_class.fit_transform(X_train)
    sent_class.fit(X_train_vec, y_train)

    # Training evaluation
    y_train_pred = sent_class.predict(X_train_vec)
    class_rep_train = sent_class.evaluate(y_train, y_train_pred, verbose=True)

    # Test evaluation
    y_test_pred = sent_class.predict_transform(X_test)
    class_rep_test = sent_class.evaluate(y_test, y_test_pred, verbose=True)

    results = {"Train": class_rep_train, "Test": class_rep_test}

    # Output
    sent_class.save_model(file_name + ".pkl")
    # # Loading a model
    # file_name = f'outputs/sentence_classifier/models/{job_id}.pkl'

    # sent_class = SentenceClassifier()
    # sent_class.load_model(file_name)

    # X_train, X_test, y_train, y_test = sent_class.split_data(training_data, verbose=True)

    # y_test_pred = sent_class.predict_transform(X_test)
    # class_rep_test = sent_class.evaluate(y_test, y_test_pred, verbose=True)
