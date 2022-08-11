"""
This pipeline creates and extracts skills from an input list of job adverts.

Prerequisites:
- You have access to our S3 bucket or have the skills classifier pkl file stored locally
- You have installed the environment requirements for this repo

Note:
- This script is not optimised to process 1000s of job adverts, so we recommend modifying this
code if you wanted to process many job adverts.
- It's advised to modify the data reduction and clustering parameters as these will be
very sensitive to the input data.
- The variable sentences_clustered contains the information needed for plotting sentences in 2D,
and finding which specific sentences were assigned to each skill

Running this script with the default job_advert_examples.txt file will print out:

The job advert:
This is a sentence about the company and the salary. We require applicants to have skills in Microsoft Excel.
Has skills:
['microsoft-excel-require']

The job advert:
We want Microsoft Excel skills for this role. Communication skills are also essential.
Has skills:
['microsoft-excel-require', 'communication-important-essential']

The job advert:
This role has a very competitive starting salary. Skills for good communication are very important.
Has skills:
['communication-important-essential']

"""

from skills_taxonomy_v2.pipeline.sentence_classifier.utils import split_sentence
from skills_taxonomy_v2.pipeline.sentence_classifier.sentence_classifier import (
	SentenceClassifier, BertVectorizer
)
from skills_taxonomy_v2.pipeline.skills_extraction.get_sentence_embeddings import get_embeddings
from skills_taxonomy_v2.pipeline.skills_extraction.cluster_embeddings import ClusterEmbeddings

import umap.umap_ as umap
import spacy
import pandas as pd

from tqdm import tqdm
from collections import Counter, defaultdict
import pickle
import json
import os
import logging
logger = logging.getLogger(__name__)

sent_classifier_model_dir = 'outputs/sentence_classifier/models/2021_08_16.pkl'
job_adverts_file = 'skills_taxonomy_v2/examples/job_advert_examples.txt'

# Parameters - these will need tweaking depending on your input data
reduction_n_neighbors = 6
reduction_min_dist = 0.0
clustering_eps = 1
clustering_min_samples = 1

def load_prerequisites(sent_classifier_model_dir):

	nlp = spacy.load("en_core_web_sm")
	bert_vectorizer = BertVectorizer(
			bert_model_name="sentence-transformers/all-MiniLM-L6-v2",
		)
	bert_vectorizer.fit()

	# Load the sentence classifier (either locally or on S3)

	sent_classifier = SentenceClassifier(bert_model_name= "bert-base-uncased")

	if os.path.exists(os.path.join(os.getcwd(), sent_classifier_model_dir)):
		sent_classifier.classifier = pickle.load(sent_classifier_model_dir)
		sent_classifier.load_bert()
	else:
		logger.info("Sentence Classifier not found locally, so trying to find on S3")
		sent_classifier.load_model(sent_classifier_model_dir.split('/')[-1])

	return nlp, bert_vectorizer, sent_classifier

def split_skills(job_adverts):

	all_job_ids = []
	all_sentences = []
	for job_advert in job_adverts:
		job_id, sentences = split_sentence(job_advert, min_length=30)
		all_job_ids += [job_id]*len(sentences)
		all_sentences += sentences

	return all_job_ids, all_sentences

def predict_skill_sents(sent_classifier, all_job_ids, all_sentences):

	sentences_vec = sent_classifier.transform(all_sentences)
	sentences_pred = sent_classifier.predict(sentences_vec)

	skill_sentences_dict = defaultdict(list)
	for job_id, sent, pred in zip(all_job_ids, all_sentences, sentences_pred):
		if pred == 1:
			skill_sentences_dict[job_id].append(sent)

	return skill_sentences_dict

def reduce_embeddings(sentence_embeddings, original_sentences, reduction_n_neighbors, reduction_min_dist):

	reducer_class = umap.UMAP(
		n_neighbors=reduction_n_neighbors,
		min_dist=reduction_min_dist,
		random_state=42,
		n_components=2,
	)
	reducer_class.fit([e for _,_,_,e in sentence_embeddings])

	sent_thresh = 250
	mask_seq = "[MASK]"
	embedding_list = []
	words_list = []
	sentence_list = []
	jobid_list = []
	sentid_list = []
	for job_id, sent_id, words, embedding in sentence_embeddings:
		original_sentence = original_sentences[sent_id]
		if len(original_sentence) < sent_thresh:
			embedding_list.append(embedding)
			words_list.append(words.replace(mask_seq, "").split())
			sentence_list.append(original_sentence)
			jobid_list.append(job_id)
			sentid_list.append(sent_id)

	sentence_embeddings_red = reducer_class.transform(embedding_list).tolist()
	sentences_data =  {
		"description": words_list,
		"original sentence": sentence_list,
		"job id": jobid_list,
		"sentence id": sentid_list,
		"embedding": sentence_embeddings_red,
	}
	
	sentences_data_df = pd.DataFrame(sentences_data)

	sentences_data_df["reduced_points x"] = sentences_data_df["embedding"].apply(lambda x: x[0])
	sentences_data_df["reduced_points y"] = sentences_data_df["embedding"].apply(lambda x: x[1])
	sentences_data_df["original sentence length"] = sentences_data_df["original sentence"].apply(lambda x:len(x))

	return sentences_data_df

def get_skill_name(skill_data):
	"""
	Find the most common words in the skill description words as a way to name skills
	"""
	common_description_words = Counter([v for d in skill_data['description'] for v in d]).most_common(3)
	return '-'.join([c[0] for c in common_description_words])

def cluster_embeddings(sentences_data_df, clustering_eps, clustering_min_samples):

	cluster_embeddings = ClusterEmbeddings(
		dbscan_eps=clustering_eps,
		dbscan_min_samples=clustering_min_samples,
		train_cluster_n=len(sentences_data_df),
		)
	_ = cluster_embeddings.get_clusters(sentences_data_df)
	sentences_clustered = cluster_embeddings.sentences_data_short_sample

	return sentences_clustered

if __name__ == '__main__':

	logger.info("Loading pre-trained models and data ...")

	# Load pre-trained models needed for this pipeline
	
	nlp, bert_vectorizer, sent_classifier = load_prerequisites(sent_classifier_model_dir)
	
	# Load your job advert texts; a list of dicts with the keys "full_text" and "job_id"

	with open(job_adverts_file) as f:
		job_adverts = json.load(f)

	# Run the pipeline to extract skills

	logger.info("Split the sentences ...")
	all_job_ids, all_sentences = split_skills(job_adverts)

	logger.info("Predict skill sentences ...")
	skill_sentences_dict = predict_skill_sents(sent_classifier, all_job_ids, all_sentences)

	logger.info("Embed skill sentences ...")
	sentence_embeddings, original_sentences = get_embeddings(skill_sentences_dict, nlp, bert_vectorizer)

	logger.info("Reduce embeddings ...")
	sentences_data_df = reduce_embeddings(sentence_embeddings, original_sentences, reduction_n_neighbors, reduction_min_dist)

	logger.info("Cluster the reduced embeddings ...")
	sentences_clustered = cluster_embeddings(sentences_data_df, clustering_eps, clustering_min_samples)
	
	logger.info("Summarise skill information ...")

	skill_name_dict = sentences_clustered.groupby('cluster_number').apply(lambda x: get_skill_name(x)).to_dict()
	job_skills_dict = sentences_clustered.groupby('job id')['cluster_number'].unique().to_dict()

	for job_advert in job_adverts:
		job_advert["Skills"] = [skill_name_dict[skill_num] for skill_num in job_skills_dict[job_advert['job_id']]]

	print(f"There are {len(skill_name_dict)} skills extracted using this data")
	for i in range(3):
		print(f'The job advert: \n{job_adverts[i]["full_text"]} \nHas skills: \n{job_adverts[i]["Skills"]}\n')

