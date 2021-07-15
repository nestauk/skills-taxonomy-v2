"""
Input sentences output word embeddings for each of them
"""

import pickle

from transformers import BertTokenizer, BertModel

from skills_taxonomy_v2.pipeline.skills_extraction.cleaning_sentences import (
    deduplicate_sentences,
    sentences2cleantokens,
    build_ngrams,
    get_common_tuples,
    remove_common_tuples
)
import spacy
import torch
import numpy
from thinc.api import set_gpu_allocator, require_gpu
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

if __name__ == '__main__':

	# Use the GPU, with memory allocations directed via PyTorch.
	# This prevents out-of-memory errors that would otherwise occur from competing
	# memory pools.
	set_gpu_allocator("pytorch")
	require_gpu(0)
	
	config_name = '2021.07.09.small'

	with open(
	    f"outputs/sentence_classifier/data/skill_sentences/{config_name}_jobsnew1_predictions.pkl", 
	    "rb") as file:
	    sentences_pred = pickle.load(file)
	    
	with open(
	    f"outputs/sentence_classifier/data/skill_sentences/{config_name}_jobsnew1_sentences.pkl",
	    "rb") as file:
	    sentences = pickle.load(file)

	with open(
	    f"outputs/sentence_classifier/data/skill_sentences/{config_name}_jobsnew1_job_ids.pkl",
	    "rb") as file:
	    job_ids = pickle.load(file)

	token_len_threshold = 20 # To adjust. I had a look and > this number seems to all be not words (urls etc)
	lemma_n = 2
	top_n_common_remove = 20

	# Filter out the non-skill sentences
	sentences = [sentences[i] for i,p in enumerate(sentences_pred.astype(bool)) if p==1]

	# Deduplicate sentences
	sentences = deduplicate_sentences(sentences)

	# Get embeddings for each token in the sentence
	nlp = spacy.load("en_core_web_trf")

	for doc in nlp.pipe(["some text", "some other text"]):
    	tokvecs = doc._.trf_data.tensors[-1]

	# lemma_sentences = []
	# clean_sentences = []
	# tokvecs_sentences = []
	# for sentence in sentences:
	#     sentence = separate_camel_case(sentence)
	#     doc = nlp(sentence)
	#     tokvecs = doc._.trf_data.tensors[-1]
	#     tokens = doc._.trf_word_pieces_  # String values of the wordpieces
	#     # Only append tokens/embeddings for words which are ok
	#     # Don't include very long words
	#     # or proper nouns/numbers
	#     # or words with numbers in (these are always garbage)
	#     # You generally take out a lot of the urls by having a token_len_threshold but not always
	#     lemma_sentence = []
	#     clean_sentence = []
	#     tokvecs_sentence = []
	#     for token in doc:
	#         # Don't include very long words
	#         # or proper nouns/numbers
	#         # or words with numbers in (these are always garbage)
	#         # You generally take out a lot of the urls by having a token_len_threshold but not always
	#         if (
	#             ("www" not in token.text)
	#             and (len(token) < token_len_threshold)
	#             and (token.pos_ not in ["PROPN", "NUM", "SPACE"])
	#             and (not re.search("\d", token.text))
	#             and (not token.text in stopwords.words())
	#         ):
	#             lemma_sentence.append(token.lemma_.lower())
	#             clean_sentence.append(token.text.lower())
	#     if lemma_sentence:
	#         lemma_sentences.append(lemma_sentence)
	#     if clean_sentence:
	#         clean_sentences.append(clean_sentence)


