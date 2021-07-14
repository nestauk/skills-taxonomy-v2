import json
from itertools import chain, combinations
import re
import pickle

from tqdm import tqdm
import spacy
import pandas as pd
from gensim import models
from gensim.corpora import Dictionary
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def separate_camel_case(sentence):
	"""
	Convert instances of 'a wordThen another' to 'a word Then another'
	"""
	for i in reversed(list(re.finditer(r'[a-z][A-Z][a-z]', sentence))):
	   sentence = sentence[:(i.start()+1)] + ' ' + sentence[(i.end()-2):]
	return sentence

def build_ngrams(documents, n=2):
	"""Create ngrams using Gensim's phrases.

	Args:
		documents: List of tokenised documents.
		n: The `n` in n-gram.

	Returns:
		List of n-grammed documents.
	"""
	if n < 2:
		return documents

	phrase_kws = {
		"scoring": "npmi",
		"threshold": 0.25,
		"min_count": 2,
		"delimiter": "_",
	}

	step = 1
	while step < n:
		phrases = models.Phrases(documents, **phrase_kws)
		bigram = models.phrases.Phraser(phrases)
		del phrases
		tokenised = bigram[documents]
		step += 1

	return list(tokenised)

def deduplicate_sentences(sentences):
	return list(dict.fromkeys(sentences))

def sentences2cleantokens(sentences, token_len_threshold=20):
	"""
	Convert list of sentences to list of list of clean words for each sentence.
	"This is a sentence about Apple" -> ["this", "is", "a", "sentence", "about"]

	Outputs a non-lemma and a lemma version.
	"""

	nlp = spacy.load("en_core_web_sm")

	lemma_sentences = []
	clean_sentences = []
	for sentence in sentences:
		sentence = separate_camel_case(sentence)
		lemma_sentence = []
		clean_sentence = []
		doc = nlp(sentence)
		for token in doc:
			# Don't include very long words 
			# or proper nouns/numbers 
			# or words with numbers in (these are always garbage)
			# You generally take out a lot of the urls by having a token_len_threshold but not always
			if (
				('www' not in token.text) and \
				(len(token) < token_len_threshold) and \
				(token.pos_ not in ['PROPN', 'NUM', 'SPACE']) and \
				(not re.search("\d", token.text)) and \
				(not token.text in stopwords.words())
				):
				lemma_sentence.append(token.lemma_.lower())
				clean_sentence.append(token.text.lower())
		if lemma_sentence:
			lemma_sentences.append(lemma_sentence)
		if clean_sentence:
			clean_sentences.append(clean_sentence)

	return clean_sentences, lemma_sentences

def get_common_tuples(sentence_words, top_n=20):
	"""
	Words like 'apply' and 'now' co-occur very commonly and are often not to do with skills.
	Get a list of them.
	"""

	# Sort list of words in alphabetical order
	# so you get tuples in the same order e.g. ('A_word', 'B_word') not ('B_word', 'A_word')
	pairs = list(chain(*[list(combinations(sorted(x), 2)) for x in sentence_words]))
	pairs = [x for x in pairs if len(x) > 0]

	edge_list = pd.DataFrame(pairs, columns=["word 1", "word 2"])
	edge_list["weight"] = 1
	edge_list_weighted = (
			edge_list.groupby(["word 1", "word 2"])["weight"].sum().reset_index(drop=False)
		)

	# Automating this for now, but you may wish to manually curate and save elsewhere
	common_word_tuples = edge_list_weighted.sort_values(
		'weight', ascending=False
		)[0:top_n][['word 1', 'word 2']].values.tolist()
	common_word_tuples_set = [set(common_word_tuple) for common_word_tuple in common_word_tuples]
	return common_word_tuples_set

def remove_common_tuples(sentence_words, common_word_tuples_set):
	"""
	Remove the common tuples for sentences.
	"""
	sentence_words_clean = []
	for sentence_words in sentence_words:
		sentence_words_set = set(sentence_words)
		pop_words = set()
		for common_word_tuple in common_word_tuples_set:
			if common_word_tuple.issubset(sentence_words_set):
				pop_words.update(common_word_tuple)
		filtered_list = list(sentence_words_set.difference(pop_words))
		if filtered_list:
			sentence_words_clean.append(filtered_list)

	return sentence_words_clean

if __name__ == '__main__':

	config_name = '2021.07.09.small'
		
	with open(
		f"outputs/sentence_classifier/data/skill_sentences/{config_name}_jobsnew1_predictions.pkl", 
		"rb") as file:
		sentences_pred = pickle.load(file)
		
	with open(
		f"outputs/sentence_classifier/data/skill_sentences/{config_name}_jobsnew1_sentences.pkl",
		"rb") as file:
		sentences = pickle.load(file)

	token_len_threshold = 20 # To adjust. I had a look and > this number seems to all be not words (urls etc)
	lemma_n = 2
	top_n_common_remove = 20

	# Filter out the non-skill sentences
	sentences = [sentences[i] for i,p in enumerate(sentences_pred.astype(bool)) if p==1]

	# Deduplicate sentences
	sentences = deduplicate_sentences(sentences)

	# Apply cleaning to the sentences
	sentence_words, lemma_sentence_words = sentences2cleantokens(sentences, token_len_threshold=token_len_threshold)

	# Get n-grams
	sentence_words = build_ngrams(sentence_words, n=lemma_n)
	lemma_sentence_words = build_ngrams(lemma_sentence_words, n=lemma_n)

	# Remove common co-occurring words
	common_word_tuples_set = get_common_tuples(sentence_words, top_n=top_n_common_remove)
	sentence_words_clean = remove_common_tuples(sentence_words, common_word_tuples_set)

	lemma_common_word_tuples_set = get_common_tuples(lemma_sentence_words, top_n=top_n_common_remove)
	lemma_sentence_words_clean = remove_common_tuples(lemma_sentence_words, lemma_common_word_tuples_set)

