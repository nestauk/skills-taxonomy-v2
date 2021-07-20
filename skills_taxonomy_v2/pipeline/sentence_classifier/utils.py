from skills_taxonomy_v2.pipeline.sentence_classifier.create_training_data import mask_text, text_cleaning

def split_sentence(data, nlp, min_length=15, max_length=100):
	"""
	Split one sentence. Only include in output if sentence length
	is in a range.
	Output is two lists, a list of each sentence and a list of the job_ids they are from.
	This has to be in utils.py and not predict_sentence_class.py so it can be used
	with multiprocessing (see https://stackoverflow.com/questions/41385708/multiprocessing-example-giving-attributeerror/42383397)
	"""
	text = data.get('full_text')
	# Occassionally this may be filled in as None
	if text:
		sentences = []
		job_id = data.get('job_id')
		text = mask_text(nlp, text)
		# Split up sentences
		doc = nlp(text)
		for sent in doc.sents:
			sentence = text_cleaning(sent.text) 
			if len(sentence) in range(min_length, max_length):
				sentences.append(sentence)
		return job_id, sentences
	else:
		return None, None