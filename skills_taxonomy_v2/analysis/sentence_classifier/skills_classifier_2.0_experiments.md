# Experiments

The results of experiments to train a classifier performed in `improve_classifier.ipynb`. 

### Goals

We want to improve the initial sentence classifier that predicts whether a sentence is a skill or not-skill sentence. The **best current baseline has a recall score of 0.83 for the positive class (skill sentence) and 0.93 for the negative class (non-skill sentence)**. 

The first step of improving the classifier was to collect more training data ([see issue #24](https://github.com/nestauk/skills-taxonomy-v2/issues/24)). By hosting a sample of the job ad data on AWS and crowdsourcing labels, we were able to generate an additional *8212* labelled sentences.

We are now trying to increase performance of the initial classifier with the additional training data to **achieve a recall score of 0.90 for the positive class (skill sentence)**.  

### Experiments:

To see how the previous iteration of experiments with a smaller set of training data performed, please see [this markdown file](https://github.com/nestauk/skills-taxonomy-v2/blob/dev/skills_taxonomy_v2/analysis/sentence_classifier/Sentence%20Splitter%20Experiments.md). Liz's best baseline is run on the new training data as the new baseline.  

| Experiment number |Change | Vectorizer | Classifier | Cleaning | Training source | Training size | Recall of the positive class | Precision of the positive class | Recall of the negative class | Precision of the positive class 
|---|---|---|---|---|---|---|---|---|---|---|
|1|Baseline|BERT last layer+scaler|LogisticRegression|Mask numbers and remove hashes, split sentences using spacy, remove bullet points|Karlis + Label Studio|9237|**0.82**|0.54|**0.82**|0.94|
|2|just label studio training data|BERT last layer|LogisticRegression|Mask numbers and remove hashes, split sentences using spacy, remove bullet points|Label Studio|8212|**0.86**|0.52|**0.83**|0.97| 
|3|Karlis + label studio, sentence preprocessing|BERT last layer|LogisticRegression|Mask + remove numbers and remove hashes, split sentences using spacy, remove bullet points, lowercase|Karlis + Label Studio|9237|**0.82**|0.55|**0.82**|0.95|
|4|use XGboost|BERT last layer|XGboost|Mask + remove numbers and remove hashes, split sentences using spacy, remove bullet points, lowercase|Karlis + Label Studio|9237|**0.52**|0.75|**0.95**|0.87|
|5|Balance training data - undersample 0 class|BERT last layer|XGboost|Mask + remove numbers and remove hashes, split sentences using spacy, remove bullet points, lowercase|Karlis + Label Studio|3677|**0.90**|0.48|**0.76**|0.97| 
|6|Balance training data - use nlpaug word synonyms to oversample 1 class|BERT last layer|XGboost|Mask + remove numbers and remove hashes, split sentences using spacy, remove bullet points, lowercase|Karlis + Label Studio|10053|**0.82**|0.85|**0.93**|**0.91**
|7|Balance training data - use contextual word embeddings to oversample 1 class|BERT last layer|XGboost|Mask + remove numbers and remove hashes, split sentences using spacy, remove bullet points, lowercase|Karlis + Label Studio|10053|**0.80**|0.89|**0.80**|0.89|
|8|Balance training data - oversample 1 class (word synonyms) + under sample 0 class|BERT last layer|XGboost|Mask + remove numbers and remove hashes, split sentences using spacy, remove bullet points, lowercase|Karlis + Label Studio|7538|**0.87**|0.73|**0.91**|0.96|
|9|Balance training data - oversample 1 class (word embeds) + under sample 0 class|BERT last layer|XGboost|Mask + remove numbers and remove hashes, split sentences using spacy, remove bullet points, lowercase|Karlis + Label Studio|7666|**0.84**|0.79|**0.94**|0.96|
|**10**|**use one hot encoding of verb positionality**|**BERT last layer+verb one hot encoding**|**XGboost**|**Mask + remove numbers and remove hashes, split sentences using spacy, remove bullet points, lowercase**|**Karlis + Label Studio**|**7732**|**0.94**|**0.73**|**0.91**|**0.98**|
|11|adjust probability threshold to 0.4|BERT last layer+verb one hot encoding|XGboost|Mask + remove numbers and remove hashes, split sentences using spacy, remove bullet points, lowercase|Karlis + Label Studio|7732|**0.94**|0.66|**0.87**|0.98|
|11|adjust probability threshold to 0.6|BERT last layer+verb one hot encoding|XGboost|Mask + remove numbers and remove hashes, split sentences using spacy, remove bullet points, lowercase|Karlis + Label Studio|7732|**0.89**|0.74|**0.92**|0.97|

### Skills Classifier 2.0 Summary