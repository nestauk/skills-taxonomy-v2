# Sentence Classifier

There are three main scripts for the scaled up Sentence Classifier:

1. `sentence_classifier.py` - module for the BertVectorizer class the updated SentenceClassifier class
2. `predict_sentence_class.py` - make predictions using the sentence classifier on an input file of data
3. `utils.py` - preprocessing steps for sentences, splitting sentences, verb features

As the training data labels aren't perfect, I would take the classification report with a slight pinch of salt. Qualitatively, after looking at sentences that the model labelled 0 but was manually labelled 1, it appears that the most recent classifier pipeline does a worse job of picking up sentences that are labelled 'skill' sentences that appear to contain a 'characteristic' as opposed to a skill i.e.:

- 'we are looking for an ethusiastic, talented individual to join the team'
- 'if you can offer us some flexibility to help us cover peak times such as back to school all the better'
- 'there are many reasons why people become care assistants but the main motivation is the chance to make a difference'
- 'number about the role are you a paediatric healthcare assistant looking to join an exceptional team of people'

where (presumably) someone labelled this as a skill sentence for 'enthusiastic', 'fexibility', '(wanting to)make a difference' etc.

### Sentence classifier - `2021.08.16.yaml`

`python skills_taxonomy_v2/pipeline/sentence_classifier/sentence_classifier.py --yaml_file_name 2021.08.16`

The test results for this config were:

Train:
Took 314.79356503486633 seconds

```
              precision    recall  f1-score   support

           0       0.98      1.00      0.99      3931
           1       1.00      0.95      0.98      1951

    accuracy                           0.98      5882
   macro avg       0.99      0.98      0.98      5882
weighted avg       0.98      0.98      0.98      5882

```

Test:
Took 132.80465602874756 seconds

```
              precision    recall  f1-score   support

           0       0.90      0.97      0.93       694
           1       0.93      0.78      0.85       344

    accuracy                           0.90      1038
   macro avg       0.90      0.90      0.90      1038
weighted avg       0.90      0.90      0.90      1038
```

### Sentence classifier - `2021.07.06.yaml`

`python skills_taxonomy_v2/pipeline/sentence_classifier/sentence_classifier.py --yaml_file_name 2021.07.06`

- 10 job adverts = 38 seconds

The test results for this config were:

```
              precision    recall  f1-score   support

           0       0.90      0.93      0.91       108
           1       0.90      0.86      0.88        81

    accuracy                           0.90       189
   macro avg       0.90      0.90      0.90       189
weighted avg       0.90      0.90      0.90       189
```

### Sentence classifier - `2021.08.06.yaml`

This config includes training data where additional cleaning was done. This allowed better splitting of sentences and so there is more of it. This training data file is `outputs/sentence_classifier/data/training_data_April2021_08_06_21.json`. Uses my own method for getting the last BERT embedding layer from 'bert-base-uncased'. This uses seed 4.

```
'Seed': 0, '0 test Recall': 0.82, '1 test Recall': 0.85
'Seed': 1, '0 test Recall': 0.89, '1 test Recall': 0.79
'Seed': 2, '0 test Recall': 0.86, '1 test Recall': 0.78
'Seed': 3, '0 test Recall': 0.79, '1 test Recall': 0.86
'Seed': 4, '0 test Recall': 0.85, '1 test Recall': 0.84
'Seed': 5, '0 test Recall': 0.82, '1 test Recall': 0.79
```

## Experiments with `sentence_embeddings` library

### paraphrase-MiniLM-L6-v2, multi_process: False

Size of training data: 798
Size of test data: 267
Train time about 13 seconds
Test time about 5 seconds

```
'Seed': 0, '0 test Recall': 0.76, '1 test Recall': 0.76
'Seed': 1, '0 test Recall': 0.83, '1 test Recall': 0.77 **
'Seed': 2, '0 test Recall': 0.72, '1 test Recall': 0.76
'Seed': 3, '0 test Recall': 0.77, '1 test Recall': 0.77
'Seed': 4, '0 test Recall': 0.76, '1 test Recall': 0.79
'Seed': 5, '0 test Recall': 0.81, '1 test Recall': 0.77
```

### paraphrase-mpnet-base-v2, multi_process: False

Size of training data: 798
Size of test data: 267
Train time about 98 seconds
Test time about 32 seconds

```
'Seed': 0, '0 test Recall': 0.83, '1 test Recall': 0.85
'Seed': 1, '0 test Recall': 0.88, '1 test Recall': 0.82
'Seed': 2, '0 test Recall': 0.83, '1 test Recall': 0.78
'Seed': 3, '0 test Recall': 0.80, '1 test Recall': 0.87
'Seed': 4, '0 test Recall': 0.85, '1 test Recall': 0.80
'Seed': 5, '0 test Recall': 0.85, '1 test Recall': 0.83
```

### paraphrase-mpnet-base-v2, multi_process: True

Size of training data: 798
Size of test data: 267
Train time about 56 seconds
Test time about 20 seconds

```
'Seed': 0, '0 test Recall': 0.83, '1 test Recall': 0.85
'Seed': 1, '0 test Recall': 0.88, '1 test Recall': 0.82**
'Seed': 2, '0 test Recall': 0.83, '1 test Recall': 0.78
'Seed': 3, '0 test Recall': 0.80, '1 test Recall': 0.87
'Seed': 4, '0 test Recall': 0.85, '1 test Recall': 0.80
'Seed': 5, '0 test Recall': 0.85, '1 test Recall': 0.83
```

## Sentence classifier - `2021.07.09.small.yaml`

```
python skills_taxonomy_v2/pipeline/sentence_classifier/sentence_classifier.py --yaml_file_name 2021.07.09.small
```

- This is the smaller trained model. It isn't as good, but takes less time.
- It uses `paraphrase-MiniLM-L6-v2` with multiprocessing, and seed 1.
- 1000 texts takes about 17 seconds.

Train:

```
Took 13.46814489364624 seconds
              precision    recall  f1-score   support

           0       0.91      0.88      0.89       429
           1       0.87      0.90      0.88       369

    accuracy                           0.89       798
   macro avg       0.89      0.89      0.89       798
weighted avg       0.89      0.89      0.89       798

```

Test:

```
Took 7.951373100280762 seconds
              precision    recall  f1-score   support

           0       0.81      0.83      0.82       144
           1       0.80      0.77      0.79       123

    accuracy                           0.81       267
   macro avg       0.80      0.80      0.80       267
weighted avg       0.81      0.81      0.80       267

```

## Sentence classifier - `2021.07.09.large.yaml`

```
python skills_taxonomy_v2/pipeline/sentence_classifier/sentence_classifier.py --yaml_file_name 2021.07.09.large
```

- It uses `paraphrase-mpnet-base-v2` with multiprocessing, and seed 1.
- 1000 texts takes about 72 seconds.

Train:

```
Took 57.075644731521606 seconds
              precision    recall  f1-score   support

           0       0.98      0.95      0.97       429
           1       0.95      0.98      0.96       369

    accuracy                           0.96       798
   macro avg       0.96      0.96      0.96       798
weighted avg       0.96      0.96      0.96       798

```

Test:

```
Took 18.327223777770996 seconds
              precision    recall  f1-score   support

           0       0.85      0.88      0.86       144
           1       0.85      0.82      0.83       123

    accuracy                           0.85       267
   macro avg       0.85      0.85      0.85       267
weighted avg       0.85      0.85      0.85       267

```

## Predicting

Running

```
python skills_taxonomy_v2/pipeline/sentence_classifier/predict_sentence_class.py --config_path 'skills_taxonomy_v2/config/predict_skill_sentences/2021.08.16.local.sample.yaml'
```

with the most recent config file will take in job adverts, split them into sentences, and make predictions using a trained skill sentence classifier model (`model_config_name`). The output will be filtered to only include skill sentences. The config file can specify that the data to be predicted on comes from S3 or locally (`data_local`), and it can be a directory of files to predict on or just one file (`input_dir+data_dir`). The outputted file(s) will be in the same folder structure as the inputs and contain one json in the form of:

```
{'job_id_1': [('sentence1'), ('sentence2')], 'job_id_2': [('sentence1'), ('sentence2'}
```

### `2021.10.27.yaml` config file:

When using the `2021.10.27.yaml` config file skill sentences are predicted on a pre-determined sample of 5 million job adverts (found via running `get_tk_sample.py`). This was run using:

```
python skills_taxonomy_v2/pipeline/sentence_classifier/predict_sentence_class.py --config_path 'skills_taxonomy_v2/config/predict_skill_sentences/2021.10.27.yaml'
```

Skill sentences for each job advert and file are stored in `outputs/sentence_classifier/data/skill_sentences/2021.10.27/`.

Out of the 647 files of job adverts, 516 had skill sentences in. This is because the 'jobs_expired' files were included in the sample and these don't contain the job advert text. This leaves us with a sample of 4,312,285 job adverts.

Skill sentences were identified in 3,572,140 unique job adverts (i.e. some job adverts had no skill sentences at all/ some were under 30 characters).

### From `2021.07.09.yaml`:

This will run predictions on a random sample of 10 of the 686 data files. The outputs of this yielded 5,823,903 skill sentences from the 1,000,000 job adverts.

### From `2021.08.16.yaml`:

This will run predictions on a random sample of 100 of the 686 data files. Only the first 10,000 job adverts from each file were processed due to memory issues.

The skill sentences predicted are stored in "outputs/sentence_classifier/data/skill_sentences/2021.08.16/textkernel-files/", only 87 files were found, suggesting that 13 from the sample of 100 didn't have any skill sentences in for one reason or another.

In each of the 87 files around 45,000 skill sentences were identified, thus we expect to have processed around 4,000,000 skill sentences. 

### From `2021.08.16.yaml` with fixes to the sample taken:

We noticed an issue where parts of the sample of job adverts came from expired files. These files had no full text field. Hence, we replaced the job adverts samples from expired files with job adverts sample from files from the same folder location (which is linked to date).

After appending the skill sentences predictions using this new sample of 5 million job adverts (in `predict_sentence_class_inc_replacements.py`) we found there are 4,118,467 job ids (all unique) in our sample with skill sentences. And in total there are 19,755,486 skill sentences found.




