# Sentence Classifier

After some experimentation in `analysis/sentence_classifier/notebooks/Sentence Classifier.ipynb` I refactored the code out of the notebook and into three scripts:
1. `create_training_data.py` - has tests in `tests/`
2. `sentence_classifier.py` - train a sentence classifier using a config yaml
3. `predict_sentence_class.py` - make predictions using the sentence classifier on an input file of data

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
python skills_taxonomy_v2/pipeline/sentence_classifier/predict_sentence_class.py
```
will take in job adverts, split them into sentences, and make predictions using the `2021.07.09.small` trained model. This outputs 4 files:
1. `outputs/sentence_classifier/data/skill_sentences/2021.07.09.small_jobsnew1_embeddings.pkl` - the BERT embeddings for each sentence.
2. `outputs/sentence_classifier/data/skill_sentences/2021.07.09.small_jobsnew1_predictions.pkl` - the model predictions for each sentence.
3. `outputs/sentence_classifier/data/skill_sentences/2021.07.09.small_jobsnew1_sentences.pkl` - each sentence text.
4. `outputs/sentence_classifier/data/skill_sentences/2021.07.09.small_jobsnew1_job_ids.pkl` - the job advert ID for each sentence.



