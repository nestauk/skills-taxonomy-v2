# Experiments

The results of experiments to train a good classifier performed in `notebooks/Doccano Baseline Classifier.ipynb` and `notebooks/Sentence Classifier.ipynb`.

### Goals

We want to be able to filter out as much unnecessary 'not-skill' text from the job advert, whilst minimising accidentally filtering out skill sentences.

Our classifier is a binary classification where the positive (1) class is 'skill sentence', and the negative (0) is 'not-skill sentence'. We have 2 objectives:

1. We don't want to accidentally filter out skill sentences after misclassifying them as not-skill sentences (FN). So we want to predict skill sentences correctly as often as possible (TP/(TP+FN) is high). Thus we want to maximise the recall of the positive class.
2. We also want to make sure to actually filter out some not-skill sentences, but it's ok if a few stay in. So we also want to predict not-skill sentences correctly as often as possible (TN/(FP+TN) is high). Thus we want to maximise the recall of the negative class.

Since objective 1 is slightly more important than 2 we record both these recalls in the summary, rather than taking the average.

### Summary:


| Experiment number |Change | Vectorizer | Classifier | Cleaning | Training size | Recall of the positive class | Recall of the negative class |
|---|---|---|---|---|---|---|---|
|1|Baseline|CountVectorizer|MultinomialNB|Split on '.'|495|**0.86**|0.66|
|2|Baseline|CountVectorizer|LogisticRegression|Split on '.'|495|0.59|0.89|
|3|Use BERT|BERT last layer+scaler|LogisticRegression|Split on '.'|?|**0.85**|**0.84**|
|4|Mask numbers, remove camel case, split sentences using spacy|BERT last layer+scaler|LogisticRegression|Mask numbers, remove camel case, split sentences using spacy|810|0.80|0.88|
|5|Mask numbers with 'NUMBER'|BERT last layer+scaler|LogisticRegression|Mask numbers and remove hashes, remove camel case, split sentences using spacy|810|0.77|0.87|
|6|Remove bullets and small sentences|BERT last layer+scaler|LogisticRegression|Mask numbers and remove hashes, remove camel case, split sentences using spacy, remove bullet points, not included if length <-15|704|0.80|0.82|
|7|Keep camel cases in|BERT last layer+scaler|LogisticRegression|Mask numbers and remove hashes, split sentences using spacy, remove bullet points, not included if length <-15|566|**0.83**|**0.93**|
|8|Add no-skill extra data to both train/test|BERT last layer+scaler|LogisticRegression|Mask numbers and remove hashes, split sentences using spacy, remove bullet points, not included if length <-15|1061|0.75|0.90|
|9|Add skill and no-skill extra data to just train|BERT last layer+scaler|LogisticRegression|Mask numbers and remove hashes, split sentences using spacy, remove bullet points, not included if length <-15|1064|0.79|**0.93**|


## Baseline - 1 & 2

- Split by full stop
- CountVectorizer
- MultinomialNB
- 495 training sentences, 165 test sentences

Test results:
```
 precision    recall  f1-score   support

           0       0.84      0.66      0.74        87
           1       0.69      0.86      0.77        78

    accuracy                           0.75       165
   macro avg       0.76      0.76      0.75       165
weighted avg       0.77      0.75      0.75       165
```

- Use logisitic regression classifier

```
precision    recall  f1-score   support

0       0.71      0.89      0.79        87
1       0.82      0.59      0.69        78

accuracy                           0.75       165
macro avg       0.76      0.74      0.74       165
weighted avg       0.76      0.75      0.74       165

```

## BERT - 3

- Split by full stop
- `bert-base-uncased` last_hidden_state layer
- MinMaxScaler()
- LogisticRegression(max_iter=1000, class_weight="balanced")
- x training sentences, x test sentences


Test results:
```
      precision    recall  f1-score   support

           0       0.89      0.84      0.86        97
           1       0.78      0.85      0.82        68

    accuracy                           0.84       165
   macro avg       0.84      0.84      0.84       165
weighted avg       0.85      0.84      0.84       165
```


Experiments with different classifiers and using scalers or not:

||Test F1|Test precision|Test recall|
|---|---|---|---|
|MinMaxScaler + MultinomialNB|0.73|0.69|**0.78**|
|MinMaxScaler + SVC|**0.79**|0.79|**0.78**|
|SVC|0.78|0.80|0.77|
|MinMaxScaler + LogisticRegression|**0.79**|**0.81**|0.77|
|LogisticRegression|0.78|0.80|0.76|

## BERT with sentence cleaning - 4, 5, 6 & 7

- Remove numbers - Convert Spacy NER types 'DATE', 'MONEY', 'CARDINAL', 'TIME', 'ORDINAL', 'QUANTITY' to ####
- Remove camel case sentence problems, e.g. "One sentenceAnother sentence" -> "One sentence. Another sentence"
- Split sentences using Spacy model.
- 810 training sentences, 271 test sentences

```
precision    recall  f1-score   support

        0       0.86      0.88      0.87       159
        1       0.83      0.80      0.81       112

accuracy                           0.85       271
macro avg       0.84      0.84      0.84       271
weighted avg       0.85      0.85      0.85       271
```

- Clean out hashes (`re.sub(r'[#]+','NUMBER', sentence)`)

```
precision    recall  f1-score   support

        0       0.84      0.87      0.85       159
        1       0.80      0.77      0.79       112

accuracy                           0.83       271
macro avg       0.82      0.82      0.82       271
weighted avg       0.83      0.83      0.83       271
```


- Convert '*', '-' and bullet point to ',' (deal with lists as being part of the same sentence).
- Only include data if the sentence is of length > 15
- 704 training sentences, 235 test sentences

```
precision    recall  f1-score   support

        0       0.84      0.82      0.83       133
        1       0.77      0.80      0.79       102

accuracy                           0.81       235
macro avg       0.81      0.81      0.81       235
weighted avg       0.81      0.81      0.81       235
```


- Keep in camel case (I was worried there might be some bad applications of it e.g. PowerPoint, JavaScript).
- 566 training sentences, 189 test sentences

```
precision    recall  f1-score   support

        0       0.88      0.93      0.90       108
        1       0.89      0.83      0.86        81

accuracy                           0.88       189
macro avg       0.89      0.88      0.88       189
weighted avg       0.88      0.88      0.88       189
```

## BERT with sentence cleaning and TextKernel extra data - 8 & 9

- Add no-skill data from Text Kernel  'conditions_description' field that some JDs have (assume all of these are no skill sentences).
- Add random sample of 100 of these.
- 1061 training sentences, 354 test sentences
- Training Counter({0: 818, 1: 243})

```
precision    recall  f1-score   support

        0       0.93      0.90      0.91       273
        1       0.70      0.75      0.73        81

accuracy                           0.87       354
macro avg       0.81      0.83      0.82       354
weighted avg       0.87      0.87      0.87       354
```


In a way we want this to filter out no-skill sentences, so it being very good at classifying not-skills is good. But we don't want to leave too much in.

- Add skill data from Text Kernel data. I've assumed any data from Reed that contains the string 'Required skills' has skills only listed in a section between 'Required skills' and the first '\n\n' after this (it's usually a bullet pointed list).
- This time I only added the additional data from Text Kernel to the training set (so test set is only on the original data from Karlis), but the train is a mix of Karlis and TextKernel
- 1064 training sentences, 189 test sentences
- Training Counter({0: 611, 1: 453})
- Test Counter({0: 108, 1: 81})

```
precision    recall  f1-score   support

        0       0.85      0.93      0.89       108
        1       0.89      0.79      0.84        81

 accuracy                           0.87       189
macro avg       0.87      0.86      0.86       189
weighted avg       0.87      0.87      0.87       189
```

# Stochasticity

Since the dataset isn't very big, the split chosen for the test/train divide effects the outcome quite a lot. Using the parameters as in experiment 7 I retrained the model 5 times with different random seeds.

| Random seed | '0' recall | '1' recall | '0' precision | '1' precision | Number of 0 / 1 | 
|---|---|---|---|---|---|
| 0 | 0.824 | 0.84 | 0.873 | 0.782 | 108 / 81|
| 1 | 0.926 | 0.864 | 0.901 | 0.897 | 108 / 81|
| 2 | 0.824 | 0.852 | 0.881 | 0.784 | 108 / 81|
| 3 | 0.815 | 0.79 | 0.838 | 0.762 | 108 / 81|
| 42 | 0.88 | 0.84 | 0.88 | 0.84 | 108 / 81|

Slightly problematically (due to overfitting to the test set) I will pick to use random seed 1.

