## Sentence Classifier

After some experimentation in `analysis/sentence_classifier/notebooks/Sentence Classifier.ipynb` I refactored the code out of the notebook and into two scripts:
1. `create_training_data.py` - has tests in `tests/`
2. `sentence_classifier.py`


### 2021.07.06.yaml

`python skills_taxonomy_v2/pipeline/sentence_classifier/sentence_classifier.py --config_path 2021.07.06`

The test results for this config were:

```
              precision    recall  f1-score   support

           0       0.90      0.93      0.91       108
           1       0.90      0.86      0.88        81

    accuracy                           0.90       189
   macro avg       0.90      0.90      0.90       189
weighted avg       0.90      0.90      0.90       189
```
