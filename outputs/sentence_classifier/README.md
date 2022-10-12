# Trained Skill Sentence Classifer

This directory contains a zipped `.pkl` file of the `2021.08.16.yaml` trained skills sentence classifier.

To learn more about the model, its features, experiments and qualitative observations, please see the [skills_taxonomy_v2/pipeline/sentence_classifier/](https://github.com/nestauk/skills-taxonomy-v2/tree/dev/skills_taxonomy_v2/pipeline/sentence_classifier) directory.

### Sentence classifier - 2021.08.16.yaml

python skills_taxonomy_v2/pipeline/sentence_classifier/sentence_classifier.py --yaml_file_name 2021.08.16

The test results for this config were:

Train: Took 314.79356503486633 seconds

              precision    recall  f1-score   support

           0       0.98      1.00      0.99      3931
           1       1.00      0.95      0.98      1951

    accuracy                           0.98      5882

macro avg 0.99 0.98 0.98 5882
weighted avg 0.98 0.98 0.98 5882

Test: Took 132.80465602874756 seconds

              precision    recall  f1-score   support

           0       0.90      0.97      0.93       694
           1       0.93      0.78      0.85       344

    accuracy                           0.90      1038

macro avg 0.90 0.90 0.90 1038
weighted avg 0.90 0.90 0.90 1038
