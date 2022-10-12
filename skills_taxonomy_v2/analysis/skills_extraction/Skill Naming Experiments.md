<!-- #region -->

# Naming skills 2.0

**As of 11-09-2021:**

Functions to generate skill cluster names can be found in `skills_taxonomy_v2/pipeline/skills_extraction/skills_naming_utils.py.` We are now no longer merging similar skill clusters - instead, we are generating skill names with counters (i.e. project management 1, project management 2) for duplicate skill names, with the assumption that identical skill names should exist in different parts of the hierarchy. These names will then be manually renamed to capture the local context of the skill cluster name.

Experiments with better naming can be found in `skills_taxonomy_v2/analysis/skills_extraction/notebooks/better_skills_naming.py`.

In an effort to improve skill cluster naming and to lower the probability of duplicate skill cluster label names, the initial approach was improved by:

1. More text cleaning - incl. getting rid of job specific language, singularising terms, getting rid of duplicate candidate ngram phrases i.e. 'day day'
2. Generating candidate ngrams _per_ cluster - candidate ngrams were generated using each cluster's available vocabulary, hopefully creating more 'local' skill labels. Minimum descriptions were also used as cluster labels in the event that candidate ngrams were not generated.
3. Merging skill clusters - if both the skill cluster name AND the cluster centroid were very close in semantic space, skills were merged together.

Experimentation with key phrase generation also took place.

## Key Phrase generation experiments

The 3 'key phrase' approaches are as follows:

1. Using Spacy's Phrases algorithm
2. Using TextRank network approach
3. Using Spacy's VERB chunking

Candidate ngrams were generated using the above methods per skill cluster. The cosine similarity of the generated candidate ngrams embeddings and the cluster centroid were then calcualed. The candidate ngram closest to the skill cluster centroid was then chosen as the skill cluster label.

See below the generated ngrams per 'key phrase approach' for a sample of k = 28 sentence clusters:

1. Phrases + embeddings

<img src="figures/naming_experiments/phrases_embeddings.png" alt="phrases" width="600"/>

2. TextRank + embeddings

<img src="figures/naming_experiments/pyrank_embeddings.png" alt="textrank" width="600"/>

3. chunks + embeddings

<img src="figures/naming_experiments/chunk_embeddings.png" alt="chunks" width="600"/>

All three approaches took a relatively similar amount of time to run on a data sample with k randomly sampled clusters, with the TextRank approach taking slightly longer. Qualitatively, it appears that the Phrases approach is the most semantically logical. The labels are also semantically similar to the TextRank approach.

## Merging skills

Skill clusters were merged based on both a) the proximity of skill cluster labels in semantic space AND 2) the proximity of the skill cluster centroids in semantic space. If the skill cluster label embeddings AND ALL skill cluster centroids were close, the skill clusters were subsequently merged.

Merged skill examples include (changes subject to thresholds):

Merged clusters 13714, 18334 and 17999:

```
{'Skills name': 'managing projects',
 'Texts': ['infrastructure engineerdevops engineer play major part clients journey aws',
  'project management methodologies prince pmi',
  'main include writing developing simulation models',
  'extensive large construction projects communication',
  'minutes arranging site visits inductions',
  'managing projects academic higher education',
  'utilise briefs planograms tasks carried plan',
  'project management team',
  'complete make applications funding behalf organisation whole',
  'responsible managing projects emergency faults',
  'organise preparation prior service',
  'project managing opening closing projects',
  'example involved developing individual plans supporting new colleagues',
  'provide initial point users computerrelated problems']}
```

Merged clusters 962 and 17669:

```
{'Skills name': 'punctual reliable',
 'Texts': ['punctuality key always focusing standards',
  'production cando attitude reliability punctuality',
  'right person hard reliable punctual',
  'calm methodical manner patience logic perspective even busy',
  'individuals patient punctual reliable flexibly',
  'motivated punctual professional times',
  'flexible working reliable hardworking punctual']}
```

There needs to be some additional thinking on improving handling more than two skill cluster names that are close together in semantic space. There also needs to be some additional thinking on new merged cluster naming.

## Duplicate names

Hopefully, by generating candidate skill phrases per skill cluster, the number of duplicate skill names decreases. In the event that there are identical skill cluster names, hopefully they sit in different parts of the skill hierarchy. We have now added numbers to identical skill cluster names that will hopefully be manually labelled as a result.

## Skills
