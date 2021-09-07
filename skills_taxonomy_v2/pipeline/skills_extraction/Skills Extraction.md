# Skills Extraction

The aim of this pipeline is to extract skills from job adverts. There are 4 steps:

1. Get embeddings from skill sentences.
2. Reduce and cluster the embeddings - these are the skills.
3. Find names for each skill.
4. Get an ESCO-TK skills index mapper.

The parameters for all these steps can be found in the config path `skills_taxonomy_v2/config/skills_extraction/2021.08.31.yaml`.

## Experimentation summary - 2021.08.31.yaml

- Step 0: Predict skill sentences. 87 random files of 10,000 job adverts. Found around 4,000,000 skill sentences.
- Step 1: Get embeddings for each skill sentence. Get embeddings for sentences in the first 10,000 job adverts from each of the 87 files, remove sentences with only masking.
- Step 2: Get skills from clustering. Removed sentences with too much masking, remove repeated sentences, and only keep sentences within a certain length bound - 318,772 sentences. Reduce embeddings from 384 to 2 dimensions. Clustered into 18,644 skills, the proportion of data points not put into a cluster was 0.29.

Finding skill names and ESCO links wasn't completed for this run.

Some preliminary analysis of the skills extraction steps (e.g. how many skills per job advert, common skills) can be found in `analysis/skills_extraction/notebooks/Skills Extraction Figures-31stAug.ipynb`.


## Experimentation summary - 2021.08.02.yaml

This experiment was run prior to changes in the codebase, so the config file will no longer run.

- Step 0: Predict skill sentences. 10 random files of 10,000 job adverts. Found 5,823,903 skill sentences.
- Step 1: Get embeddings for each skill sentence. Get embeddings for sentences in the first 10,000 job adverts from each of the 10 files, remove sentences with only masking. About 400,000 sentences with embeddings.
- Step 2: Get skills from clustering. Removed sentences with too much masking and remove repeated sentences - 392,625 sentences. Clustered into 24,912 skills, the proportion of data points not put into a cluster was 0.108.
- Step 3: 19796 out of 24912 (79%) TK skills were linked with ESCO skills, 0 of these were linked to multiple ESCO skills. 4301 out of 13958 (31%) ESCO skills were linked with TK skills, 2450 of these were linked to multiple TK skills.

This experimentation yields skills found in 78,674 job adverts - ready for the taxonomy step.

Some preliminary analysis of the skills extraction steps (e.g. how many skills per job advert, common skills) can be found in `analysis/skills_extraction/notebooks/Skills Extraction Figures.ipynb`.

## Step 1: Get embeddings for each skill sentence

Input : Skill sentences from TK job adverts.

Output : Embeddings for each sentence.

#### Details

In the previous step in the pipeline (Sentence Classifier) we extracted skill sentences from job adverts. With these we want to clean them (mask out proper nouns etc), and output embeddings for each of them.

This is done by running:

```
python -i skills_taxonomy_v2/pipeline/skills_extraction/get_sentence_embeddings.py --config_path 'skills_taxonomy_v2/config/skills_extraction/2021.08.31.yaml'
```

## Step 2: Extract skills using clustering

Input : Embeddings for each sentence.

Output : Which skill each sentence has been clustered into and its reduced embedding.

#### Details

- Remove data points where the sentence has too much masking
- Left with keywords for each sentence - assumption that these words are the essence of a skill, e.g. 'This was once a technical sentence about a engineering skill'-> 'technical engineering'
- umap embeddings reduction to 2D
- DBSCAN clustering of 2D reduced data
- Output sentences grouped into clusters - assumption that these are each a single skill. e.g. in once cluster 'technical engineering', 'software engineer'
- To examine the sample size of sentences used in clustering we also calculate the cumulative vocab size as more sentences are added. 

This is done by running:

```
python -i skills_taxonomy_v2/pipeline/skills_extraction/extract_skills.py --config_path 'skills_taxonomy_v2/config/skills_extraction/2021.08.31.yaml'
```

Outputs:

- The sentences data including the embedding reduction and which cluster/skill the sentence was in - `outputs/skills_extraction/skills_extraction_larger/2021.08.31_sentences_data.json`
- The centroids of each cluster `outputs/skills_extraction/skills_extraction_larger/2021.08.31_cluster_centroids.json`
- The reducer class the created the 2D representation of the embeddings `outputs/skills_extraction/skills_extraction_larger/2021.08.31_reducer_class.pkl`
- The cumulative vocabulary size with each sentence added `outputs/skills_extraction/skills_extraction_larger/2021.08.31_num_sentences_and_vocab_size.json`

## Step 3: Get skills names

Input : The sentences clustered into skills and the embeddings for the sample of sentences used to create these.

Output : Skill names, key examples and cleaned texts for each skill.

#### Details

- With 'essence of skill' words from 'sentences' - lemmatize, get n-grams, remove non-acronym capitals, remove duplicates
- The skill name is the top TFIDF words for each cluster
- Description is the 5 closest sentences to the centroid of the cluster

This is done by running:

```
python -i skills_taxonomy_v2/pipeline/skills_extraction/skills_naming.py --config_path 'skills_taxonomy_v2/config/skills_extraction/2021.08.31.yaml'
```

Outputs:

- The skills data - `outputs/skills_extraction/extracted_skills/2021.08.31_skills_data.json`


## Step 4: Find ESCO mappings

Input : ESCO skills and our bottom up skills

Output : A dictionary of how the two sets of skills can be mapped

This is done by running:

```
python -i skills_taxonomy_v2/pipeline/skills_extraction/esco_skills_mappings.py --config_path 'skills_taxonomy_v2/config/skills_extraction/2021.08.31.yaml'
```

Outputs:

- The ESCO ID to TK ID dictionary mapper - `s3://skills-taxonomy-v2 + outputs/skills_extraction/data/2021.08.02_esco2tk_mapper.json`
- The TK ID to ESCO ID dictionary mapper - `s3://skills-taxonomy-v2 + outputs/skills_extraction/data/2021.08.02_tk2esco_mapper.json`
- A dictionary of ESCO skills, this is because ESCO doesn't have a numerical unique identifier, so we save one out for use in linking back to traditional ESCO identifier - `s3://skills-taxonomy-v2/outputs/skills_extraction/data/2021.08.02_esco_ID2skill.json`
