# Skills Extraction

The aim of this pipeline is to extract skills from job adverts. There are 3 steps:

1. Get embeddings from skill sentences.
2. Reduce, cluster and find names for each cluster - these are the skills.
3. Get an ESCO-TK skills index mapper.

The parameters for all these steps can be found in the config path `skills_taxonomy_v2/config/skills_extraction/2021.08.02.yaml`.

## Experimentation summary (pre-scaling up to process all job adverts)

- Step 0: Predict skill sentences. 10 random files of 10,000 job adverts. Found 5,823,903 skill sentences.
- Step 1: Get embeddings for each skill sentence. Get embeddings for sentences in the first 5000 job adverts from each of the 10 files, remove sentences with only masking. 208,087 sentences with embeddings.
- Step 2: Get skills from clustering. Removed sentences with too much masking and remove repeated sentences - 207,303 sentences. Clustered into 4573 skills which gave a siloutte score of 0.13, the proportion of data points not put into a cluster was 0.035.
- Step 3: 3275 out of 4594 TK skills were linked with ESCO skills, 0 of these were linked to multiple ESCO skills. 1731 out of 13958 ESCO skills were linked with TK skills, 642 of these were linked to multiple TK skills.

This experimentation yields skills found in 41,508 job adverts - ready for the taxonomy step.

## Step 1: Get embeddings for each skill sentence

Input : Skill sentences from TK job adverts.

Output : Embeddings for each sentence.

#### Details

In the previous step in the pipeline (Sentence Classifier) we extracted skill sentences from job adverts. With these we want to clean them (mask out proper nouns etc), and output embeddings for each of them.

This is done by running:

```
python -i skills_taxonomy_v2/pipeline/skills_extraction/get_sentence_embeddings.py --config_path 'skills_taxonomy_v2/config/skills_extraction/2021.08.02.yaml'
```

## Step 2: Extract skills using clustering

Input : Embeddings for each sentence.

Output : Skill names, descriptions and the sentence IDs that went in there.

#### Details

- Remove data points where the sentence has too much masking
- Left with keywords for each sentence - assumption that these words are the essence of a skill, e.g. 'This was once a technical sentence about a engineering skill'-> 'technical engineering'
- umap embeddings reduction to 2D
- DBSCAN clustering of 2D reduced data
- Output sentences grouped into clusters - assumption that these are each a single skill. e.g. in once cluster 'technical engineering', 'software engineer'
- With 'essence of skill' words from 'sentences' - lemmatize, get n-grams, remove non-acronym capitals, remove duplicates
- The skill name is the top TFIDF words for each cluster
- Description is the 5 closest sentences to the centroid of the cluster

This is done by running:

```
python -i skills_taxonomy_v2/pipeline/skills_extraction/extract_skills.py --config_path 'skills_taxonomy_v2/config/skills_extraction/2021.08.02.yaml'
```

Outputs:

- The skills data - `outputs/skills_extraction/data/2021.08.02_skills_data.json`
- The sentences data including the embedding reduction and which cluster/skill the sentence was in - `outputs/skills_extraction/data/2021.08.02_sentences_data.json`

## Step 3: Find ESCO mappings

Input : ESCO skills and our bottom up skills

Output : A dictionary of how the two sets of skills can be mapped

This is done by running:

```
python -i skills_taxonomy_v2/pipeline/skills_extraction/esco_skills_mappings.py --config_path 'skills_taxonomy_v2/config/skills_extraction/2021.08.02.yaml'
```

Outputs:

- The ESCO ID to TK ID dictionary mapper - `s3://skills-taxonomy-v2 + outputs/skills_extraction/data/2021.08.02_esco2tk_mapper.json`
- The TK ID to ESCO ID dictionary mapper - `s3://skills-taxonomy-v2 + outputs/skills_extraction/data/2021.08.02_tk2esco_mapper.json`
- A dictionary of ESCO skills, this is because ESCO doesn't have a numerical unique identifier, so we save one out for use in linking back to traditional ESCO identifier - `s3://skills-taxonomy-v2/outputs/skills_extraction/data/2021.08.02_esco_ID2skill.json`
