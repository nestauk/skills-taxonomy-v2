# Skills Extraction

The aim of this pipeline is to extract skills from job adverts. There are 7 scripts to run this pipeline:

1. Get embeddings from skill sentences ([`metaflow/flow.py`](metaflow/flow.py)).
2. Get a sample of the embeddings ([`get_embeddings_data_sample.py`](get_embeddings_data_sample.py)).
3. Reduce the embeddings ([`reduce_embeddings.py`](reduce_embeddings.py)).
4. Cluster the reduced embeddings to find the skills ([`cluster_embeddings.py`](cluster_embeddings.py)).
5. Get average embeddings for each skill cluster ([`skills_naming_embeddings.py`](skills_naming_embeddings.py)).
6. Find names for each skill ([`skills_naming.py`](skills_naming.py)).
7. Get duplicate sentences dictionary for analysis pieces ([`get_duplicate_sentence_skills.py`](get_duplicate_sentence_skills.py)).

Some util functions are stored in:
- cleaning_sentences.py
- extract_skills_utils.py

And some legacy (and unused) code from previous iterations are in:
- extract_skills.py
- esco_skills_mappings.py
- get_sentence_embeddings.py
- get_sentence_embeddings_utils.py

The parameters for all these steps can be found in the config path directory `skills_taxonomy_v2/config/skills_extraction/`.

<img src="../../../outputs/reports/figures/Jan 2022/extract_skill_methodology_overview.jpg" width="700">

# Logs of results from different config files

## January 2022

These are the results from the [`2022.01.14.yaml`](https://github.com/nestauk/skills-taxonomy-v2/blob/7059e46116daf93051347557c7bb69e7e3de64ab/skills_taxonomy_v2/config/skills_extraction/2022.01.14.yaml) config file.

### 1. Finding sentence embeddings

We found the embeddings using the same method as from November 2021 below, but to quicken things up this was done in Metaflow with Batch, running:
```
python skills_taxonomy_v2/pipeline/skills_extraction/metaflow/flow.py --environment=conda --datastore=s3 run
```
Embeddings for 19,755,486 skill sentences from 4,118,467 job adverts were found this way.

A few things are filtered out, e.g. sentences over 20 tokens, if the sentences is all masked words, which left us with 19,134,468 sentences with embeddings.

### 2. Reducing sentence embeddings

We used our analysis findings as found before. Since only 14% of the data has changed, and we suspect the changes not to be too drastic. As such we carried over the findings that:
- A sample size of 300k embeddings is enough to fit the reducer class to
- Only using sentences less than 250 sentences (over this will be likely to be multi-skill)
- Clusterable 2D reduced embeddings will be found with n_neighbors = 6, and min_dist = 0.0

Thus, we get our new sample to fit on by running: `skills_taxonomy_v2/pipeline/skills_extraction/get_embeddings_data_sample.py` (note, this used to be in the analysis folder, but since its output is used in the `reduce_embeddings.py` script we moved it to the pipeline folder). This takes a random 2000 embeddings from each file, then filters out any where the original sentence was a repeat or over 250 characters.

- In the sample - there are 723,657 embeddings [from embeddings_sample]
- In the sample - there are 723,657 unique sentences with embeddings where the sentences is <250 characters long
- In the sample - there were 197,366 sentences which were too long to be included (>250 characters long)

We then run 
```
python skills_taxonomy_v2/pipeline/skills_extraction/reduce_embeddings.py --config_path skills_taxonomy_v2/config/skills_extraction/2022.01.14.yaml
```
This process left us with reduced embeddings for 10,378,654 sentences from 2,624,694 job adverts (filtered to only include sentences <250 characters and no repeated sentences).


### 3. Clustering reduced sentence embeddings

Again, we used the same parameters as in the Novemeber 2021 run to cluster our reduced embeddings into skills, namely:
- Only use sentences <100 words
- dbscan_eps = 0.01
- dbscan_min_samples = 4
- Fit our clustering algorithm on 300,000 random <100 character sentences - this created 11332 clusters. 
- For the 8598 clusters found with less than 10 sentences in, we iteratively merged nearest neighbours when the Eucliean distance was less than 0.05. 
- This resulted in us finding 6686 skill clusters. 

```
python skills_taxonomy_v2/pipeline/skills_extraction/cluster_embeddings.py --config_path skills_taxonomy_v2/config/skills_extraction/2022.01.14.yaml
```

Out of the 10,378,654 sentences in our sample, 3,633,001 had under 100 characters - using these we then went about predicting clusters using the centroids from these 6686 clusters. 

We use the predicted clusters for all 3,633,001 sentences as our 6686 skills.

- There are 10,378,654 unique sentences with skills in (inc -2 skill), these are from 2,624,694 unique job adverts
- There are 3,633,001 unique sentences with skills in (not inc -2 skill), these are from 1,287,661 unique job adverts

122696 out of 187576 (65%) of predictions were the same as original clusters for sample clustering was fitted to (only when merged cluster != -1).

Outputs:
- `s3://skills-taxonomy-v2/outputs/skills_extraction/extracted_skills/2022.01.14_sentences_skills_data.json` (each sentences ID/file ID with which cluster it was predicted to be in, and if in the training set - also which cluster it was originally assigned to) 
- `s3://skills-taxonomy-v2/outputs/skills_extraction/extracted_skills/2022.01.14_sentences_skills_data_lightweight.json` (the same as above but in a list of list form (in order ['job id', 'sentence id',  'Cluster number predicted']), not a dictionary, to save memory)
- `s3://skills-taxonomy-v2/outputs/skills_extraction/extracted_skills/2022.01.14_skills_data.json` (a dictionary of skill numbers and the sentences in them, along with the skill centroid coordinate).

Some plots of the skills in 2D space are done in `Extracted skills - 2022.01.14.ipynb`.

In `Skills Extraction Analysis and Figures.ipynb` we find:

- The mean number of sentences for each skills is 543.4556469708302
- The median number of sentences for each skills is 366.0
- There are 6153 skills with more than 200 sentences

### 4. Skills naming

Need to first get the average embedding for each skill with:
```
python skills_taxonomy_v2/pipeline/skills_extraction/skills_naming_embeddings.py
```
Then:
```
python skills_taxonomy_v2/pipeline/skills_extraction/skills_naming.py --config_path 'skills_taxonomy_v2/config/skills_extraction/2022.01.14.yaml'
```

This will output `outputs/skills_extraction/extracted_skills/2022.01.14_skills_data_named.json`.

### 5. Duplicate sentences

In reduce_embeddings.py we filter out duplicated sentences. However, for analysis to do with the job adverts we need all the duplicates included. e.g. if the same sentence is used in two job adverts only one of them is brought forward, and thus the analysis will miss out including the second job advert in the counts.

Running:
```
python skills_taxonomy_v2/pipeline/skills_extraction/get_duplicate_sentence_skills.py

```
we save out several intermediatary files containing dictionaries of {'job id': [[words_id, sent_id], [words_id, sent_id]]} in `outputs/skills_extraction/word_embeddings/data/{file_date}_words_id_list_0.json`. This script also combines them all into one smaller and more useful dictionary which only includes the duplicated sentences in `outputs/skills_extraction/word_embeddings/data/2022.01.14_unique_words_id_list.json`.

Note: 'words_id' is different to 'sent_id' since it is the unique identifier for the sentences with masked words removed rather than the unique identifier for the original sentence (as sent_id is).

## November 2021

These are the results from the [`2021.11.05.yaml`](https://github.com/nestauk/skills-taxonomy-v2/blob/7059e46116daf93051347557c7bb69e7e3de64ab/skills_taxonomy_v2/config/skills_extraction/2021.11.05.yaml) config file.

### 1. Finding sentence embeddings

First we found embeddings for 4,312,285 job advert (edit: this may be 3,572,140 job adverts with skill sentences) using the `all-MiniLM-L6-v2` model: `get_sentence_embeddings.py --config_path skills_taxonomy_v2/config/skills_extraction/2021.11.05.yaml`.

We then had a look at whether sentence length can predict whether the sentence contains one skill or not, and whether the sentence is indeed one sentences or multiple sentences not split up well. This involved manually tagging a sample of sentences of different lengths. This analysis is discussed in `Multi-skill sentences.ipynb` and led us to believe that sentences over 250 characters, around 80% of sentences are less than 250 characters.

### 2. Reducing sentence embeddings

Since the sample of 4,312,285 job advert was too much data for the reduction class to process, we randomly sampled the embeddings to find a good point for which increasing the sample size doesn't effect the output. This sample included up to 2000 sentence embeddings from each data file, as long as the sentence length was less than 250 characters long. In total this sample had 742,771 embeddings. This sampling process is done in the script `skills_taxonomy_v2/analysis/skills_extraction/get_embeddings_data_sample.py`.

In the notebook `Experiment - Data reduction sample size.ipynb` we then found the minimum sample size for which similarities in the data reduction stabilises. This iterative algorithm is as follows:
- Step 1: Take out a hold out set of 10,000 embeddings
- Step 2: Reduce a set of the none-hold out embeddings
- Step 3: Find the nearest neighbour for each of the hold out embeddings.
- Step 4: Increase the size of the set of none-hold out embeddings
- Step 5: Find the size of the intersection of nearest neighbours from the previous set to the new set
- Step 6: Repeat steps 2-5 until you have exhausted all the none-hold out embeddings

This process suggested that a sample size of 300k embeddings is enough to use to fit our reducer class.

We then experimented with the parameters for the reducer class, through a subjective process of trying out different values for the UMAP number of neighbours and minimum distance parameters, we looked at the 2D data reduction and decided whether it looked in a suitable state for clustering. For some parameter combinations the local structure was very defined (i.e. there were clear clusters of skills, but their close neighbourhoods included very different skills), and for other parameter combinations there was a better global structure (i.e. the skill topics similar to each other were close by, but there was clear clusters). A balance of nicely clustered sentences which had a relationship with their neighbouring sentences, were found with n_neighbors = 6, and min_dist = 0.0. To make this process easier we always reduced to 2 dimensions.

Using these parameters, namely:
- Fit the reducer to a random sample of 300k sentences
- Don't include sentences longer than 250 characters
- umap_n_neighbors = 6
- umap_min_dist = 0.0
- umap_n_components = 2

We found reduced embeddings for the 4,312,285 job adverts. This was done by running `reduce_embeddings.py --config_path skills_taxonomy_v2/config/skills_extraction/2021.11.05.yaml`. This process left us with reduced embeddings for 4,097,008 sentences from 1,012,869 job adverts.

### 3. Clustering reduced sentence embeddings

Since there is a trade-off between the local and global structure to the data reduction, we found quite a lot of sentences were in the middle of the plot of 2D reduced embeddings and were quite vague with what the skill was about. This also correlated to the size of the sentence, where they were generally longer in the centre. This 'mega-cluster' of not very linked sentences caused problems in finding optimal clustering parameters, and thus we opted to remove some of it by another sentence length threshold. We decided a threshold of 100 characters solved our problem, which reduced our 4,097,008 sentences to 1,465,639 sentences.

In `skills_taxonomy_v2/analysis/skills_extraction/notebooks/Experiment - Clustering parameters.py` we experiment with the parameters for clustering. After experimenting with different parameters and trying to optimise the following criteria:
- Shouldn't be too many clusters (last time 20k was a lot) - between 7000 and 15000
- Shouldn't be any massive clusters - 0 clusters of size >10,000 sentences
- Shouldn't be too many small clusters - average size of clusters is >10
- Number of sentences not clustered (the "-1 cluster") < 200,000

We decided that `dbscan_eps = 0.01` and `dbscan_min_samples = 4` were good parameters for these metrics. When fit to 300000 random sentences this produces:
- 11551 clusters
- 0 clusters which are larger than 10,000 sentences
- 8892 clusters which have <10 sentences
- 117,923 sentences not clustered
- Average size of cluster is 16 sentences


By looking into some of these clusters we felt like some of the small ones could be merged together further.
For clusters made up of less than 10 sentences we found the nearest neighbouring cluster, with 10 sentences from the nearest neighbour we manually tagged whether we thought the clusters should be merged or not, and whether the small cluster was a "good" skill cluster or not.

Some examples of this are below:

|Small cluster sentences | 10 sentences from the nearest neighbour cluster | Centroid distance between them | Should merge? | Small cluster is a skill?|
|---|---|---|---|---|
|'demonstrates a can do attitude to work', 'motivated adaptable hardworking and imaginative', 'strong work ethic and can do attitude', 'are you enthusiastic and motivated with a positive outlook', 'must have a caring nature and positive personality', 'must have a mature attitude and be reliable' | 'be reliable and have a friendly attitude', 'a caring dedicated and can do attitude', 'organised and disciplined approach', 'candidate will possess a passionate attitude towards teaching', 'attitude and aptitude are the most important attributes' | 0.014343634 | MERGE | GOOD CLUSTER |
|'good organisation skills and ability to work on several projects at the same time', 'this role will involve working on a number of high value projects in central scotland', 'you must be flexible willing to carry out any activities in support of the transition projects', 'with scope to temporarily increase hours of work if oneoff projects are in need of this', 'ensuring that the hr database including employee documentation is accurately maintained' | 'ba will perform the role of a coordinator between the system development team and end users', 'supervision of contractors while on site 5', 'contribute to the delivery of the camden plan and the five year forward view', 'this means we can take on more significant and complex projects', 'responsibilities responsible for supporting the development of the sites lean strategy', 'you will also be responsible for the delivery of specific projects within the team', 'requirements you will have worked in a planning role previously for a building contractor', 'ideally you will be chartered and have experience of taking projects from the start until completion', 'to work above and beyond when required to delivery projects to successful completion'| 0.020053591|DON'T MERGE| NOT GOOD CLUSTER|
|'knowledge of latest cdm regulations', 'detailed knowledge of cdm regulations', 'ensuring client responsibilities are discharged with regards to cdm regulations', 'knowledge of cdm regulations is essential', 'a solid understanding of hs and cdm regulations and their application', 'knowledge of cdm regulations and nrswa', 'has a good knowledge of the cdm regulations', 'comprehensive knowledge of cdm regulations', 'full compliance with cdm regulations'|'the position will include assisting with the folowing', 'required skills 3 years experience in similar position', 'strong experience in a similar managementleadership position', 'experience in a similar position is essential', 'skills experience proven experience within a similar position', 'experience in a similar position is required', 'previous managementsupervisory experience is essential for this position', 'previous experience in a similar position is also essential'|0.576391757|DON'T MERGE| GOOD CLUSTER|

Using this data, we found that setting a distance threshold of 0.05 to merge clusters gave quite good results, achieving an accuracy of 70% on the 139 labelled data points. This analysis was performed in `Effect of merging clusters distance threshold.ipynb`.

||Should be merged| Shouldn't be merged|
|---|---|---|
|Predicted to merge|55 (40%)|12 (9%)|
|Prediction to not merge|30 (22%)|42 (30%)|

Furthermore, for the 12 times the clusters shouldn't be merged but werem, 11 of these weren't good skill clusters in the first place.

|Should merge  |Merge prediction  |Small cluster is a good skill cluster?|
|---|---|---|
|Don't merge|Don't merge|NOT GOOD CLUSTER|10|
|Don't merge|Don't merge|GOOD CLUSTER|32|
|Don't merge|Merge|NOT GOOD CLUSTER|11|
|Don't merge|Merge|GOOD CLUSTER|1|
|Merge|Don't merge|NOT GOOD CLUSTER|13|
|Merge|Don't merge|GOOD CLUSTER|17|
|Merge|Merge|NOT GOOD CLUSTER|19|
|Merge|Merge|GOOD CLUSTER|35|

We thus fit our clustering algorithm on 300,000 random <100 character sentences - this created 11551 clusters. Then for the 8892 clusters found with less than 10 sentences in, we iteratively merged nearest neighbours when the Eucliean distance was less than 0.05. This resulted in us finding 6784 skill clusters. 

Out of the 4,097,008 sentences in our sample, 1,465,639 had under 100 characters - using these we then went about predicting clusters using the centroids from these 6784 clusters. We use the predicted clusters for all 1,465,639 sentences as our 6784 skills. Note that in 35% of the sentences which were used in the fitting of the clustering algorithm, the predicted cluster was different to the cluster it had been assigned in the fitting. Analysis a figure plotting of these clusters is given in `Extracted skills - 2021.11.05.ipynb` and creating these clusters was found by running `cluster_embeddings.py --config_path skills_taxonomy_v2/config/skills_extraction/2021.11.05.yaml`. This saves two files: `s3://skills-taxonomy-v2/outputs/skills_extraction/extracted_skills/2021.11.05_sentences_skills_data.json` (each sentences ID/file ID with which cluster it was predicted to be in, and if in the training set - also which cluster it was originally assigned to) and `s3://skills-taxonomy-v2/outputs/skills_extraction/extracted_skills/2021.11.05_skills_data.json` (a dictionary of skill numbers and the sentences in them, along with the skill centroid coordinate).

### 4. Skills naming

This has also been improved (see `skills_taxonomy_v2/analysis/skills_extraction/Skill Naming Experiments.md` for details).

It can be run with:
```
python skills_taxonomy_v2/pipeline/skills_extraction/skills_naming.py --config_path 'skills_taxonomy_v2/config/skills_extraction/2021.11.05.yaml'
```

This will output `outputs/skills_extraction/extracted_skills/2021.11.05_skills_data_named.json`.

## August 2021

These are the results from the [`2021.08.31.yaml`](https://github.com/nestauk/skills-taxonomy-v2/blob/7059e46116daf93051347557c7bb69e7e3de64ab/skills_taxonomy_v2/config/skills_extraction/2021.08.31.yaml) config file.

- Step 0: Predict skill sentences. 87 random files of 10,000 job adverts. Found around 4,000,000 skill sentences.
- Step 1: Get embeddings for each skill sentence. Get embeddings for sentences in the first 10,000 job adverts from each of the 87 files, remove sentences with only masking.
- Step 2: Get skills from clustering. Removed sentences with too much masking, remove repeated sentences, and only keep sentences within a certain length bound - 322,071 sentences. Reduce embeddings from 384 to 2 dimensions. Clustered into 18,894 skills, the proportion of data points not put into a cluster was 0.28. After removed sentences not clustered there are 232,394 sentences.

Finding skill names and ESCO links wasn't completed for this run.

Some preliminary analysis of the skills extraction steps (e.g. how many skills per job advert, common skills) can be found in `analysis/skills_extraction/notebooks/Skills Extraction Analysis and Figures.ipynb`.

## Step 1: Get embeddings for each skill sentence

Input : Skill sentences from TK job adverts.

Output : Embeddings for each sentence.

#### Details

In the previous step in the pipeline (Sentence Classifier) we extracted skill sentences from job adverts. With these we want to clean them (mask out proper nouns etc), and output embeddings for each of them.

This is done by running:

```
python -i skills_taxonomy_v2/pipeline/skills_extraction/get_sentence_embeddings.py --config_path 'skills_taxonomy_v2/config/skills_extraction/2021.08.31.yaml'
```

Outputs:

Outputs are stored within the `outputs/skills_extraction/word_embeddings/data/2021.08.31/` folder in the same folder structure as the skill sentences were saved in. For each file of job adverts two outputs are stored, one with the embeddings (ending in `_embeddings.json`) and one with the original sentences (ending in `_original_sentences.json`).

For example, the embeddings for the file of skill sentences in `outputs/sentence_classifier/data/skill_sentences/2021.08.16/textkernel-files/semiannual/2021/2021-04-01/jobs_new.5_2021.08.16.json` are stored in `outputs/skills_extraction/word_embeddings/data/2021.08.31/semiannual/2021/2021-04-01/jobs_new.5_2021.08.16_embeddings.json`

## Step 2: Extract skills using clustering

Input : Embeddings for each sentence.

Output : Which skill each sentence has been clustered into and its reduced embedding.

#### Details

- Remove data points where the sentence has too much masking
- Left with keywords for each sentence - assumption that these words are the essence of a skill, e.g. 'This was once a technical sentence about a engineering skill'-> 'technical engineering'
- UMAP embeddings reduction to 2D
- DBSCAN clustering of 2D reduced data
- Output sentences grouped into clusters - assumption that these are each a single skill. e.g. in once cluster 'technical engineering', 'software engineer'
- To examine the sample size of sentences used in clustering we also calculate the cumulative vocab size as more sentences are added.

This is done by running:

```
python -i skills_taxonomy_v2/pipeline/skills_extraction/extract_skills.py --config_path 'skills_taxonomy_v2/config/skills_extraction/2021.08.31.yaml'
```

Outputs:

- The sentences data including the embedding reduction and which cluster/skill the sentence was in - `outputs/skills_extraction/extracted_skills/2021.08.31_sentences_data.json`
- The centroids of each cluster `outputs/skills_extraction/extracted_skills/2021.08.31_cluster_centroids.json`
- The reducer class the created the 2D representation of the embeddings `outputs/skills_extraction/extracted_skills/2021.08.31_reducer_class.pkl`
- The cumulative vocabulary size with each sentence added `outputs/skills_extraction/extracted_skills/2021.08.31_num_sentences_and_vocab_size.json`
- A dictionary of sentence id to embeddings `outputs/skills_extraction/extracted_skills/2021.08.31_sentence_id_2_embedding_dict.json.gz`
    )

## Step 3: Get skills names

Input : The sentences clustered into skills and the embeddings for the sample of sentences used to create these.

Output : Skill names, key examples and cleaned texts for each skill.

#### Details

- Finds the skills name - The closest single ngram to the centroid of all the sentence embeddings which were clustered to create the skill using cosine similarity.
- Also finds the original sentences which are closest to the centroid of the skill cluster and all the cleaned sentences that went into creating the skill cluster.

This is done by running:

```
python -i skills_taxonomy_v2/pipeline/skills_extraction/skills_naming.py --config_path 'skills_taxonomy_v2/config/skills_extraction/2021.08.31.yaml'
```

Outputs:

- The skills data - `outputs/skills_extraction/extracted_skills/2021.08.31_skills_data.json`


## Step 4: Find ESCO mappings

Note: This code needs updating and doesn't currently work.

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
