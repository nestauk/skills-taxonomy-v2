# Analysis of skills and the skills taxonomy

This analysis folder contains analysis and experimentation notebooks for 5 themes:

1. `tk_analysis/`
2. `sentence_classifier/`
3. `skills_extraction/`
4. `skills_taxonomy/`
5. `skills_taxonomy_application/`

Outputs (figures and data) from analysis are saved to a corresponding folder in the `outputs/` folder.

## `tk_analysis/`

This folder contains two notebooks:

1. `TextKernel Data.ipynb` - Provides a summary of TextKernel dataset.
2. `TextKernel Data Sample for Skills.ipynb` - Comparison of our sample of TextKernel data to all data.

Outputs are in `outputs/tk_analysis`.

## `sentence_classifier/`

This folder contains three markdown files:

1. `Sentence Splitter Experiments.md` - Notes on experiments for the initial skills classifier
2. `skills_classifier_2.0_experiments.md` - Notes on experiments for the skills classifier after labelling more data
3. `skills_codebook.md` - A codebook for labellers to follow when labelling sentences as either skill sentences or not.

Within `notebooks`, there's a number of `.ipynb` and `.py` files related to experimentation:

1. `Skills Classifier 1.0 - Analyse sentence predictions.py` - initial experiments from skills classifier 1.0
2. `Skills Classifier 1.0 - Doccano Baseline Classifier.py` - initial experiments from skills classifier 1.0
3. `Skills Classifier 1.0 - TextKernel Automated Training Data.py` - initial experiments from skills classifier 1.0
4. `Skills Classifier 1.0 - Sentence Classifier.py` - initial experiments from skills classifier 1.0
5. `Skills Classifier 2.0 - improve_classifier.ipynb` - experiments using more labelled data from skills classifier 2.0

## `skills_extraction/`

In this folder we have two scripts for various bits of analysis and figure plotting after extracting skills:

1. `Effect of sample size.ipynb` - Investigate the effect of sample size of skill sentences and how many words are in the vocab.
2. `Skills Extraction Analysis and Figures.ipynb` - Various analysis and figure generation of the skills extracted. Outputs are in `outputs/skills_extraction/figures/..`

In this folder we also have experimentation notebooks showing 4 approaches for skills extraction approaches, including:

1. Network approach
2. Transformers sentence embeddings approach - no masking
3. Word2vec approach
4. Transformers sentence embeddings approach - masking

The last approach was what we used in the final pipeline (refactored in `skills_taxonomy_v2/pipeline/skills_extraction/`).

## `skills_taxonomy/`

In this folder we perform some analysis on the skills taxonomy created when running `skills_taxonomy_v2/pipeline/skills_taxonomy/build_taxonomy.py`.

1. `Evaluate hierarchy.ipynb` - Evaluate hierarchy based on popular skill groups for job titles. Output csvs stored in `outputs/skills_taxonomy/evaluation/`.
2. `Tranversal Skills.ipynb` - Identify the most and least transversal skills and skill groups. Outputs in `outputs/skills_taxonomy/transversal/`.
3. `Renaming sample of skill groups.ipynb` - Manually creating names for some of the skill groups, outputs `skills_taxonomy_v2/utils/2021.09.06_level_a_rename_dict.json` which is used in other notebooks.
4. `Skills Taxonomy Analysis and Figures.ipynb` - this notebook provides some visualisation and analysis of it. Outputs are stored in `outputs/skills_taxonomy/figures/2021.09.06/`.
5. `Plot interactive hierarchy.ipynb` - this notebook plot interactive plots of the hierarchy for interrogation.

## `skills_taxonomy_application/`

This folder contains two notebooks to analyse the skills taxonomy in application to the job location and whether the job was advertise pre or post COVID.

1. `Application - Geography.ipynb` See how different skill groups in the taxonomy relate with location of the job advert. Outputs in `outputs/skills_taxonomy_application/region_application/`.
2. `Application - COVID.ipynb` See how different skill groups in the taxonomy relate with whether the job advert was out pre or post COVID. Outputs in `outputs/skills_taxonomy_application/covid_application`.

This folder also contains the script `locations_to_nuts2.py` to convert longitude and latitude coordinates from the job adverts to NUTS2 regional classifications - this was neccessary for use in `Application - Geography.ipynb` .
