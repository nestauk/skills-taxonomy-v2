This analysis folder contains analysis and experimentation notebooks.

Outputs (figures and data) from analysis are saved to a corresponding folder in the `outputs/` folder.


## `esco_skills`

Can delete?

## `tk_analysis`

- Summary of TextKernel dataset
- Comparison of our sample of TextKernel data to all data

Outputs in `outputs/tk_analysis`

## `sentence_classifier`

## `skills_extraction`

In this folder we have two scripts for various bits of analysis and figure plotting after extracting skills:
1. `Effect of sample size.ipynb` - Investigate the effect of sample size of skill sentences and how many words are in the vocab.
2. `Skills Extraction Analysis and Figures.ipynb` - Various analysis and figure generation of the skills extracted. Outputs are in `outputs/skills_extraction/figures/..`

In this folder we also have experimentation notebooks showing 4 approaches for skills extraction approaches, including:
1. Network approach
2. Transformers sentence embeddings approach - no masking
3. Word2vec approach
4. Transformers sentence embeddings approach - masking

The last approach was what we used in the final pipeline (refactored in `skills_taxonomy_v2/pipeline/skills_extraction/`).

## `skills_taxonomy`

In this folder we perform some analysis on the skills taxonomy created when running `skills_taxonomy_v2/pipeline/skills_taxonomy/build_taxonomy.py`.

1. `Evaluate hierarchy.ipynb` - Evaluate hierarchy based on popular skill groups for job titles. Output csvs stored in `outputs/skills_taxonomy/evaluation/`.
2. `Tranversal Skills.ipynb` - Identify the most and least transversal skills and skill groups. Outputs in `outputs/skills_taxonomy/transversal/`.
3. `Renaming sample of skill groups.ipynb` - Manually creating names for some of the skill groups, outputs `skills_taxonomy_v2/utils/2021.09.06_level_a_rename_dict.json` which is used in other notebooks.
4. `Skills Taxonomy Analysis and Figures.ipynb` - this notebook provides some visualisation and analysis of it. Outputs are stored in `outputs/skills_taxonomy/figures/2021.09.06/`.




