# A New Approach to Building a Skills Taxonomy

The full technical report and blog article for this project will be released soon (link to follow).

## Introduction

There is no official and fully open skills taxonomy in the UK. There is a really important need for such a taxonomy that would enable consistent conceptualisation of workforce skills, together with consistent terminology and language around skills used by educators, careers advisers, policy makers and employers. The lack of a consistent language has multiple consequences such as creating confusion over the skills required for particular roles or the training needs of employees. At the same time, the effects of COVID-19 and Brexit have triggered rapid changes in skill demands as well as new skill shortages. This shifting landscape has only increased the need for an open and up-to-date skills taxonomy for the UK which could help to provide better quality and up to date information, in turn to better inform policy.

Therefore, in partnership with the [Economic Statistics Centre of Excellence (ESCoE)](https://www.escoe.ac.uk/), we are releasing an updated skills taxonomy that is more open, more up-to-date and methodologically refined.

This repo contains the source code for this project.

An overview of the methodology, coloured by the three main steps to the pipeline, can be visualised below:

![](./outputs/reports/figures/Jan%202022/methodology_overview_pipeline.jpg)

### The taxonomy file

The taxonomy file is given [here](./outputs/taxonomy_data/2022.01.21_hierarchy_structure_named.json). To view this JSON file in a friendly format, you should download it and open it using Firefox. Alternatively, you could also use an online tool such as [JSON formatter](https://jsonformatter.org/json-viewer).

### Pipeline steps

More details of the steps included in this project, and running instructions, can be found in their respective READMEs:

1. [tk_data_analysis](skills_taxonomy_v2/pipeline/tk_data_analysis/README.md) - Get a sample of the TextKernel job adverts.
2. [sentence_classifier](skills_taxonomy_v2/pipeline/sentence_classifier/README.md) - Training a classifier to predict skill sentences.
3. [skills_extraction](skills_taxonomy_v2/pipeline/skills_extraction/README.md) - Extracting skills from skill sentences.
4. [skills_taxonomy](skills_taxonomy_v2/pipeline/skills_taxonomy/README.md) - Building the skills taxonomy from extracted skills.

### Analysis

This repository also contains various pieces of analysis of the taxonomy. These are discussed in the main analysis [README file](skills_taxonomy_v2/analysis/README.md).

<img src="./outputs/reports/figures/Jan 2022/hierarchy_numbers.jpg" width="700">

#### Examples of the hierarchy
<img src="./outputs/reports/figures/Jan 2022/taxonomy_example.jpg" width="700">


## Running the code

This repository has been made public in the interest of openness, and hopefully that some of the scripts and functions it contains may be useful for others. However, the TextKernel dataset of job adverts is not available for use anymore (either by Nesta staff or the general public). Because of this, the pipeline can no longer be run from start to finish.

### Conda environment

When you are running scripts from this repo for the first time you need to create the environment by running `make conda-create` to create the conda environment. Then everytime after this you can activate it using `conda activate skills-taxonomy-v2`. If you update the requirements then run `make conda-update`.

As a one off, if needed, you will also have to run:

```
conda install pytorch torchvision torchaudio -c pytorch
conda install -c conda-forge spacy==3.0.0
python -m spacy download en_core_web_sm
conda install cdlib=0.2.3
```
and
```
conda install -c anaconda py-xgboost
```
or, if you aren't using anaconda:
```
conda install -c conda-forge py-xgboost
```


## Contributor guidelines

[Technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

---

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
(<a href="http://nestauk.github.io/ds-cookiecutter">Read the docs here</a>).
</small>
