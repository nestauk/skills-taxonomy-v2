# skills-taxonomy-v2

## Introduction

The UK labour market has been especially tumultuous in light of COVID-19 and Brexit. Employers [now face both long-running and circumstantial skills challenges](https://www.edge.co.uk/documents/167/04.05.21_Skills_shortages_bulletin_summary.pdf) due to such disruptive events. As we observe the impact of current events on the skills landscape, an informed labour market is more important than ever. Therefore, in partnership with the [Economic Statistics Centre of Excellence (ESCoE)](https://www.escoe.ac.uk/), we are releasing an updated skills taxonomy that is more open, more up-to-date and methodologically refined.

This repo contains the source code for this project. To read the full technical report, click [here](https://docs.google.com/document/d/1ZHE6Z6evxyUsSiojdNatSa_yMDJ8_UlB1K4YD1AhGG8/edit#heading=h.r6ck9fjaexcl).

An overview of the methodology, coloured by the three main steps to the pipeline, can be visualised below:

![](./outputs/reports/figures/methodology_overview_pipeline.jpg)

### Pipeline steps

More details of the steps included in this project, and running instructions, can be found in their respective READMEs:

1. [sentence_classifier](skills_taxonomy_v2/pipeline/sentence_classifier/README.md) - Training a classifier to predict skill sentences.
2. [skills_extraction](skills_taxonomy_v2/pipeline/skills_extraction/README.md) - Extracting skills from skill sentences.
3. [skills_taxonomy](skills_taxonomy_v2/pipeline/skills_taxonomy/README.md) - Building the skills taxonomy from extracted skills.

### Analysis

This repository also contains various pieces of analysis of the taxonomy. These are discussed in the main analysis [README file](skills_taxonomy_v2/analysis/README.md).

## Setup

- Meet the data science cookiecutter [requirements](http://nestauk.github.io/ds-cookiecutter), in brief:
  - Install: `git-crypt`
  - Have a Nesta AWS account configured with `awscli`
- Run `make install` to configure the development environment:
  - Setup the conda environment
  - Configure pre-commit
  - Configure metaflow to use AWS

## Running locally

When you are running scripts from this repo for the first time you need to create the environment by running `make conda-create` to create the conda environment. Then everytime after this you can activate it using `conda activate skills-taxonomy-v2`. If you update the requirements then run `make conda-update`.

As a one off, if needed, you will also have to run:

```
conda install pytorch torchvision torchaudio -c pytorch
```

to use pytorch, and

```
conda install -c conda-forge spacy==3.0.0
python -m spacy download en_core_web_sm
```

for spaCy.

Then

```
conda install cdlib=0.2.3
```

(this doesn't work when added to the environment.yaml).

to install xgboost:

```
conda install -c anaconda py-xgboost
```

or

```
conda install -c conda-forge py-xgboost
```

if you aren't using anaconda.



## Contributor guidelines

[Technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

---

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
(<a href="http://nestauk.github.io/ds-cookiecutter">Read the docs here</a>).
</small>
