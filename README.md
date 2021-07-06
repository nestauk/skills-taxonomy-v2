# skills-taxonomy-v2

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
conda install -c conda-forge spacy
python -m spacy download en_core_web_sm
```
for spaCy.

## Contributor guidelines

[Technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

---

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
(<a href="http://nestauk.github.io/ds-cookiecutter">Read the docs here</a>).
</small>
