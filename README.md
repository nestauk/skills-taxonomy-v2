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
conda install -c conda-forge spacy==3.0.0
python -m spacy download en_core_web_sm
```
for spaCy.

Then
```
conda install cdlib=0.2.3
```
(this doesn't work when added to the environment.yaml).


### Running on EC2

Check out [this](https://kstathou.medium.com/how-to-set-up-a-gpu-instance-for-machine-learning-on-aws-b4fb8ba51a7c) if you need to set up a new instance. To connect to the one for this project there is one called `i-0a193c947acc1e53c`, you need to download the relevant `nesta_core.pem` file from `s3://nesta-production-config/nesta_core.pem`.

Connect to it with:
```
chmod 400 nesta_core.pem
ssh -i "nesta_core.pem" ubuntu@ec2-35-176-103-64.eu-west-2.compute.amazonaws.com
```
(you may need to link to correct `nesta_core.pem` location).

EC2 will have the cuda GPU neccessary to get the spacy transformers word embeddings. To get this to work on an EC2 instance:
```
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install -U spacy[cuda102,transformers]
pip install transformers[sentencepiece]
python -m spacy download en_core_web_trf
```
Note this won't work on a macbook since they don't have a NVIDIA CUDA GPU.

Stop:
```
aws ec2 stop-instances --instance-ids i-0a193c947acc1e53c

```

## Contributor guidelines

[Technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

---

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
(<a href="http://nestauk.github.io/ds-cookiecutter">Read the docs here</a>).
</small>



