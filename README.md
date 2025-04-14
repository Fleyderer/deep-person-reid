# Person Re-Identification

> All commands are executed inside project directory.

## Installation

Clone this repo: 

```
git clone https://github.com/Fleyderer/deep-person-reid.git .
```

Create conda environment and activate it:

```
conda env create -f environment.yml
conda activate torchreid
```

## Datasets

This project supports training with most popular person reid datasets: DukeMTMC-reID, Market1501, CUHK03, MSMT17.

> There is also LUPerson-NL dataset under `ml-projects-cv-cubicmedia-cloud/person_id/datasets/`, which can be used as a good pretrain, but not used yet during last research. This dataset has a size of ~100 GB, so be careful while downloading.


Download training datasets using commands:

```
mkdir reid-data

# DukeMTMC-reID
aws --endpoint-url=https://storage.yandexcloud.net/ s3 cp s3://ml-projects-cv-cubicmedia-cloud/person_id/datasets/dukemtmc-reid.zip reid-data/

# Market1501
aws --endpoint-url=https://storage.yandexcloud.net/ s3 cp s3://ml-projects-cv-cubicmedia-cloud/person_id/datasets/market1501.zip reid-data/

# CUHK03
aws --endpoint-url=https://storage.yandexcloud.net/ s3 cp s3://ml-projects-cv-cubicmedia-cloud/person_id/datasets/cuhk03.zip reid-data/

# MSMT17
aws --endpoint-url=https://storage.yandexcloud.net/ s3 cp s3://ml-projects-cv-cubicmedia-cloud/person_id/datasets/msmt17.zip reid-data/

# Unzip all directories 
unzip 'reid-data/*.zip' -d reid-data/

```

## Train model

To run current train pipeline:

```
python train.py --exp-name baseline --sources msmt17 cuhk03 dukemtmcreid market1501 --transforms 'random_flip' 'random_crop' 'color_jitter' --market1501-500k --pretrained
```

> Specify needed GPUs using `--gpus 0 1` if you need to train using first two available GPUs.