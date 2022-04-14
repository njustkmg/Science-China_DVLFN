# DVSFN - News Detection

This is the implement of the paper **Deep Visual-Semantic Fusion Network Considering Cross-Modal Inconsistency for Rumor Detection**.

## Requirements

```
torch==1.7.0
scikit-learn=0.23.2
transformers==4.4.2
geomloss==0.2.3
numpy==1.19.4
pandas==1.1.4
```

## Experiments

To start training a DVSFN model, run the following command.

 `python run.py`

## Datasets

We use two multi-modal rumor datasets, [Fakeddit](https://github.com/entitize/Fakeddit) and [NerualNews](https://cs-people.bu.edu/rxtan/projects/didan/).

Put the dataset in the data folder. `train.csv`, `valid.csv`, `test.csv` are organized as:

```
id	text		        image	label
a	this is rumor		a.png	1
b	this is non-rumor	b.png	0
```

### Image Feature

For each image,  we extract global and region features:

1. we extract 36 region features using a Faster-RCNN model from [here](https://github.com/peteanderson80/bottom-up-attention).
2. we extract global features using ResNet-101.

## Required Arguments

1. data_dir: Directory for text and label
2. region_dir: Directory for extracted region image features
3. global_dir: Directory for extracted global features
4. statistic_dir: Directory for extracted statistical features
