# RDST
Source code for RDST in paper:
**"A Debiased Self-Training Framework with Graph Self-Supervised Pre-training Aided for Semi-Supervised Rumor Detection"**

## Run
You need to split the datasets with ```sort.py``` before running the code.

The code can be run in the following ways:

```shell script
nohup python main.py --gpu 0 
```

## Dependencies
- python == 3.8
- numpy == 1.24.3
- scipy == 1.7.2
- pytorch == 1.10.0
- pytorch-geometric == 2.0.1
- pytorch-scatter == 2.0.9
- pytorch-sparse == 0.6.12
- jieba == 0.42.1
- gensim == 4.3.1
