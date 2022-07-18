# Repository for NLU project course academic year 2021/2022

In this repository there will be contained everything used for the project:

- source code
- report
- datasets

## Task 

The task to perform both **Slot filling** and **Intent classification** tasks.

In the literature there are proofs that models trained jointly on both tasks perform better than individual task specific models. 

### Slot filling
The sequence labelling task is defined as:
- Given a sequence of tokens $w = {w_1, w_2, ..., w_n}$,
- defining a sequence of labels as $l = {l_1, l_2, ..., l_n}$
- compute the sequence $\hat{l}$ such as $\hat{l} = \underset{l}{\operatorname{argmax}} P(l|w)$ 

\
An example from ATIS dataset is the following: 

| Slot Filling | | | | | | | | | | | | | |
|----- |----- |----- |----- |----- |----- |----- |----- |----- |----- |----- |----- |----- |----- |
| Input sequence: |i |'d |like |to |see |all |the |economy |fares |from |baltimore |to | philadelphia|
| Output sequence: | O |O |O |O |O |O |O |B-economy |O |O |B-fromloc.city_name |O |B-toloc.city_name|



### Intent classification
The intent classification problem is defined as follows:
- Given a sequence of tokens $w = {w_1, w_2, ..., w_n}$,
- And a set of labels $L$ where $l \in L$
- estimate the label $\hat{l}$ such as $\hat{l} = \underset{l}{\operatorname{argmax}} P(l|w)$ 

An example from ATIS dataset is the following:

| Intent Classification | | | | | | | | | | | | | |
|----- |----- |----- |----- |----- |----- |----- |----- |----- |----- |----- |----- |----- |----- |
| Input sequence: |i |'d |like |to |see |all |the |economy |fares |from |baltimore |to | philadelphia|
| Output label: | airfare |


## Dataset
The datasets that are used in the evaluation are ATIS (Airline Travel Information Systems) and SNIPS

### ATIS

Airline Travel Information Systems (ATIS) dataset is made of entries as such

```json
[
    {
        "utterance": "i want to fly from boston at 838 am and arrive in denver at 1110 in the morning",
        "slots": "O O O O O B-fromloc.city_name O B-depart_time.time I-depart_time.time O O O B-toloc.city_name O B-arrive_time.time O O B-arrive_time.period_of_day",
        "intent": "flight"
    },
    ...
]
```

### SNIPS

SNIPS dataset is made of entries as such

```json
[
    {
        "utterance": "listen to westbam alumb allergic on google music",
        "slots": "O O B-artist O B-album O B-service I-service",
        "intent": "PlayMusic"
    },
    ...
 ]
```

## Models

### Baseline LSTM

The first model to be evaluated is the model from LAB 10 of the course. It consists in a simple LSTM architecture, with two linear fully connected layers, one for each task.

### Improved simple model

Some easy improvements to the models are:
- Apply dropout
- Increase the size of embeddings
- Increase size of hidden layers
- Use bidirectional LSTM or GRU
- Usi multi layered LSTM or GRU
- Apply random weightings for joint loss instead of simple sum

These improvements do not change drastically the architecture, but allow to get a few points more. Further tuning of the hyperparameters could increase the score. 

### BERT

Bidirectional Encoder Representations from Transformers (BERT) allow to encode sentences better than Recurrent models and at the same time being faster. 

- Use pre-trained BERT to extract embeddings and train a simple MLP
- Use pre-trained BERT and fine tune for the tasks

both use:
- dropout
- random weightings

### AlBERT

AlBERT is a model based on BERT, but faster and better optimized for downstream tasks

- Use pre-trained AlBERT and fine tune for the tasks

- dropout
- random weightings
