---
language:
- it-IT
license: apache-2.0
base_model: facebook/wav2vec2-xls-r-300m
tags:
- automatic-speech-recognition
- PolyAI/minds14
- generated_from_trainer
metrics:
- wer
model-index:
- name: prova_5000esempi_minds_2
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# prova_5000esempi_minds_2

This model is a fine-tuned version of [facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m) on the POLYAI/MINDS14 - IT-IT dataset.
It achieves the following results on the evaluation set:
- Loss: 0.5244
- Wer: 0.3319

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 2
- total_train_batch_size: 16
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 500
- num_epochs: 100
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | Wer    |
|:-------------:|:-----:|:----:|:---------------:|:------:|
| No log        | 3.39  | 100  | 6.0565          | 1.0    |
| No log        | 6.78  | 200  | 3.6930          | 1.0    |
| No log        | 10.17 | 300  | 3.0253          | 1.0    |
| No log        | 13.56 | 400  | 2.9434          | 1.0    |
| 5.5648        | 16.95 | 500  | 2.8106          | 1.0    |
| 5.5648        | 20.34 | 600  | 1.1459          | 0.8165 |
| 5.5648        | 23.73 | 700  | 0.6756          | 0.5402 |
| 5.5648        | 27.12 | 800  | 0.5536          | 0.4420 |
| 5.5648        | 30.97 | 900  | 0.5198          | 0.4371 |
| 0.1069        | 34.36 | 1000 | 0.5288          | 0.4387 |
| 0.1069        | 37.75 | 1100 | 0.5463          | 0.4204 |
| 0.1069        | 41.14 | 1200 | 0.5317          | 0.3918 |
| 0.1069        | 44.53 | 1300 | 0.5344          | 0.3923 |
| 0.1069        | 47.92 | 1400 | 0.5049          | 0.3864 |
| 0.0494        | 51.31 | 1500 | 0.5248          | 0.3697 |
| 0.0494        | 54.69 | 1600 | 0.5632          | 0.3880 |
| 0.0494        | 58.08 | 1700 | 0.4871          | 0.3734 |
| 0.0494        | 61.47 | 1800 | 0.5386          | 0.3751 |
| 0.0494        | 64.86 | 1900 | 0.5424          | 0.3740 |
| 0.0273        | 68.25 | 2000 | 0.5260          | 0.3659 |
| 0.0273        | 72.34 | 2100 | 0.5285          | 0.3443 |
| 0.0273        | 75.73 | 2200 | 0.5000          | 0.3416 |
| 0.0273        | 79.12 | 2300 | 0.5135          | 0.3475 |
| 0.0273        | 82.51 | 2400 | 0.5099          | 0.3432 |
| 0.0175        | 85.9  | 2500 | 0.5172          | 0.3378 |
| 0.0175        | 89.29 | 2600 | 0.5332          | 0.3400 |
| 0.0175        | 92.68 | 2700 | 0.5368          | 0.3411 |
| 0.0175        | 96.07 | 2800 | 0.5243          | 0.3314 |
| 0.0175        | 99.46 | 2900 | 0.5244          | 0.3319 |


### Framework versions

- Transformers 4.36.2
- Pytorch 2.1.2+cu118
- Datasets 2.16.0
- Tokenizers 0.15.0
