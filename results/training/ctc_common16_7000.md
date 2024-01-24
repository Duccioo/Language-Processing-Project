---
language:
- it
license: apache-2.0
base_model: facebook/wav2vec2-xls-r-300m
tags:
- automatic-speech-recognition
- mozilla-foundation/common_voice_16_0
- generated_from_trainer
metrics:
- wer
model-index:
- name: ctc_common16_7000
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# ctc_common16_7000

This model is a fine-tuned version of [facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m) on the MOZILLA-FOUNDATION/COMMON_VOICE_16_0 - IT dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2198
- Wer: 0.1835

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
- train_batch_size: 16
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 2
- total_train_batch_size: 32
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 500
- training_steps: 2000
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | Wer    |
|:-------------:|:-----:|:----:|:---------------:|:------:|
| No log        | 0.46  | 100  | 6.7360          | 1.0    |
| No log        | 0.91  | 200  | 4.2361          | 1.0    |
| No log        | 1.37  | 300  | 3.0609          | 1.0    |
| No log        | 1.83  | 400  | 2.8673          | 1.0    |
| 5.6517        | 2.28  | 500  | 2.8296          | 1.0    |
| 5.6517        | 2.74  | 600  | 0.8398          | 0.6400 |
| 5.6517        | 3.2   | 700  | 0.4292          | 0.3785 |
| 5.6517        | 3.65  | 800  | 0.3258          | 0.2885 |
| 5.6517        | 4.11  | 900  | 0.2770          | 0.2503 |
| 0.6978        | 4.57  | 1000 | 0.2602          | 0.2353 |
| 0.6978        | 5.02  | 1100 | 0.2413          | 0.2243 |
| 0.6978        | 5.48  | 1200 | 0.2459          | 0.2195 |
| 0.6978        | 5.94  | 1300 | 0.2244          | 0.2090 |
| 0.6978        | 6.39  | 1400 | 0.2266          | 0.1986 |
| 0.1637        | 6.85  | 1500 | 0.2225          | 0.1980 |
| 0.1637        | 7.31  | 1600 | 0.2244          | 0.1969 |
| 0.1637        | 7.76  | 1700 | 0.2162          | 0.1869 |
| 0.1637        | 8.22  | 1800 | 0.2190          | 0.1841 |
| 0.1637        | 8.68  | 1900 | 0.2211          | 0.1844 |
| 0.1011        | 9.13  | 2000 | 0.2198          | 0.1835 |


### Framework versions

- Transformers 4.36.2
- Pytorch 2.1.2+cu118
- Datasets 2.16.0
- Tokenizers 0.15.0
