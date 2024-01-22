---
language:
- it
license: apache-2.0
base_model: facebook/wav2vec2-xls-r-300m
tags:
- automatic-speech-recognition
- mozilla-foundation/common_voice_11_0
- generated_from_trainer
metrics:
- wer
model-index:
- name: ctc_common_11_5000
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# ctc_common_11_5000

This model is a fine-tuned version of [facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m) on the MOZILLA-FOUNDATION/COMMON_VOICE_11_0 - IT dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2142
- Wer: 0.1859

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
- eval_batch_size: 16
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
| No log        | 0.46  | 100  | 7.1144          | 1.0    |
| No log        | 0.91  | 200  | 4.3866          | 1.0    |
| No log        | 1.37  | 300  | 3.1052          | 1.0    |
| No log        | 1.83  | 400  | 2.8774          | 1.0    |
| 2.9801        | 2.28  | 500  | 2.7656          | 0.9974 |
| 2.9801        | 2.74  | 600  | 0.6312          | 0.4981 |
| 2.9801        | 3.2   | 700  | 0.3947          | 0.3376 |
| 2.9801        | 3.65  | 800  | 0.3169          | 0.2747 |
| 2.9801        | 4.11  | 900  | 0.2725          | 0.2439 |
| 0.574         | 4.57  | 1000 | 0.2446          | 0.2265 |
| 0.574         | 5.02  | 1100 | 0.2434          | 0.2233 |
| 0.574         | 5.48  | 1200 | 0.2271          | 0.2148 |
| 0.574         | 5.94  | 1300 | 0.2298          | 0.2048 |
| 0.574         | 6.39  | 1400 | 0.2189          | 0.1973 |
| 0.1485        | 6.85  | 1500 | 0.2173          | 0.1962 |
| 0.1485        | 7.31  | 1600 | 0.2206          | 0.1965 |
| 0.1485        | 7.76  | 1700 | 0.2161          | 0.1946 |
| 0.1485        | 8.22  | 1800 | 0.2200          | 0.1877 |
| 0.1485        | 8.68  | 1900 | 0.2169          | 0.1873 |
| 0.0967        | 9.13  | 2000 | 0.2142          | 0.1859 |


### Framework versions

- Transformers 4.36.2
- Pytorch 2.1.2+cu118
- Datasets 2.16.0
- Tokenizers 0.15.0
