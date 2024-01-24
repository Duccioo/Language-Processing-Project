---
license: apache-2.0
base_model: openai/whisper-base
tags:
- generated_from_trainer
datasets:
- mozilla-foundation/common_voice_11_0
metrics:
- wer
model-index:
- name: seq2seq_base_common11_7000_2
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: mozilla-foundation/common_voice_11_0 it
      type: mozilla-foundation/common_voice_11_0
      config: null
      split: None
      args: it
    metrics:
    - name: Wer
      type: wer
      value: 0.3188494077834179
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# seq2seq_base_common11_7000_2

This model is a fine-tuned version of [openai/whisper-base](https://huggingface.co/openai/whisper-base) on the mozilla-foundation/common_voice_11_0 it dataset.
It achieves the following results on the evaluation set:
- Loss: 0.6976
- Wer: 0.3188

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
| No log        | 0.46  | 100  | 0.6481          | 0.3695 |
| No log        | 0.91  | 200  | 0.6149          | 0.3525 |
| No log        | 1.37  | 300  | 0.6176          | 0.3576 |
| No log        | 1.83  | 400  | 0.6516          | 0.3676 |
| 0.4555        | 2.28  | 500  | 0.7025          | 0.3785 |
| 0.4555        | 2.74  | 600  | 0.7065          | 0.3557 |
| 0.4555        | 3.2   | 700  | 0.7670          | 0.3727 |
| 0.4555        | 3.65  | 800  | 0.7417          | 0.3735 |
| 0.4555        | 4.11  | 900  | 0.7313          | 0.3691 |
| 0.1345        | 4.57  | 1000 | 0.7186          | 0.3627 |
| 0.1345        | 5.02  | 1100 | 0.7295          | 0.3629 |
| 0.1345        | 5.48  | 1200 | 0.7490          | 0.3599 |
| 0.1345        | 5.94  | 1300 | 0.7293          | 0.3395 |
| 0.1345        | 6.39  | 1400 | 0.7267          | 0.3682 |
| 0.0308        | 6.85  | 1500 | 0.7152          | 0.3184 |
| 0.0308        | 7.31  | 1600 | 0.7102          | 0.3170 |
| 0.0308        | 7.76  | 1700 | 0.7017          | 0.3132 |
| 0.0308        | 8.22  | 1800 | 0.6984          | 0.3233 |
| 0.0308        | 8.68  | 1900 | 0.6982          | 0.3174 |
| 0.0046        | 9.13  | 2000 | 0.6976          | 0.3188 |


### Framework versions

- Transformers 4.36.2
- Pytorch 2.1.2+cu118
- Datasets 2.16.0
- Tokenizers 0.15.0
