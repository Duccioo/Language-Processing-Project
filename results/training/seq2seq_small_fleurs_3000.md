---
license: apache-2.0
base_model: openai/whisper-small
tags:
- generated_from_trainer
datasets:
- google/fleurs
metrics:
- wer
model-index:
- name: seq2seq_small_fleurs_3000
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: google/fleurs it_it
      type: google/fleurs
      config: null
      split: None
      args: it_it
    metrics:
    - name: Wer
      type: wer
      value: 0.13941756477872047
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# seq2seq_small_fleurs_3000

This model is a fine-tuned version of [openai/whisper-small](https://huggingface.co/openai/whisper-small) on the google/fleurs it_it dataset.
It achieves the following results on the evaluation set:
- Loss: 0.3992
- Wer: 0.1394

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
| No log        | 1.05  | 100  | 0.1897          | 0.1126 |
| No log        | 2.11  | 200  | 0.2062          | 0.1126 |
| No log        | 3.16  | 300  | 0.2484          | 0.1187 |
| No log        | 4.21  | 400  | 0.3226          | 0.1492 |
| 0.1308        | 5.26  | 500  | 0.3826          | 0.1829 |
| 0.1308        | 6.32  | 600  | 0.3591          | 0.1552 |
| 0.1308        | 7.37  | 700  | 0.3832          | 0.1677 |
| 0.1308        | 8.42  | 800  | 0.4141          | 0.1677 |
| 0.1308        | 9.47  | 900  | 0.4251          | 0.1701 |
| 0.0415        | 10.53 | 1000 | 0.4348          | 0.1759 |
| 0.0415        | 11.58 | 1100 | 0.4205          | 0.1601 |
| 0.0415        | 12.63 | 1200 | 0.4042          | 0.1531 |
| 0.0415        | 13.68 | 1300 | 0.4225          | 0.1528 |
| 0.0415        | 14.74 | 1400 | 0.4080          | 0.1481 |
| 0.008         | 15.79 | 1500 | 0.4048          | 0.1433 |
| 0.008         | 16.84 | 1600 | 0.4012          | 0.1411 |
| 0.008         | 17.89 | 1700 | 0.3998          | 0.1403 |
| 0.008         | 18.95 | 1800 | 0.3993          | 0.1391 |
| 0.008         | 20.0  | 1900 | 0.3993          | 0.1396 |
| 0.0003        | 21.05 | 2000 | 0.3992          | 0.1394 |


### Framework versions

- Transformers 4.36.2
- Pytorch 2.1.2+cu118
- Datasets 2.16.0
- Tokenizers 0.15.0
