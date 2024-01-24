---
license: apache-2.0
base_model: openai/whisper-tiny
tags:
- generated_from_trainer
datasets:
- mozilla-foundation/common_voice_16_0
metrics:
- wer
model-index:
- name: seq2seq_tiny_common16_7000
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: mozilla-foundation/common_voice_16_0 it
      type: mozilla-foundation/common_voice_16_0
      config: null
      split: None
      args: it
    metrics:
    - name: Wer
      type: wer
      value: 0.37965479749932046
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# seq2seq_tiny_common16_7000

This model is a fine-tuned version of [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) on the mozilla-foundation/common_voice_16_0 it dataset.
It achieves the following results on the evaluation set:
- Loss: 0.8071
- Wer: 0.3797

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
| No log        | 0.46  | 100  | 0.8443          | 0.4795 |
| No log        | 0.91  | 200  | 0.7885          | 0.4706 |
| No log        | 1.37  | 300  | 0.7528          | 0.4312 |
| No log        | 1.83  | 400  | 0.7618          | 0.4628 |
| 0.6433        | 2.28  | 500  | 0.8207          | 0.4834 |
| 0.6433        | 2.74  | 600  | 0.8130          | 0.5324 |
| 0.6433        | 3.2   | 700  | 0.8335          | 0.4645 |
| 0.6433        | 3.65  | 800  | 0.8179          | 0.4729 |
| 0.6433        | 4.11  | 900  | 0.8089          | 0.4551 |
| 0.1913        | 4.57  | 1000 | 0.8188          | 0.4204 |
| 0.1913        | 5.02  | 1100 | 0.8513          | 0.4358 |
| 0.1913        | 5.48  | 1200 | 0.8367          | 0.4351 |
| 0.1913        | 5.94  | 1300 | 0.8196          | 0.4150 |
| 0.1913        | 6.39  | 1400 | 0.8248          | 0.4311 |
| 0.0387        | 6.85  | 1500 | 0.8227          | 0.3993 |
| 0.0387        | 7.31  | 1600 | 0.8140          | 0.4168 |
| 0.0387        | 7.76  | 1700 | 0.8118          | 0.3848 |
| 0.0387        | 8.22  | 1800 | 0.8084          | 0.4041 |
| 0.0387        | 8.68  | 1900 | 0.8075          | 0.3729 |
| 0.0059        | 9.13  | 2000 | 0.8071          | 0.3797 |


### Framework versions

- Transformers 4.36.2
- Pytorch 2.1.2+cu118
- Datasets 2.16.0
- Tokenizers 0.15.0
