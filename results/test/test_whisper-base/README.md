---
license: apache-2.0
base_model: openai/whisper-base
tags:
- generated_from_trainer
datasets:
- data
model-index:
- name: test_whisper-base
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# test_whisper-base

This model is a fine-tuned version of [openai/whisper-base](https://huggingface.co/openai/whisper-base) on the data dataset.
It achieves the following results on the evaluation set:
- eval_loss: 2.3502
- eval_wer: 0.7810
- eval_runtime: 10.6114
- eval_samples_per_second: 3.77
- eval_steps_per_second: 0.471
- step: 0

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

### Framework versions

- Transformers 4.36.2
- Pytorch 2.1.2+cu118
- Datasets 2.16.0
- Tokenizers 0.15.0
