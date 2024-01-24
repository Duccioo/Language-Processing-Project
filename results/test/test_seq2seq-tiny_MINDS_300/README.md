---
base_model: results\model\seq2seq_tiny_MINDS
tags:
- generated_from_trainer
datasets:
- data
model-index:
- name: test_seq2seq-tiny_MINDS_300
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# test_seq2seq-tiny_MINDS_300

This model is a fine-tuned version of [results\model\seq2seq_tiny_MINDS](https://huggingface.co/results\model\seq2seq_tiny_MINDS) on the data dataset.
It achieves the following results on the evaluation set:
- eval_loss: 4.4882
- eval_wer: 0.9311
- eval_runtime: 6.2669
- eval_samples_per_second: 6.383
- eval_steps_per_second: 0.798
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
