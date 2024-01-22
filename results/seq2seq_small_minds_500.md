---
license: apache-2.0
base_model: openai/whisper-small
tags:
- generated_from_trainer
datasets:
- PolyAI/minds14
metrics:
- wer
model-index:
- name: seq2seq_prova
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: PolyAI/minds14 it-IT
      type: PolyAI/minds14
      config: null
      split: None
      args: it-IT
    metrics:
    - name: Wer
      type: wer
      value: 0.28565965583173997
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# seq2seq_prova

This model is a fine-tuned version of [openai/whisper-small](https://huggingface.co/openai/whisper-small) on the PolyAI/minds14 it-IT dataset.
It achieves the following results on the evaluation set:
- Loss: 0.6693
- Wer: 0.2857

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
- num_epochs: 25
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | Wer    |
|:-------------:|:-----:|:----:|:---------------:|:------:|
| No log        | 4.12  | 70   | 0.3632          | 0.1545 |
| No log        | 8.24  | 140  | 0.4237          | 0.2268 |
| No log        | 12.35 | 210  | 0.4550          | 0.1728 |
| No log        | 16.47 | 280  | 0.5353          | 0.2076 |
| No log        | 20.59 | 350  | 0.6201          | 0.2195 |
| No log        | 24.71 | 420  | 0.6700          | 0.2459 |


### Framework versions

- Transformers 4.36.2
- Pytorch 2.1.2+cu118
- Datasets 2.16.0
- Tokenizers 0.15.0
