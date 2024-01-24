---
license: apache-2.0
base_model: openai/whisper-tiny
tags:
- generated_from_trainer
datasets:
- PolyAI/minds14
metrics:
- wer
model-index:
- name: modelwhispertiny-minds
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
      value: 0.0
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# modelwhispertiny-minds

This model is a fine-tuned version of [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) on the PolyAI/minds14 it-IT dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0001
- Wer: 0.0

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
- training_steps: 1000
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch  | Step | Validation Loss | Wer    |
|:-------------:|:------:|:----:|:---------------:|:------:|
| No log        | 10.53  | 100  | 0.0556          | 0.0875 |
| No log        | 21.05  | 200  | 0.0021          | 0.0004 |
| No log        | 31.58  | 300  | 0.0007          | 0.0    |
| No log        | 42.11  | 400  | 0.0007          | 0.0306 |
| 0.1382        | 52.63  | 500  | 0.0398          | 0.0760 |
| 0.1382        | 63.16  | 600  | 0.0100          | 0.0153 |
| 0.1382        | 73.68  | 700  | 0.0009          | 0.0002 |
| 0.1382        | 84.21  | 800  | 0.0002          | 0.0    |
| 0.1382        | 94.74  | 900  | 0.0001          | 0.0    |
| 0.0056        | 105.26 | 1000 | 0.0001          | 0.0    |


### Framework versions

- Transformers 4.37.0.dev0
- Pytorch 2.1.0+cu121
- Datasets 2.16.1
- Tokenizers 0.15.0
