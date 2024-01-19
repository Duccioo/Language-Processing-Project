<img src="image_bot.jpeg" style="border-radius: 50%;" alt="Transcribot Image" width=200 height=200 align="right" >

# Language Processing Project

Project for the Language Processing exam.

The repository is divided into 2 parts:

- Grammar Design (`grammar_design` folder)
- NLP project (`NLP` folder)

<details>
<summary>Grammar Design</summary>
The original text of the Assignment:

> Using lark implement a parser for the definition of functions, with the following
> rules

> - the functions are defined as:
>   function name(par1,par2,…) {
>   return par1 op par2 op par3…;
>   }

> where name is the function name with the usual restrictions (an alphanumeric string beginning with a letter), par1.. are the function parameters whose names follow the same rules as variables names, op is + or \* (sum or product). The function body contains
> only the return instruction that involves the parameters.

> - assume that only one function can be defined
> - after the function definition, there are the calls whose syntax is: "name(cost1,cost2,…);" where name is the name of a defined function, cost1,… are numeric constants in the same number as the function arguments.

> - print the result of each function call

</details>

## NLP Project

This project aims to develop a system for transcribing audio into text using the powerful transformers library. The primary models utilized in this project are **wav2vec2 xlsr** and **Whisper**, both of which are state-of-the-art models for speech recognition tasks.

The wav2vec2 xlsr model is particularly well-suited for multilingual speech recognition tasks, as it has been trained on a diverse range of languages. On the other hand, the Whisper model is specifically designed for low-resource languages, making it an excellent choice for improving transcription accuracy in challenging scenarios.

The code is runnable on Colab on this link:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mlu3WtDwkJp9hWuxhIaXO6LXBxfOBwfR?usp=sharing)

### Models

- [CTC](https://distill.pub/2017/ctc/):

  - Model **Wav2Vec2** base:

    - [Wav2vec2-base preaddestrato solo in poche lingue (non italiano)](https://huggingface.co/facebook/wav2vec2-base)

    - [Wav2vec2-base-it-voxpopuli-v2 (Modello base finetuning su dataset italiano)](https://huggingface.co/facebook/wav2vec2-base-it-voxpopuli-v2)

  - Model **Wav2Vec2 XLS-R** (53 language):

    - [Wav2vec2-large-xlsr-53 originale non preaddestrato](https://huggingface.co/facebook/wav2vec2-large-xlsr-53)

    - [modello finetuning in italiano Large](https://huggingface.co/facebook/wav2vec2-large-xlsr-53-italian)

    - Modello XLS-R 300M di parametri addestrato su 128 lingue (anche italiano):

      - [modello base 300M preaddestrato](https://huggingface.co/facebook/wav2vec2-xls-r-300m)
      - [modello finetuning in italiano](https://huggingface.co/dbdmg/wav2vec2-xls-r-300m-italian)

- SEQ2SEQ:

  - Model **Whisper**:

    - [Github Repo ufficiale](https://github.com/openai/whisper)
    - [Modello base addestrato in più lingue (anche italiano)](https://huggingface.co/openai/whisper-base)

    - [Modello Small (244M di parametri)](https://huggingface.co/openai/whisper-small)
    - [Modello Tiny (39M di parametri)](https://huggingface.co/openai/whisper-tiny)
    - [EdoAbati/whisper-large-v2-it](https://huggingface.co/EdoAbati/whisper-large-v2-it)
    - [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)
    - [fsicoli/whisper-large-v3-it-cv16](https://huggingface.co/fsicoli/whisper-large-v3-it-cv16)

### Dataset

To train and evaluate these models, we will be leveraging the common voice 11, common voice 16, and fleurs datasets. These datasets consist of a vast collection of multilingual audio recordings and their corresponding transcriptions, enabling us to build a robust and accurate transcription system.

Dataets used:

- [PolyAI/minds14](https://huggingface.co/datasets/PolyAI/minds14)
- Mozilla/common_voice sono state prese 2 versioni:
  - [Versione 11](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0)
  - [Versione 16](https://huggingface.co/datasets/mozilla-foundation/common_voice_16_0)
- [Google/fleurs](https://huggingface.co/datasets/google/fleurs)

### Risultati

### Training results:

|           Model            | Training Loss | Epoch | Step | Validation Loss |  Wer   |
| :------------------------: | :-----------: | :---: | :--: | :-------------: | :----: |
|       ctc 500 minds        |    0.0175     | 99.46 | 2900 |     0.5244      | 0.3319 |
|  seq2seq small 500 minds   |    No log     | 24.71 | 420  |     0.6700      | 0.2459 |
| seq2seq base 7000 common11 |    0.0046     | 9.13  | 2000 |     0.6976      | 0.3188 |
|     ctc 7000 common11      |    0.0967     | 9.13  | 2000 |     0.2142      | 0.1859 |

### Link Utili

- [Github Repo Ufficiale ASR](https://github.com/huggingface/transformers/tree/main/examples/pytorch/speech-recognition)

- [Automatic Speech Recognition Hugging Face ](https://huggingface.co/docs/transformers/tasks/asr)

- [recording audio in google colab](https://gist.github.com/korakot/c21c3476c024ad6d56d5f48b0bca92be)

- [Esempio su come usare file audio per Inference](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english)

- [Finetuning Wav2Vec2 XLSR](https://huggingface.co/blog/fine-tune-xlsr-wav2vec2)

- [Finetuning Whisper Model (seq2seq)](https://huggingface.co/blog/fine-tune-whisper)
- [COLAB: Finetuning Whisper Model (seq2seq)](https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/fine_tune_whisper.ipynb#scrollTo=-2zQwMfEOBJq)

- [1 script for XLSR and Whisper](https://github.com/voidful/asr-trainer?tab=readme-ov-file)
- FineTune on Costum Datasets:

  - [Make your Own Audio Dataset](https://huggingface.co/docs/datasets/audio_dataset)
  - [HuggingSound](https://github.com/jonatasgrosman/huggingsound/tree/main)

- State of Art and Optimizations!:
  - [Distil Whisper](https://github.com/huggingface/distil-whisper/tree/main)
  - [Streaming Datasets from Internet](https://huggingface.co/blog/audio-datasets#streaming-mode-the-silver-bullet)
  - [Speculative Decoding (inference 2x faster)](https://huggingface.co/blog/whisper-speculative-decoding)
  - [Faster Whisper and Large audio Files](https://github.com/piegu/language-models/blob/master/Speech_to_Text_with_faster_whisper_on_large_audio_file_in_any_language.ipynb?source=post_page-----e4d4d2daf0cd--------------------------------)
