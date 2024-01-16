# Language Processing Project

Progetto per l'esame di Language Processing.

La repo si divide in 2 parti:

- Grammar Design
- NLP project

## NLP

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mlu3WtDwkJp9hWuxhIaXO6LXBxfOBwfR?usp=sharing)

### Modelli

- [CTC](https://distill.pub/2017/ctc/):

  - Modello **Wav2Vec2** base:

    - [Wav2vec2-base preaddestrato solo in poche lingue (non italiano)](https://huggingface.co/facebook/wav2vec2-base)

    - [Wav2vec2-base-it-voxpopuli-v2 (Modello base finetuning su dataset italiano)](https://huggingface.co/facebook/wav2vec2-base-it-voxpopuli-v2)

  - Modello **Wav2Vec2 XLS-R** (supporta 53 lingue):

    - [Wav2vec2-large-xlsr-53 originale non preaddestrato](https://huggingface.co/facebook/wav2vec2-large-xlsr-53)

    - [modello finetuning in italiano Large](https://huggingface.co/facebook/wav2vec2-large-xlsr-53-italian)

    - Modello XLS-R 300M di parametri addestrato su 128 lingue (anche italiano):

      - [modello base 300M preaddestrato](https://huggingface.co/facebook/wav2vec2-xls-r-300m)
      - [modello finetuning in italiano](https://huggingface.co/dbdmg/wav2vec2-xls-r-300m-italian)

- SEQ2SEQ:

  - Modello **Whisper**:

    - [Github Repo ufficiale](https://github.com/openai/whisper)
    - [Modello base addestrato in più lingue (anche italiano)](https://huggingface.co/openai/whisper-base)

    - [Modello Small (244M di parametri)](https://huggingface.co/openai/whisper-small)
    - [Modello Tiny (39M di parametri)](https://huggingface.co/openai/whisper-tiny)

### Dataset

Sono stati utilizzati 3 dataset differenti:

- [PolyAI/minds14](https://huggingface.co/datasets/PolyAI/minds14)
- Mozilla/common_voice sono state prese 2 versioni:
  - [Versione 11](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0)
  - [Versione 16](https://huggingface.co/datasets/mozilla-foundation/common_voice_16_0)
- [Google/fleurs](https://huggingface.co/datasets/google/fleurs) (forse?)

### Link Utili

- [Github Repo Ufficiale ASR](https://github.com/huggingface/transformers/tree/main/examples/pytorch/speech-recognition)

- [Automatic Speech Recognition Hugging Face ](https://huggingface.co/docs/transformers/tasks/asr)

- [recording audio in google colab](https://gist.github.com/korakot/c21c3476c024ad6d56d5f48b0bca92be)

- [Esempio su come usare file audio per Inference](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english)

- [Finetuning Whisper Model (seq2seq)](https://huggingface.co/blog/fine-tune-whisper)
- [COLAB: Finetuning Whisper Model (seq2seq)](https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/fine_tune_whisper.ipynb#scrollTo=-2zQwMfEOBJq)

- [HuggingSound](https://github.com/jonatasgrosman/huggingsound/tree/main)

- [1 script for XLSR and Whisper](https://github.com/voidful/asr-trainer?tab=readme-ov-file)

- [CallBacks for Transformer Library](https://huggingface.co/docs/transformers/main_classes/callback)

### Risultati
