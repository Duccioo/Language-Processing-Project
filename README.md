# Language Processing Project

Progetto per l'esame di Language Processing

## Modelli

- Modello base addestrato solo in poche lingue

  - [Wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base)

  - [Wav2vec2-base-it-voxpopuli-v2](https://huggingface.co/facebook/wav2vec2-base-it-voxpopuli-v2)

- Modello XLS-R originale non preaddestrato (supporta 53 lingue):

  - [Wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53)

- Modello XLS-R 300m di parametri addestrato su 128 lingue (anche italiano):

  - [modello base preaddestrato](https://huggingface.co/facebook/wav2vec2-xls-r-300m)
  - [modello finetuning in italiano del modello base 300m](https://huggingface.co/dbdmg/wav2vec2-xls-r-300m-italian)
  - [modello finetuning in italiano Large](https://huggingface.co/facebook/wav2vec2-large-xlsr-53-italian)

## Link Utili

- [Automatic Speech Recognition Hugging Face ](https://huggingface.co/docs/transformers/tasks/asr)

- [MIND14 Dataset](https://huggingface.co/datasets/PolyAI/minds14)

- [recording audio in google colab](https://gist.github.com/korakot/c21c3476c024ad6d56d5f48b0bca92be)

- [Github Repo Ufficiale di Wav2Vec2](https://github.com/huggingface/transformers/tree/main/examples/pytorch/speech-recognition)
