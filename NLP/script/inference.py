from transformers import pipeline
import librosa
import argparse


# Creazione del parser
parser = argparse.ArgumentParser(description="Inference for transcribe audio files")

# Aggiunta delle opzioni
parser.add_argument(
    "--model", type=str, help="path to model, or model name from Hugging Face"
)
parser.add_argument("--audio_file", type=str, help="path to audio file to transcribe")


# Parsing degli argomenti
args = parser.parse_args()


def main():
    """
    Executes the main function of the program.

    This function takes no parameters.

    It loads the model name and audio file name from the command line arguments.

    It initializes a transcriber pipeline for automatic speech recognition, using the specified model and tokenizer.

    The speech signal is loaded from the audio file using the librosa library, with a sample rate of 16000.

    The transcriber is then used to transcribe the speech signal into text, with a batch size of 8.

    The transcribed speech is printed to the console.

    This function does not return any value.
    """
    model_name = args.model
    audio_file_name = args.audio_file

    transcriber = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        tokenizer=model_name,
        chunk_length_s=30,
    )
    speech, sr = librosa.load(audio_file_name, sr=16000)
    transcribed_speech = transcriber(speech, batch_size=8)
    print(transcribed_speech)


if __name__ == "__main__":
    main()
