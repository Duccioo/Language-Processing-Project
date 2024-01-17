import telebot
from pydub import AudioSegment
import io
import speech_recognition as sr
import requests
from transformers import pipeline

from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


# from transformers import
# from transformers import AutoTokenizer

# from datasets import Audio, load_dataset

import librosa
import torch
from dotenv import load_dotenv
import os

load_dotenv()


# Inserisci il tuo token ottenuto da BotFather
TOKEN = os.environ["TELEGRAM_TOKEN"]

# Creazione dell'istanza del bot
bot = telebot.TeleBot(TOKEN)


def download_content(url):
    """
    Download the content from a given URL.

    Parameters:
    - url (str): The URL to download the content from.

    Returns:
    - content (bytes): The content downloaded from the URL.

    - None: If the download encountered an error.
    """
    try:
        # Effettua la richiesta GET al link
        response = requests.get(url)

        # Verifica se la richiesta ha avuto successo
        if response.status_code == 200:
            # Ottieni il contenuto dalla risposta
            content = response.content
            return content
        else:
            print(f"Errore nella richiesta. Codice di stato: {response.status_code}")
            return None
    except Exception as e:
        print(f"Errore durante il download: {e}")
        return None


def transcribe_audio_base(audio_data):
    # Funzione per ottenere la trascrizione del file audio

    recognizer = sr.Recognizer()

    # Converte i dati audio in formato WAV
    audio = AudioSegment.from_file(io.BytesIO(audio_data), format="ogg")

    # Converti l'audio in formato WAV
    wav_data = audio.export(format="wav").read()

    # Leggi la trascrizione utilizzando il riconoscimento vocale di Google
    with sr.AudioFile(io.BytesIO(wav_data)) as source:
        audio_data = recognizer.record(source)

    try:
        transcription = recognizer.recognize_google(audio_data, language="it-IT")
        return transcription
    except sr.UnknownValueError:
        return "Impossibile riconoscere la trascrizione"
    except sr.RequestError as e:
        return f"Errore nella richiesta al servizio di riconoscimento vocale: {e}"
    # return "AAAAAA"


def transcribe_audio_NLP(audio_data):
    # model_name = "dbdmg/wav2vec2-xls-r-300m-italian"

    model_name = "results/model/ctc_common_11_7000"
    # model_type = 3

    speech, sr = librosa.load(io.BytesIO(audio_data), sr=16000)

    # if model_type == 1:
    #     # ctc
    #     wav2vec2_processor = Wav2Vec2Processor.from_pretrained(model_name)
    #     wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(model_name)

    #     input_values = wav2vec2_processor(
    #         speech, sampling_rate=16000, return_tensors="pt"
    #     )

    #     with torch.no_grad():
    #         logits = wav2vec2_model(**input_values).logits
    #     pred_ids = torch.argmax(logits, dim=-1)
    #     transcribed_speech = wav2vec2_processor.batch_decode(pred_ids)[0]

    # elif model_type == 2:
    #     # seq2seq
    #     processor = WhisperProcessor.from_pretrained(model_name)
    #     model = WhisperForConditionalGeneration.from_pretrained(model_name)

    #     forced_decoder_ids = processor.get_decoder_prompt_ids(
    #         language="italian", task="transcribe"
    #     )
    #     input_features = processor(
    #         speech, sampling_rate=16000, return_tensors="pt"
    #     ).input_features

    #     predicted_ids = model.generate(
    #         input_features, forced_decoder_ids=forced_decoder_ids
    #     )
    #     transcribed_speech = processor.batch_decode(
    #         predicted_ids, skip_special_tokens=True
    #     )[0]

    # else:

    transcriber = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        tokenizer=model_name,
        chunk_length_s=30,
        generate_kwargs={"task": "transcribe", "language": "<|it|>"},
    )
    transcribed_speech = transcriber(speech, batch_size=8)["text"]

    # print("**" * 5, transcribed_speech, "**" * 5)

    return transcribed_speech


def summary(text):
    """
    Generates a summary of the given text using the T5 model for Italian summarization.

    Parameters:
        text (str): The text to be summarized.

    Returns:
        str: The generated summary of the text.
    """
    T5_model_ita_base = "efederici/it5-base-summarization"

    tokenizer = AutoTokenizer.from_pretrained(T5_model_ita_base)

    inputs = tokenizer(text, return_tensors="pt")

    model = AutoModelForSeq2SeqLM.from_pretrained(T5_model_ita_base)

    outputs = model.generate(
        inputs["input_ids"],
        max_length=50,
        # min_length=10,
        length_penalty=1.80,
        num_beams=2,
        # early_stopping=True,
    )

    return tokenizer.decode(
        outputs[0], clean_up_tokenization_spaces=True, skip_special_tokens=True
    )


# Gestione del messaggio vocale
@bot.message_handler(content_types=["voice", "audio"])
def handle_voice_message(message):
    # Ottieni i dati audio dal messaggio
    if message.content_type == "voice":
        file_info = bot.get_file(message.voice.file_id)
    else:
        file_info = bot.get_file(message.audio.file_id)

    audio_data = download_content(
        f"https://api.telegram.org/file/bot{TOKEN}/{file_info.file_path}"
    )

    # Ottieni la trascrizione
    transcription = transcribe_audio_NLP(audio_data)
    # transcription2 = transcribe_audio_base(audio_data)

    summary_text = summary(transcription)

    # Invia la trascrizione come risposta
    bot.reply_to(
        message,
        f"🎙️ <b>Transcription</b>: {transcription}\n\n📚 <b>Summary</b>: {summary_text}",
        parse_mode="HTML",
    )


# Avvia il bot
if __name__ == "__main__":
    bot.polling(non_stop=True)
