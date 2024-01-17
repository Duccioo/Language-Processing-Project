import telebot
from pydub import AudioSegment
import io
import speech_recognition as sr
import requests

from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

# from datasets import Audio, load_dataset

import librosa
import torch

# Inserisci il tuo token ottenuto da BotFather
TOKEN = "6941788422:AAFV-jYkmsmDZGha_2wn6Afu81EKITi0Jlw"

# Creazione dell'istanza del bot
bot = telebot.TeleBot(TOKEN)


def download_content(url):
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


# Funzione per ottenere la trascrizione del file audio
def transcribe_audio_base(audio_data):
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
    model_name = "dbdmg/wav2vec2-xls-r-300m-italian"
    wav2vec2_processor = Wav2Vec2Processor.from_pretrained(model_name)
    wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(model_name)

    speech, sr = librosa.load(io.BytesIO(audio_data), sr=16000)
    input_values = wav2vec2_processor(speech, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        logits = wav2vec2_model(**input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    transcribed_speech = wav2vec2_processor.batch_decode(pred_ids)
    return transcribed_speech[0]


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
    transcription2 = transcribe_audio_base(audio_data)

    # Invia la trascrizione come risposta
    bot.reply_to(
        message,
        f"Trascrizione con modelli addestrati: {transcription}\n\n--------\nTrascrizione con modello Google: {transcription2}",
    )


# Avvia il bot
if __name__ == "__main__":
    bot.polling(none_stop=True)
