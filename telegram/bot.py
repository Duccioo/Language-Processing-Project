import os
import time
import io
import requests

import librosa
from dotenv import load_dotenv

# telegram
import telebot
from telegraph import Telegraph


# AI

from transformers import pipeline
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    WhisperForConditionalGeneration,
)
import torch
from faster_whisper import WhisperModel

# carico variabili d'ambiente
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


def transcribe_audio_speculative(
    audio_data,
    model_name: str = "openai/whisper-small",
    assistant_model_name: str = "openai/whisper-tiny",
    cache_dir: str = "models",
):
    torch_dtype = torch.float32
    speech, sr = librosa.load(io.BytesIO(audio_data), sr=16000)

    start_time = time.time()
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation="sdpa",
        cache_dir=cache_dir,
    )
    processor = AutoProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir,
    )

    assistant_model = WhisperForConditionalGeneration.from_pretrained(
        assistant_model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation="sdpa",
        cache_dir=cache_dir,
    )

    inputs = processor(speech, sampling_rate=16000, return_tensors="pt")
    output = model.generate(
        **inputs,
        assistant_model=assistant_model,
        use_cache=True,
        language="it",
        task="transcribe",
    )

    transcribed_speech = processor.batch_decode(
        output, skip_special_tokens=True, normalize=True
    )[0]

    generation_time = round(time.time() - start_time, 2)

    return transcribed_speech, generation_time


def summary(
    text: str,
    model_summary: str = "efederici/it5-base-summarization",
    cache_dir: str = "./model",
):
    tokenizer = AutoTokenizer.from_pretrained(model_summary, cache_dir=cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_summary, cache_dir=cache_dir)

    inputs = tokenizer(text, return_tensors="pt")

    outputs = model.generate(
        inputs["input_ids"],
        max_length=150,
        # min_length=10,
        length_penalty=1.0,
        num_beams=3,
        # early_stopping=True,
    )

    return tokenizer.decode(
        outputs[0], clean_up_tokenization_spaces=True, skip_special_tokens=True
    )


def add_return(stringa, frequency: int = 3, char: str = " <br> "):
    """
    Concatenates a string with a given character at certain intervals.

    Args:
        stringa (str): The input string to be modified.
        frequency (int): The frequency at which the character should be added (default is 3).
        char (str): The character to be added at the specified frequency (default is "<br>").

    Returns:
        str: The modified string.
    """
    nuova_stringa = ""
    for i in range(len(stringa)):
        nuova_stringa += stringa[i]
        if (i) % frequency == 0 and stringa[i] == ".":
            if i + 2 <= len(stringa):
                if stringa[i + 1] != "." and stringa[i + 2] != ".":
                    nuova_stringa += char
    return nuova_stringa


def transcribe_audio_long(
    audio_data,
    message,
    model_size: str = "large-v3",
    model_summary: str = "efederici/it5-base-summarization",
    cache_dir: str = "models",
):
    """
    Transcribe long audio data using a specified model, and summarize the transcribed text.

    Args:
        audio_data: The audio data to transcribe.
        message: The message associated with the audio data.
        model_size: The size of the transcribing model (default is "large-v3").
        model_summary: The model used for summarization (default is "efederici/it5-base-summarization").
        cache_dir: The directory to cache models (default is "models").

    Returns:
        page_url: The URL of the transcribed and summarized audio page on Telegraph.
        generation_time: The time taken for transcribing and summarizing the audio.
        summary_text: The summarized text of the transcribed audio.
    """
    # Crea un account Telegraph
    # telegraph_account = telegraph.api.create_account(short_name="Prova duccio")
    tph = Telegraph()
    tph.create_account(short_name="transcribot", author_name="transcribot", author_url="")

    msg = bot.send_message(
        message.chat.id,
        "L'audio inviato è molto lungo!\nCaricamento Modello Faster in corso...",
    )

    # Run on CPU with INT8
    start_time = time.time()
    model = WhisperModel(
        model_size,
        compute_type="int8",
        cpu_threads=1,
        download_root=cache_dir,
        num_workers=1,
    )
    segments, _ = model.transcribe(
        io.BytesIO(audio_data), beam_size=5, vad_filter=True, language="it"
    )

    bot.edit_message_text(
        "Modello Caricato!\nTrascrizione in corso...", message.chat.id, msg.message_id
    )

    # save audio segments with start and end time, and transcript by audio segment
    start_segments, end_segments, text_segments = list(), list(), ""
    for i, segment in enumerate(segments):
        if i > 0 and i % 2 == 0:
            bot.edit_message_text(
                "Trascrizione in corso" + ("." * i), msg.chat.id, msg.message_id
            )
        start, end, text = segment.start, segment.end, segment.text
        start_segments.append(start)
        end_segments.append(end)

        text_segments = text_segments + text

    text_segments_wspaces = add_return(text_segments)

    summary_text = summary(
        text_segments, model_summary=model_summary, cache_dir=cache_dir
    )

    bot.delete_message(message.chat.id, msg.message_id)

    generation_time = round(time.time() - start_time, 2)

    # Crea una pagina su Telegraph
    page = tph.create_page(
        title="🎙️ Trascrizione Audio 🎙️",
        author_name="Transcribot",
        author_url="https://t.me/NLP_transcribot",
        html_content=f"""<p>{text_segments_wspaces}</p><br><br><p><a href=\"https://github.com/Duccioo/Language-Processing-Project\"><b>🔗Scopri di più🔗</b></a></p>""",
    )

    # Ottieni il link della pagina
    page_url = "https://telegra.ph/{}".format(page["path"])

    return page_url, generation_time, summary_text


def generate_reply_text(audio_transcription, time_taked, summary_text=""):
    # check summary
    if summary_text == "":
        final_text = f"🎙️ <b>Transcription</b> ({time_taked}s): {audio_transcription}"
    else:
        final_text = f"""📚 <b>Summary</b>: {summary_text}\n\n 🎙️<b>Complete Transcription</b> ({time_taked}s): <a href= \"{ audio_transcription }\">LINK</a>"""

    return final_text


@bot.message_handler(commands=["start", "help"])
def send_welcome(message):
    text = f"Ciao {message.from_user.first_name}, questo è un bot per la trascrizione 🎙️ di Audio in Italiano 🇮🇹!\nPer iniziare una Trascrizione inviami un messaggio vocale😊\n\n{telebot.formatting.mbold('✋MAX 20MB✋')}\n\nPer più informazioni visita la [🔗Repo di Github del Progetto](https://github.com/Duccioo/Language-Processing-Project)!"

    bot.reply_to(message, text, parse_mode="MARKDOWN")


# Gestione del messaggio vocale
@bot.message_handler(content_types=["voice", "audio"])
def handle_voice_message(message):
    # Ottieni i dati audio dal messaggio

    summary_text = ""
    speculative_model_name = os.environ["TELEGRAM_BIG_MODEL"]
    speculative_assistant_model_name = os.environ["TELEGRAM_SMALL_MODEL"]
    faster_model_name = os.environ["TELEGRAM_FASTER_MODEL"]
    summary_model_name = os.environ["TELEGRAM_SUMMARY_MODEL"]
    download_dir = (
        os.environ["TELEGRAM_DOWNLOAD_DIR"]
        if "TELEGRAM_DOWNLOAD_DIR" in os.environ
        else "models"
    )

    if message.content_type == "voice":
        message_info = message.voice
        file_info = bot.get_file(message.voice.file_id)
        duration = message.voice.duration
    else:
        message_info = message.audio
        file_info = bot.get_file(message.audio.file_id)
        duration = message.audio.duration

    audio_data = download_content(
        f"https://api.telegram.org/file/bot{TOKEN}/{file_info.file_path}"
    )

    # Ottieni la trascrizione
    if duration > 30:
        transcription, time_in, summary_text = transcribe_audio_long(
            audio_data,
            message,
            model_size=faster_model_name,
            model_summary=summary_model_name,
            cache_dir=download_dir,
        )
    else:
        transcription, time_in = transcribe_audio_speculative(
            audio_data,
            speculative_model_name,
            speculative_assistant_model_name,
            cache_dir=download_dir,
        )

    reply_text = generate_reply_text(transcription, time_in, summary_text)

    # Invia la trascrizione come risposta
    bot.reply_to(
        message,
        reply_text,
        parse_mode="HTML",
    )


# Avvia il bot
if __name__ == "__main__":
    bot.polling(non_stop=True)
