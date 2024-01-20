import telebot
from pydub import AudioSegment
import io

# import speech_recognition as sr
import requests
from transformers import pipeline

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    WhisperForConditionalGeneration,
)

import torch
import librosa
from dotenv import load_dotenv
import os

import time

from faster_whisper import WhisperModel


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


def transcribe_audio_speculative(audio_data):
    torch_dtype = torch.float32
    # model_id = "openai/whisper-small"
    # model_id = "results\model\seq2seq_base_common11_7000"
    assistant_model_id = "results\model\seq2seq_tiny_common16_7000"
    speech, sr = librosa.load(io.BytesIO(audio_data), sr=16000)

    start_time = time.time()

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(model_id)

    assistant_model = WhisperForConditionalGeneration.from_pretrained(
        assistant_model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation="sdpa",
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

    # pipe = pipeline(
    #     "automatic-speech-recognition",
    #     model=model,
    #     tokenizer=processor.tokenizer,
    #     feature_extractor=processor.feature_extractor,
    #     max_new_tokens=128,
    #     chunk_length_s=30,
    #     batch_size=8,
    #     generate_kwargs={
    #         "assistant_model": assistant_model,
    #         "language": "<|it|>",
    #         "task": "transcribe",
    #     },
    #     torch_dtype=torch_dtype,
    #     # device=device,
    # )

    # transcribed_speech = pipe(inputs)

    generation_time = round(time.time() - start_time, 2)

    return transcribed_speech, generation_time


def transcribe_audio_NLP(audio_data):
    # model_name = "dbdmg/wav2vec2-xls-r-300m-italian"

    model_name = "openai/whisper-tiny"
    # model_type = 3

    speech, sr = librosa.load(io.BytesIO(audio_data), sr=16000)

    # else:
    start_time = time.time()
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        tokenizer=model_name,
        chunk_length_s=30,
        generate_kwargs={"task": "transcribe", "language": "<|it|>"},
    )
    transcribed_speech = transcriber(speech, batch_size=8)["text"]
    generation_time = round(time.time() - start_time, 2)
    # print("**" * 5, transcribed_speech, "**" * 5)

    return transcribed_speech, generation_time


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


def transcribe_audio_long(audio_data):
    # model_size = "large-v2"
    model_size = "large-v3"

    # get device
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Run on CPU with INT8
    start_time = time.time()

    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, _ = model.transcribe(
        io.BytesIO(audio_data), beam_size=5, vad_filter=True, language="it"
    )

    # save audio segments with start and end time, and transcript by audio segment
    start_segments, end_segments, text_segments = list(), list(), ""
    for segment in segments:
        # print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        start, end, text = segment.start, segment.end, segment.text
        start_segments.append(start)
        end_segments.append(end)
        text_segments = text_segments + text

    # summary_text = ""
    summary_text = summary(text_segments)

    generation_time = round(time.time() - start_time, 2)

    return text_segments, generation_time, summary_text


def generate_reply_text(audio_transcription, time_taked, summary_text=""):
    # check summary
    if summary_text == "":
        final_text = f"🎙️ <b>Transcription</b> ({time_taked}s): {audio_transcription}"
    else:
        final_text = f"🎙️ <b>Transcription</b> ({time_taked}s): {audio_transcription}\n\n📚 <b>Summary</b>: {summary_text}"

    return final_text


# Gestione del messaggio vocale
@bot.message_handler(content_types=["voice", "audio"])
def handle_voice_message(message):
    # Ottieni i dati audio dal messaggio
    if message.content_type == "voice":
        file_info = bot.get_file(message.voice.file_id)
        duration = message.voice.duration
    else:
        file_info = bot.get_file(message.audio.file_id)
        duration = message.audio.duration

    audio_data = download_content(
        f"https://api.telegram.org/file/bot{TOKEN}/{file_info.file_path}"
    )

    summary_text = ""

    # Ottieni la trascrizione
    if duration > 30:
        transcription, time_in, summary_text = transcribe_audio_long(audio_data)
    else:
        transcription, time_in = transcribe_audio_speculative(audio_data)

    reply_text = generate_reply_text(transcription, time_in, summary_text)

    # Invia la trascrizione come risposta
    bot.reply_to(message, reply_text, parse_mode="HTML")


# Avvia il bot
if __name__ == "__main__":
    bot.polling(non_stop=True)
