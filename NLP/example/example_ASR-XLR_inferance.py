from transformers import pipeline
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
    Wav2Vec2CTCTokenizer,
)
import librosa
import torch

if __name__ == "__main__":
    model_name_1 = "prova_5000esempi_minds_2"
    model_name_2 = "dbdmg/wav2vec2-xls-r-300m-italian"
    model_name_3 = "saved_model_final_ASR"
    # model_name = "dbdmg/wav2vec2-xls-r-300m-italian"
    processor_name = "dbdmg/wav2vec2-xls-r-300m-italian"
    inference_type = 2
    transcriber_speech = ""

    audio_file_name = "audio8.opus"

    if inference_type == 1:
        processor = Wav2Vec2Processor.from_pretrained(model_name_1)
        model = Wav2Vec2ForCTC.from_pretrained(model_name_1).to("cuda")

        # tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        #     "facebook/wav2vec2-xls-r-300m",
        #     unk_token="[UNK]",
        #     pad_token="[PAD]",
        #     word_delimiter_token="|",
        # )
        transcriber = pipeline(
            "automatic-speech-recognition",
            model=model_name_1,
            # tokenizer=tokenizer,
        )
        transcribed_speech = transcriber(audio_file_name)

    # test 2
    elif inference_type == 2:
        wav2vec2_processor = Wav2Vec2Processor.from_pretrained(model_name_1)
        wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(model_name_1)

        file_name = audio_file_name
        speech, sr = librosa.load(file_name, sr=16000)
        input_values = wav2vec2_processor(
            speech, sampling_rate=16000, return_tensors="pt"
        )

        with torch.no_grad():
            logits = wav2vec2_model(**input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        # print(wav2vec2_processor.batch_decode(pred_ids))
        # print(logits.shape)
        transcribed_speech = wav2vec2_processor.batch_decode(pred_ids)

    print(transcribed_speech)
