from transformers import pipeline
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    # Wav2Vec2FeatureExtractor,
    # Wav2Vec2Model,
    # Wav2Vec2CTCTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
)

# from datasets import Audio, load_dataset

import librosa
import torch


def main():
    model_name_list = [
        "results\model\ctc_500esempi_minds",
        "results\model\ctc_5000esempi_mozilla",
        "results\model\seq2seq_prova",
        "dbdmg/wav2vec2-xls-r-300m-italian",
        "saved_model_final_ASR",
        "openai/whisper-tiny",
        "results\model\seq2seq_prova",
    ]
    model_name = model_name_list[0]
    model_type = 1  # 1->CTC, 2->Seq2Seq
    transcriber_speech = ""

    audio_file_name = "audio.wav"

    # if inference_type == 1:
    #     processor = Wav2Vec2Processor.from_pretrained(model_name)
    #     model = Wav2Vec2ForCTC.from_pretrained(model_name).to("cuda")
    #     transcriber = pipeline(
    #         "automatic-speech-recognition",
    #         model=model_name,
    #         # tokenizer=tokenizer,
    #     )
    #     transcribed_speech = transcriber(audio_file_name)

    # test 2
    if model_type == 1:
        wav2vec2_processor = Wav2Vec2Processor.from_pretrained(model_name)
        wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(model_name)

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

    elif model_type == 2:
        # seq2seq
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="italian", task="transcribe"
        )
        speech, sr = librosa.load(audio_file_name, sr=16000)
        input_features = processor(
            speech, sampling_rate=16000, return_tensors="pt"
        ).input_features

        predicted_ids = model.generate(
            input_features, forced_decoder_ids=forced_decoder_ids
        )
        transcribed_speech = processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )

    elif model_type == 3:
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            chunk_length_s=30,
            # device=device,
        )
        speech, sr = librosa.load(audio_file_name, sr=16000)
        transcribed_speech = pipe(speech, batch_size=8)["text"]

    elif model_type == 4:
        transcriber = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            tokenizer=model_name,
            chunk_length_s=30,
        )
        speech, sr = librosa.load(audio_file_name, sr=16000)
        # transcribed_speech = pipe(speech, batch_size=8)["text"]

        transcribed_speech = transcriber(speech, batch_size=8)

    print(transcribed_speech)


if __name__ == "__main__":
    main()
