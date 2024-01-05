from datasets import load_dataset, Audio
from transformers import AutoProcessor
import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Union
import evaluate
from transformers import AutoModelForCTC, TrainingArguments, Trainer
from transformers import AutoProcessor
from transformers import AutoModelForCTC


processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")


def uppercase(example):
    return {"transcription": example["transcription"].upper()}


def prepare_dataset(batch):
    audio = batch["audio"]
    batch = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        text=batch["transcription"],
    )
    batch["input_length"] = len(batch["input_values"][0])
    return batch


@dataclass
class DataCollatorCTCWithPadding:
    processor: AutoProcessor
    padding: Union[bool, str] = "longest"

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [
            {"input_values": feature["input_values"][0]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features, padding=self.padding, return_tensors="pt"
        )

        labels_batch = self.processor.pad(
            labels=label_features, padding=self.padding, return_tensors="pt"
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels

        return batch


def compute_metrics(pred):
    wer = evaluate.load("wer")
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer_1 = wer.compute(predictions=pred_str, references=label_str)

    return {"wer": wer_1}


if __name__ == "__main__":
    # dataset
    minds = load_dataset("PolyAI/minds14", name="it-IT", split="train[:500]")
    # minds = load_dataset("PolyAI/minds14", name="en-US", split="train[:100]")
    minds = minds.train_test_split(test_size=0.2)
    minds = minds.remove_columns(["english_transcription", "intent_class", "lang_id"])
    minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
    minds = minds.map(uppercase)
    encoded_minds = minds.map(
        prepare_dataset, remove_columns=minds.column_names["train"], num_proc=4
    )
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding="longest")

    model = AutoModelForCTC.from_pretrained(
        "facebook/wav2vec2-base",
        ctc_loss_reduction="mean",
        gradient_checkpointing=True,
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    model.freeze_feature_encoder()

    # wer = evaluate.load("wer")
    # model.gradient_checkpointing_enable()

    # training:
    training_args = TrainingArguments(
        output_dir="saved_model",
        evaluation_strategy="steps",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        warmup_steps=50,
        max_steps=50,
        fp16=True,
        group_by_length=True,
        save_steps=50,
        eval_steps=5,
        # gradient_checkpointing=True,
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_minds["train"],
        eval_dataset=encoded_minds["test"],
        # tokenizer=processor,
        tokenizer=processor.feature_extractor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # trainer.train()

    # inferences:
    dataset = load_dataset("PolyAI/minds14", "en-US", split="train")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    print(dataset[0]["audio"]["array"])
    # processor = AutoProcessor.from_pretrained("stevhliu/my_awesome_asr_mind_model")
    # inputs = processor(
    #     dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt"
    # )
    # model = AutoModelForCTC.from_pretrained("stevhliu/my_awesome_asr_mind_model")
    # with torch.no_grad():
    #     logits = model(**inputs).logits

    # predicted_ids = torch.argmax(logits, dim=-1)
    # transcription = processor.batch_decode(predicted_ids)
