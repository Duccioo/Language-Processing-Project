from datasets import load_dataset, load_metric, Audio
from datasets import ClassLabel
import random
import pandas as pd
from IPython.display import display, HTML

import re
import json

from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor

import numpy as np
import random

import evaluate

from transformers import TrainingArguments


import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import Wav2Vec2ForCTC
from transformers import Trainer

import os


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels

        return batch


def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(
        dataset
    ), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    display(HTML(df.to_html()))


def remove_special_characters(batch):
    chars_to_remove_regex = "[\,\?\.\!\-\;\:\"\“\%\‘\”\�']"

    batch["transcription"] = re.sub(
        chars_to_remove_regex, "", batch["transcription"]
    ).lower()
    return batch


def extract_all_chars(batch):
    all_text = " ".join(batch["transcription"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


def compute_metrics(pred):
    wer_metric = evaluate.load("wer")
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer_1 = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer_1}


if __name__ == "__main__":
    dataset_train = load_dataset("PolyAI/minds14", name="it-IT", split="train[0:500]")
    dataset_test = load_dataset("PolyAI/minds14", name="it-IT", split="train[500:600]")

    dataset_train = dataset_train.remove_columns(
        ["english_transcription", "intent_class", "lang_id"]
    )
    dataset_test = dataset_test.remove_columns(
        ["english_transcription", "intent_class", "lang_id"]
    )

    dataset_train = dataset_train.map(remove_special_characters)
    dataset_test = dataset_test.map(remove_special_characters)
    vocab_train = dataset_train.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=dataset_train.column_names,
    )
    vocab_test = dataset_test.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=dataset_test.column_names,
    )
    vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]

    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open("vocab.json", "w") as vocab_file:
        json.dump(vocab_dict, vocab_file)

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        "./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
    )

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    # dataset
    dataset_train = dataset_train.cast_column("audio", Audio(sampling_rate=16_000))
    dataset_test = dataset_test.cast_column("audio", Audio(sampling_rate=16_000))

    # sample of dataset
    rand_int = random.randint(0, len(dataset_train) - 1)
    print("Target text:", dataset_train[rand_int]["transcription"])
    print("Input array shape:", dataset_train[rand_int]["audio"]["array"].shape)
    print("Sampling rate:", dataset_train[rand_int]["audio"]["sampling_rate"])

    def prepare_dataset(batch):
        audio = batch["audio"]

        # batched output is "un-batched"
        batch["input_values"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_values[0]
        batch["input_length"] = len(batch["input_values"])

        with processor.as_target_processor():
            batch["labels"] = processor(batch["transcription"]).input_ids
        return batch

    dataset_train_processed = dataset_train.map(
        prepare_dataset, remove_columns=dataset_train.column_names
    )
    dataset_test_processed = dataset_test.map(
        prepare_dataset, remove_columns=dataset_test.column_names
    )

    max_input_length_in_sec = 15.0
    dataset_train_processed = dataset_train_processed.filter(
        lambda x: x
        < max_input_length_in_sec * processor.feature_extractor.sampling_rate,
        input_columns=["input_length"],
    )
    # dataset_test_processed = dataset_test_processed.filter(
    #     lambda x: x
    #     < max_input_length_in_sec * processor.feature_extractor.sampling_rate,
    #     input_columns=["input_length"],
    # )

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    model_base_name = "facebook/wav2vec2-xls-r-300m"

    model = Wav2Vec2ForCTC.from_pretrained(
        model_base_name,
        attention_dropout=0.0,
        hidden_dropout=0.05,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )

    model.freeze_feature_encoder()

    checkpoint_dir = "checkpoint_model_XLSR"

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        group_by_length=True,
        per_device_train_batch_size=6,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1,
        evaluation_strategy="steps",
        num_train_epochs=50,
        gradient_checkpointing=True,
        fp16=True,
        save_steps=50,
        eval_steps=120,
        logging_steps=300,
        learning_rate=1e-4,
        warmup_steps=500,
        save_total_limit=2,
    )

    print("-" * 10, "Training", "-" * 10)
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset_train_processed,
        eval_dataset=dataset_test_processed,
        tokenizer=processor.feature_extractor,
    )

    checkpoints = [
        f
        for f in os.listdir(checkpoint_dir)
        if os.path.isdir(os.path.join(checkpoint_dir, f))
        and f.startswith("checkpoint-")
    ]

    if len(checkpoints) > 0:
        print(f"Checkpoint Found, trying to load it!")
        resume_training = True

    else:
        print(f"Nessun checkpoint trovato...")
        resume_training = False

    trainer.train(resume_from_checkpoint=resume_training)
    trainer.save_model("saved_model_final_XLSR")
    processor.save_pretrained("saved_model_final_XLSR")
