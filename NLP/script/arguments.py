from dataclasses import dataclass, field
import os
from typing import Dict, List, Optional, Union
from dotenv import load_dotenv
import sys
import os
import argparse


def list_field(default=None, metadata=None):
    """
    Create a new field with a default value and metadata.
    :param default: The default value for the field (default is None).
    :param metadata: Any additional metadata for the field (default is None).
    :return: A new field with the specified default value and metadata.
    """
    return field(default_factory=lambda: default, metadata=metadata)


load_dotenv()


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=str(os.environ["MODEL"])
        if "MODEL" in os.environ
        else "facebook/wav2vec2-xls-r-300m",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pretrained tokenizer or tokenizer identifier from huggingface.co/models"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    freeze_feature_encoder: bool = field(
        default=bool(os.environ["FREEZE_FEATURE_ENCODER"])
        if "FREEZE_FEATURE_ENCODER" in os.environ
        else True,
        metadata={"help": "Whether to freeze the feature encoder layers of the model."},
    )
    attention_dropout: float = field(
        default=float(os.environ["ATTENTION_DROPOUT"])
        if "ATTENTION_DROPOUT" in os.environ
        else 0.0,
        metadata={"help": "The dropout ratio for the attention probabilities."},
    )
    activation_dropout: float = field(
        default=float(os.environ["ACTIVATION_DROPOUT"])
        if "ACTIVATION_DROPOUT" in os.environ
        else 0.0,
        metadata={
            "help": "The dropout ratio for activations inside the fully connected layer."
        },
    )
    feat_proj_dropout: float = field(
        default=0.0, metadata={"help": "The dropout ratio for the projected features."}
    )
    hidden_dropout: float = field(
        default=float(os.environ["HIDDEN_DROPOUT"])
        if "HIDDEN_DROPOUT" in os.environ
        else 0.0,
        metadata={
            "help": "The dropout probability for all fully connected layers in the embeddings, encoder, and pooler."
        },
    )
    final_dropout: float = field(
        default=float(os.environ["FINAL_DROPOUT"])
        if "FINAL_DROPOUT" in os.environ
        else 0.0,
        metadata={"help": "The dropout probability for the final projection layer."},
    )
    mask_time_prob: float = field(
        default=0.05,
        metadata={
            "help": (
                "Probability of each feature vector along the time axis to be chosen as the start of the vector "
                "span to be masked. Approximately ``mask_time_prob * sequence_length // mask_time_length`` feature "
                "vectors will be masked along the time axis."
            )
        },
    )
    mask_time_length: int = field(
        default=10,
        metadata={"help": "Length of vector span to mask along the time axis."},
    )
    mask_feature_prob: float = field(
        default=0.0,
        metadata={
            "help": (
                "Probability of each feature vector along the feature axis to be chosen as the start of the vectorspan"
                " to be masked. Approximately ``mask_feature_prob * sequence_length // mask_feature_length`` feature"
                " bins will be masked along the time axis."
            )
        },
    )
    mask_feature_length: int = field(
        default=10,
        metadata={"help": "Length of vector span to mask along the feature axis."},
    )
    layerdrop: float = field(
        default=0.0, metadata={"help": "The LayerDrop probability."}
    )
    ctc_loss_reduction: Optional[str] = field(
        default="mean",
        metadata={
            "help": "The way the ctc loss should be reduced. Should be one of 'mean' or 'sum'."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: str = field(
        default=str(os.environ["DATASET_NAME"])
        if "DATASET_NAME" in os.environ
        else "mozilla-foundation/common_voice_11_0",
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )

    dataset_config_name: str = field(
        default=str(os.environ["DATASET_CONFIG_NAME"])
        if "DATASET_CONFIG_NAME" in os.environ
        else None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_split_name: str = field(
        default=str(os.environ["TRAIN_SPLIT_NAME"])
        if "TRAIN_SPLIT_NAME" in os.environ
        else "train",
        metadata={
            "help": (
                "The name of the training data set split to use (via the datasets library). Defaults to "
                "'train'"
            )
        },
    )
    eval_split_name: str = field(
        default=str(os.environ["EVAL_SPLIT_NAME"])
        if "EVAL_SPLIT_NAME" in os.environ
        else "test",
        metadata={
            "help": "The name of the evaluation data set split to use (via the datasets library). Defaults to 'test'"
        },
    )
    audio_column_name: str = field(
        default=str(os.environ["AUDIO_COLUMN_NAME"])
        if "AUDIO_COLUMN_NAME" in os.environ
        else "audio",
        metadata={
            "help": "The name of the dataset column containing the audio data. Defaults to 'audio'"
        },
    )
    text_column_name: str = field(
        default=str(os.environ["TEXT_COLUMN_NAME"])
        if "TEXT_COLUMN_NAME" in os.environ
        else "text",
        metadata={
            "help": "The name of the dataset column containing the text data. Defaults to 'text'"
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=int(os.environ["PREPROCESSING_NUM_WORKERS"])
        if "PREPROCESSING_NUM_WORKERS" in os.environ
        else None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=int(os.environ["MAX_TRAIN_SAMPLES"])
        if "MAX_TRAIN_SAMPLES" in os.environ
        else None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=int(os.environ["MAX_EVAL_SAMPLES"])
        if "MAX_EVAL_SAMPLES" in os.environ
        else None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of validation examples to this "
                "value if set."
            )
        },
    )
    chars_to_ignore: Optional[List[str]] = list_field(
        default=[",", "?", ".", "!", "-", ";", ":", '"', "“", "%", "‘", "”", "�"],
        metadata={"help": "A list of characters to remove from the transcripts."},
    )
    eval_metrics: List[str] = list_field(
        default=["wer"],
        metadata={
            "help": "A list of metrics the model should be evaluated on. E.g. `'wer cer'`"
        },
    )
    max_duration_in_seconds: float = field(
        default=int(os.environ["MAX_DURATION_IN_SECONDS"])
        if "MAX_DURATION_IN_SECONDS" in os.environ
        else 20.0,
        metadata={
            "help": (
                "Filter audio files that are longer than `max_duration_in_seconds` seconds to"
                " 'max_duration_in_seconds`"
            )
        },
    )
    min_duration_in_seconds: float = field(
        default=int(os.environ["MIN_DURATION_IN_SECONDS"])
        if "MIN_DURATION_IN_SECONDS" in os.environ
        else 0.0,
        metadata={
            "help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"
        },
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to only do data preprocessing and skip training. This is especially useful when data"
                " preprocessing errors out in distributed training due to timeout. In this case, one should run the"
                " preprocessing in a non-distributed setup with `preprocessing_only=True` so that the cached datasets"
                " can consequently be loaded in distributed training"
            )
        },
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )

    trust_remote_code: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    unk_token: str = field(
        default="[UNK]",
        metadata={"help": "The unk token for the tokenizer"},
    )
    pad_token: str = field(
        default="[PAD]",
        metadata={"help": "The padding token for the tokenizer"},
    )
    word_delimiter_token: str = field(
        default="|",
        metadata={"help": "The word delimiter token for the tokenizer"},
    )


def training_arg_env_replace(training_arg):
    training_arg.output_dir = (
        str(os.environ["OUTPUT_DIR"])
        if "OUTPUT_DIR" in os.environ and training_arg.output_dir == ""
        else training_arg.output_dir
    )
    training_arg.num_train_epochs = int(
        (os.environ["NUM_TRAIN_EPOCHS"])
        if "NUM_TRAIN_EPOCHS" in os.environ
        else training_arg.num_train_epochs
    )
    training_arg.per_device_train_batch_size = int(
        (os.environ["PER_DEVICE_TRAIN_BATCH_SIZE"])
        if "PER_DEVICE_TRAIN_BATCH_SIZE" in os.environ
        else training_arg.per_device_train_batch_size
    )
    training_arg.per_device_eval_batch_size = int(
        (os.environ["PER_DEVICE_EVAL_BATCH_SIZE"])
        if "PER_DEVICE_EVAL_BATCH_SIZE" in os.environ
        else training_arg.per_device_eval_batch_size
    )
    training_arg.gradient_accumulation_steps = int(
        (os.environ["GRADIENT_ACCUMULATION_STEPS"])
        if "GRADIENT_ACCUMULATION_STEPS" in os.environ
        else training_arg.gradient_accumulation_steps
    )
    training_arg.learning_rate = (
        float(os.environ["LEARNING_RATE"])
        if "LEARNING_RATE" in os.environ
        else training_arg.learning_rate
    )
    training_arg.output_dir = (
        (os.environ["OUTPUT_DIR"])
        if "OUTPUT_DIR" in os.environ
        else training_arg.output_dir
    )
    training_arg.warmup_steps = (
        int(os.environ["WARMUP_STEPS"])
        if "WARMUP_STEPS" in os.environ
        else training_arg.warmup_steps
    )
    training_arg.length_column_name = (
        str(os.environ["LENGTH_COLUMN_NAME"])
        if "LENGTH_COLUMN_NAME" in os.environ
        else "input_length"
    )
    training_arg.save_steps = (
        int(os.environ["SAVE_STEPS"])
        if "SAVE_STEPS" in os.environ
        else training_arg.save_steps
    )
    training_arg.eval_steps = (
        int(os.environ["EVAL_STEPS"])
        if "EVAL_STEPS" in os.environ
        else training_arg.eval_steps
    )
    training_arg.save_total_limit = (
        int(os.environ["SAVE_TOTAL_LIMIT"])
        if "SAVE_TOTAL_LIMIT" in os.environ
        else training_arg.save_total_limit
    )
    training_arg.gradient_checkpointing = (
        bool(os.environ["GRADIENT_CHECKPOINTING"])
        if "GRADIENT_CHECKPOINTING" in os.environ
        else training_arg.gradient_checkpointing
    )
    training_arg.group_by_length = (
        bool(os.environ["GROUP_BY_LENGTH"])
        if "GROUP_BY_LENGTH" in os.environ
        else training_arg.group_by_length
    )
    training_arg.freeze_feature_encoder = (
        bool(os.environ["FREEZE_FEATURE_ENCODER"])
        if "FREEZE_FEATURE_ENCODER" in os.environ
        else training_arg.freeze_feature_encoder
    )
    training_arg.resume_from_checkpoint = (
        bool(os.environ["RESUME_FROM_CHECKPOINT"])
        if "RESUME_FROM_CHECKPOINT" in os.environ
        else training_arg.resume_from_checkpoint
    )
    training_arg.evaluation_strategy = "steps"

    training_arg.fp16 = (
        bool(os.environ["FP16"]) if "FP16" in os.environ else training_arg.fp16
    )


def env_to_cmd_args(env_vars):
    """
    Convert environment variables to command line arguments.

    Args:
        env_vars (dict): A dictionary containing environment variables.

    Returns:
        list: A list of command line arguments.
    """
    cmd_args = []
    for key, value in env_vars.items():
        cmd_args.append(f"--{key}={value}")
    return cmd_args


def get_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="model2")
    parser.add_argument("--type_model", type=str, default="tts")
    parser.add_argument("--new_option", type=str, default="new_value")
    args, _ = parser.parse_known_args()
    return args


def merge_args(env_args, list_cmd):
    cmd_args_dict = parse_options(list_cmd)
    cmd_args_list = [elem for elem in list_cmd if "-" in elem]
    for key, value in env_args.items():
        if key.lower() not in cmd_args_dict:
            cmd_args_list.append(f"--{key.lower()}={value}")
        elif cmd_args_dict[key.lower()] != value:
            cmd_args_list.append(f"--{key.lower()}={cmd_args_dict[key.lower()]}")
    return cmd_args_list


def parse_options(sys_argv):
    options = {}
    # print("sys_argv", sys_argv)
    for i in range(0, len(sys_argv)):
        if sys_argv[i].startswith("-"):
            if "=" in sys_argv[i]:
                key = sys_argv[i][2:].split("=")[0]
                options[key] = sys_argv[i][2:].split("=")[1]
            else:
                key = sys_argv[i][2:]
                options[key] = True
    return options


if __name__ == "__main__":
    env_vars = {
        "MODEL_NAME": "model1",
        "TYPE_MODEL": "asr",
        "PPPP": "1111",
        "PROVA": False,
    }
    load_dotenv()
    # env_args = env_to_cmd_args(env_vars)
    # cmd_args = get_cmd_args()
    final_args = merge_args(env_vars, sys.argv)
    print(final_args)
