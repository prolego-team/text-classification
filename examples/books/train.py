"""
Example script to train a classifier to classify sentences from books.

Usage:
    python -m examples.books.train <args> <optional_args>
"""

import os
import re
from typing import List

import click
from sklearn.model_selection import train_test_split

from text_classification import (
    configs,
    dataset_utils,
    model_utils,
    training_utils
)


RANDOM_SEED = 12345

# see huggingface.co documentation for a full list of training arguments
# and what they mean:
# https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments
TRAINING_ARGUMENTS = {
    "do_train": True,
    "evaluation_strategy": "steps",  # to disable evaluation during training, change to "no"
    "logging_steps": 2,
    "num_train_epochs": 1.0,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 16,
    "do_predict": False,
    "block_size": 128,
    "seed": RANDOM_SEED,
    "gradient_accumulation_steps": 1,
    "save_strategy": "no",  # to save intermediate model checkpoints duing training, change to "steps" or "epochs",
    "weight_decay": 0  # make non-zero (e.g., 0.02) to apply regularization in AdamW optimizer
}


def split_txt_file(txt_filepath: str) -> List[str]:
    """
    read a text file and split text into sentences
    """
    split_chars = [".", "!", ";", "?"]
    split_regex = "|".join(map(re.escape, split_chars))
    with open(txt_filepath, "r") as f:
        data = f.read()
    data = data.replace("\n", " ")
    split_data = re.split(split_regex, data)
    return split_data


@click.command()
@click.argument("training_config_filepath", type=click.Path(exists=True))
@click.option("inference_config_filepath", default="inference_config.json",
              help="Path to save the inference config file created after training.")
@click.option("--do_class_weights", "-cw", is_flag=True,
              help="Weight the loss by relative class frequency to account for class imbalance.")
def main(**kwargs):
    """
    Train a transformers model to classify sentences based on originating novel.

    \b
    TRAINING_CONFIG_FILEPATH: Path to training config .json file used for training. See
    test_data/training_config.json for an example. \f

    Optional:
        inference_config_filepath: Path where the inference config associated with the trained
           model should be written. Defaults to inference_config.json.
        do_class_weights: If True, modify the loss function to weight scores by the relative
           class frequency to account for class imbalance.
    """

    # read training config
    training_config = configs.read_config_for_training(kwargs["training_config_filepath"])
    TRAINING_ARGUMENTS["output_dir"] = training_config.model_config.saved_model_dirpath

    # read text and create input examples
    data_dirpath = "examples/books/data"
    book_titles = [title.split(".txt")[0] for title in os.listdir(data_dirpath)
                   if title.endswith(".txt")]
    examples = []
    for book_title in book_titles:
        book_txt_filepath = os.path.join(data_dirpath, book_title + ".txt")
        sentences = split_txt_file(book_txt_filepath)
        book_examples = [dataset_utils.InputMultilabelExample(str(i), sentence, [book_title])
                         for i, sentence in enumerate(sentences)]
        examples += book_examples

    # load tokenizer
    num_labels = len(book_titles)
    _, tokenizer = model_utils.load_pretrained_model_and_tokenizer(
        training_config.model_config,
        num_labels)

    # create train and eval datasets
    train_examples, eval_examples = train_test_split(
        examples, test_size=0.01, random_state=RANDOM_SEED, shuffle=True)
    # train_examples = train_examples[:100]
    # eval_examples = eval_examples[:10]
    train_dataset = dataset_utils.MultilabelDataset(
        train_examples,
        book_titles,
        tokenizer,
        TRAINING_ARGUMENTS["block_size"],
        predict=False)
    eval_dataset = dataset_utils.MultilabelDataset(
        eval_examples,
        book_titles,
        tokenizer,
        TRAINING_ARGUMENTS["block_size"],
        predict=False)

    # train model
    training_utils.train_multilabel_classifier(
        train_dataset,
        eval_dataset,
        training_config.model_config,
        num_labels,
        TRAINING_ARGUMENTS,
        do_eval=True,
        do_class_weights=kwargs["do_class_weights"]
    )

    # create and save inference config
    trained_model_config = configs.ModelConfig(
        training_config.model_config.saved_model_dirpath,
        "main",
        None,
        "multilabel")
    inference_config = configs.InferenceConfig(
        trained_model_config, book_titles, TRAINING_ARGUMENTS["block_size"])
    configs.save_config_for_inference(inference_config, kwargs["inference_config_filepath"])


if __name__ == "__main__":
    main()
