"""
train a classifier to classify sentences from books
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

TRAINING_ARGUMENTS = {
    "do_train": True,
    "num_train_epochs": 1.0,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 16,
    "do_predict": False,
    "block_size": 128,
}


def split_txt_file(txt_filepath: str) -> List[str]:
    """
    split text in txt file into sentences
    """
    split_chars = [".", "!", ";", "?"]
    split_regex = " |".join(split_chars)
    with open(txt_filepath, "r") as f:
        data = f.readlines()
        data = data.replace("\n", "")
        split_data = re.split(split_regex, data)
    return split_data


@click.command()
@click.argument("training_config_filepath", type=click.Path(exists=True))
@click.option("--do_class_weights", "-cw", is_flag=True,
              help="Weight the loss by relative class frequency to account for class imbalance.")
def main(**kwargs):

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

    # load model and tokenizer, save tokenizer
    num_labels = len(book_titles)
    model, tokenizer = model_utils.load_pretrained_model_and_tokenizer(
        training_config.model_config,
        num_labels)
    tokenizer.save_pretrained(TRAINING_ARGUMENTS["output_dir"])

    # create train and eval datasets
    train_examples, eval_examples = train_test_split(
        examples, test_size=0.2, random_state=RANDOM_SEED, shuffle=True)
    train_dataset = dataset_utils.MultilabelDataset(train_examples, tokenizer)
    eval_dataset = dataset_utils.MultilabelDataset(eval_examples, tokenizer)

    # train model
    training_utils.train_multilabel_classifier(
        train_dataset,
        eval_dataset,
        model,
        do_eval=True,
        do_class_weights=kwargs["do_class_weights"]
    )

    # create and save inference config


if __name__ == "__main__":
    main()
