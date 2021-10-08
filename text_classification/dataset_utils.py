"""
Utilities for handling data, e.g., reading and parsing examples
and creating datasets compatible with HuggingFace Trainer.
"""

from dataclasses import asdict, dataclass
from typing import Optional, List, Dict, Union
from ast import literal_eval

import numpy as np
import pandas as pd
import torch
from transformers import PreTrainedTokenizer, BatchEncoding

from text_classification.common import dataframe_to_tsv, tsv_to_dataframe


@dataclass
class InputMultilabelExample:
    """
    An example for multilabel classification has a few components:
       guid: unique identifier
       text: text data, e.g., word, sentence, phrase, paragraph
       labels: list of (string) classes associated with text,
          or None if labels don't exist
    """
    guid: str
    text: str
    labels: Optional[List[str]]


@dataclass
class OutputMultilabelExample(InputMultilabelExample):
    """
    An OutputMultilabelExample is returned as a result of inference.
    It differs from an InputMultilabelExample in two ways:
       1) the attribute "labels" is not optional, i.e., it cannot be None
       2) contains an additional attribute, "logits"
             logits: list of classification scores, one for each label
    """
    labels: List[str]
    logits: List[float]


@dataclass
class MultilabelDataset():
    """
    Custom dataset class for multi-label classification.
    """

    def __init__(
            self,
            examples: List[InputMultilabelExample],
            class_labels: List[str],
            tokenizer: PreTrainedTokenizer,
            max_length: int,
            predict: bool = False):
        """
        tokenize text and create one-hot encoded labels
        """
        # tokenization
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = self.tokenize(examples)

        # one-hot encoding
        self.num_labels = len(class_labels)
        if predict:
            # assign all labels to a dummy label (null)
            self.labels = [[0] * self.num_labels for _ in examples]
        else:
            self.label_to_int = {l: i for i, l in enumerate(class_labels)}
            self.labels = [self.to_one_hot(example) for example in examples]

        # store example guids and input text
        self.texts = [example.text for example in examples]
        self.guids = [example.guid for example in examples]

    def to_one_hot(self, example: InputMultilabelExample) -> List[int]:
        """
        convert list of string labels to one-hot encoded label
        """
        one_hot_vector = [0] * self.num_labels
        for label in example.labels:
            one_hot_vector[self.label_to_int[label]] = 1
        return one_hot_vector

    def tokenize(self, examples: List[InputMultilabelExample]) -> BatchEncoding:
        """
        tokenize text in examples
        """
        example_texts = [example.text for example in examples]
        return self.tokenizer(
            example_texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_class_weights(one_hot_labels: List[int]) -> np.array:
    """
    compute negative:positive example ratio, used for weighting
    class imbalance within the loss function during model training
    """
    one_hot_labels = np.array(one_hot_labels)
    positive_count = np.sum(one_hot_labels, axis=0)
    negative_count = np.sum((one_hot_labels == 0).astype(int), axis=0)
    return negative_count / positive_count


def multilabel_examples_to_tsv(
        multilabel_examples: Union[List[InputMultilabelExample], List[OutputMultilabelExample]],
        tsv_filepath: str) -> None:
    """
    Write a list of multilabel examples to tsv.
    """

    # put examples in a dataframe
    examples_dicts = [asdict(example) for example in multilabel_examples]

    examples_df = pd.DataFrame(examples_dicts)

    dataframe_to_tsv(examples_df, tsv_filepath)


def tsv_to_input_multilabel_examples(
        tsv_filepath: str,
        generate_guids: bool = False) -> List[InputMultilabelExample]:
    """
    Read a tsv file and convert to a list of input multilabel examples. If
    generate_guids is True, also create a unique (to this dataset) id
    for each example.

    Note: There are specific formatting assumptions for the tsv file.
    Refer to test_data/multilabel_examples.tsv (or, if the data is unlabeled,
    test_data/multilabel_examples_without_labels.tsv) for an example.
    Specifically:
       - the tsv file should have a column header called "text"
       - if labels exist, the tsv file should also have a header called "labels"
       - if generate_guids is False, the tsv file should also have a header called "guid"
       - rows under the "text" and "guid" headers both contain strings
       - rows under the "labels" header contain lists of strings (lists indicated using square brackets [])
    """

    # read tsv as dataframe
    dataframe = tsv_to_dataframe(tsv_filepath)
    if generate_guids:
        dataframe["guid"] = range(dataframe.shape[0])

    if "labels" not in dataframe.columns:
        dataframe["labels"] = ""

    # generate a list of examples
    # if labels exist in the dataframe, literal_eval converts a string to
    # a list (e.g., "['Label 1', 'Label 2']" to ["Label 1", "Label 2"])
    multilabel_examples = [
        InputMultilabelExample(
            str(row["guid"]),
            str(row["text"]),
            None if row["labels"] == "" else literal_eval(row["labels"]))
        for _, row in dataframe.iterrows()
    ]

    return multilabel_examples


def dictionaries_to_input_multilabel_examples(
        example_dictionaries: List[Dict],
        generate_guids: bool) -> List[InputMultilabelExample]:
    """
    Convert a list of dictionaries (one per example) to a list
    of multi-label examples. If generate_guids is True, also
    create a unique (to this dataset) id for each example.

    Assumptions:
       - each dictionary contains the key "text"
       - if labels exist, each dictionary contains the key "labels"
       - if generate_guids is False, each dictionary also contains
         a key called "guid"
       - "text" and "guid" each contain a string
       - "labels" contains a list of strings or None

    See multilabel_example_dictionaries and
    multilabel_example_dictionaries_without_labels within
    unit_tests/conftest.py for an example.
    """
    if generate_guids:
        return [InputMultilabelExample(str(i),
                                       example_dict["text"],
                                       example_dict.get("labels"))
                for i, example_dict in enumerate(example_dictionaries)]
    else:
        return [InputMultilabelExample(example_dict["guid"],
                                       example_dict["text"],
                                       example_dict.get("labels"))
                for example_dict in example_dictionaries]


def sorted_class_labels(
        multilabel_examples: Union[List[InputMultilabelExample], List[OutputMultilabelExample]]) -> List[str]:
    """
    returns a sorted list of class labels from multilabel examples
    """
    label_list = [example.labels for example in multilabel_examples
                  if example.labels is not None]
    if len(label_list) == 0:
        return []
    else:
        return sorted(set([label for labels in label_list for label in labels]))
