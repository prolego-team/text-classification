from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import torch
from transformers import PreTrainedTokenizer, BatchEncoding


@dataclass
class InputMultilabelExample:
    guid: str
    text: str
    labels: Optional[List[str]]


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
    one_hot_labels = np.array(one_hot_labels)
    positive_count = np.sum(one_hot_labels, axis=0)
    negative_count = np.sum((one_hot_labels == 0).astype(int), axis=0)
    return negative_count / positive_count
