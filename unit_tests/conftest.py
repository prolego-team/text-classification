"""
fixtures for unit tests
"""

from typing import List, Dict

import pytest
from transformers import AutoTokenizer

from text_classification.dataset_utils import (
    InputMultilabelExample,
    MultilabelDataset,
    OutputMultilabelExample)


@pytest.fixture
def num_labels() -> int:
    """number of class labels"""
    return 4


@pytest.fixture
def class_labels(num_labels: int) -> List[str]:
    """list of class labels (strings)"""
    return ["Label " + str(i) for i in range(num_labels)]


@pytest.fixture
def output_multilabel_examples(class_labels: List[str]) -> List[OutputMultilabelExample]:
    """list of 10 output multi-label examples, each labeled with all class labels
       with logits 0.9"""
    logits = [0.9] * len(class_labels)
    return [OutputMultilabelExample(str(i), "Text " + str(i), class_labels, logits)
            for i in range(10)]


@pytest.fixture
def input_multilabel_examples(class_labels: List[str]) -> List[InputMultilabelExample]:
    """list of 10 input multi-label examples, each labeled with all class labels"""
    return [InputMultilabelExample(str(i), "Text " + str(i), class_labels)
            for i in range(10)]


@pytest.fixture
def input_multilabel_examples_without_labels() -> List[InputMultilabelExample]:
    """list of 10 input multi-label examples with class_labels set to None"""
    return [InputMultilabelExample(str(i), "Text " + str(i), None)
            for i in range(10)]


@pytest.fixture
def multilabel_dataset(
        input_multilabel_examples: List[InputMultilabelExample],
        class_labels: List[str]) -> MultilabelDataset:
    """
    dummy multilabel dataset
    """
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    max_length = 128
    return MultilabelDataset(
        input_multilabel_examples,
        class_labels,
        tokenizer,
        max_length,
        predict=False
    )


@pytest.fixture
def input_multilabel_example_dictionaries(input_multilabel_examples) -> List[Dict]:
    """
    list of dictionaries containing multilabel examples, e.g.:
       [{"guid": "0",
         "text": "Text 0",
         "labels": ["Label 0", "Label 1"]},
        {"guid": "1",
         "text": "Text 1",
         "labels": ["Label 0"]}]
    """
    return [multilabel_example.__dict__
            for multilabel_example in input_multilabel_examples]


@pytest.fixture
def input_multilabel_example_dictionaries_without_labels(
        input_multilabel_examples_without_labels) -> List[Dict]:
    """
    list of dictionaries containing multilabel examples without labels, e.g.:
       [{"guid": "0",
         "text": "Text 0"},
        {"guid": "1",
         "text": "Text 1"}]
    """
    return [{"guid": multilabel_example.guid,
             "text": multilabel_example.text}
            for multilabel_example in input_multilabel_examples_without_labels]
