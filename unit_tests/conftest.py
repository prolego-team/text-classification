"""
fixtures for unit tests
"""

from typing import List

import pytest
from transformers import AutoTokenizer

from text_classification.dataset_utils import InputMultilabelExample, MultilabelDataset


@pytest.fixture
def num_labels() -> int:
    return 4


@pytest.fixture
def class_labels(num_labels: int) -> List[str]:
    return ["Label " + str(i) for i in range(num_labels)]


@pytest.fixture
def multilabel_examples(class_labels: List[str]) -> List[InputMultilabelExample]:
    return [InputMultilabelExample(str(i), "Text " + str(i), class_labels)
            for i in range(10)]


@pytest.fixture
def multilabel_examples_without_labels() -> List[InputMultilabelExample]:
    return [InputMultilabelExample(str(i), "Text " + str(i), None)
            for i in range(10)]


@pytest.fixture
def multilabel_dataset(
    multilabel_examples: List[InputMultilabelExample],
    class_labels: List[str]) -> MultilabelDataset:
    """
    create a dummy multilabel dataset
    """
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    max_length = 128
    return MultilabelDataset(
        multilabel_examples,
        class_labels,
        tokenizer,
        max_length,
        predict=False
    )
