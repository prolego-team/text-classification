"""
unit tests for dataset_utils.py
"""

from typing import List

import pytest
from transformers import AutoTokenizer, BatchEncoding

from text_classification import dataset_utils


@pytest.mark.parametrize("predict", [True, False])
@pytest.mark.usefixtures("multilabel_examples")
@pytest.mark.usefixtures("class_labels")
def test_MultilabelDataset(
        multilabel_examples: List[dataset_utils.InputMultilabelExample],
        class_labels: List[str],
        predict: bool) -> None:
    """
    test attributes of MultilabelDataset class
    """
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    max_length = 128
    dataset = dataset_utils.MultilabelDataset(
        multilabel_examples,
        class_labels,
        tokenizer,
        max_length,
        predict
    )

    # dataset.num_labels
    assert type(dataset.num_labels) == int
    assert dataset.num_labels == len(class_labels)
    n_examples = len(multilabel_examples)
    # dataset.texts
    assert type(dataset.texts) == list
    assert len(dataset.texts) == n_examples
    # dataset.labels (one-hot encoded labels)
    assert type(dataset.labels) == list
    assert len(dataset.labels) == n_examples
    for label in dataset.labels:
        assert len(label) == len(class_labels)
        for one_hot in label:
            assert one_hot in [0, 1]
    # dataset.encodings (tokenized text)
    assert type(dataset.encodings) == BatchEncoding


def test_compute_class_weights() -> None:
    """
    test that compute_class_weights returns expected results
    using some dummy data
    """
    one_hot_labels = [[0, 0, 0, 1],
                      [0, 1, 1, 0],
                      [1, 1, 0, 0]]
    pos_counts = [1, 2, 1, 1]
    neg_counts = [2, 1, 2, 2]
    expected_class_weights = [neg / pos for pos, neg in zip(pos_counts, neg_counts)]

    actual_class_weights = dataset_utils.compute_class_weights(one_hot_labels)
    for expected, actual in zip(expected_class_weights, actual_class_weights):
        assert actual == pytest.approx(expected)
