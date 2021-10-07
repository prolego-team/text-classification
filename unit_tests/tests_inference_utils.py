"""
unit tests for text_classification/inference_utils.py
"""

from typing import List

import pytest

from text_classification import inference_utils, configs
from text_classification.dataset_utils import InputMultilabelExample


def test_one_hot_to_index_labels() -> None:
    """
    test that expected outputs are created by one_hot_to_index_labels
    """
    one_hot_predictions = [
        [1, 0, 0, 0],
        [0, 1, 0, 1],
        [0, 0, 0, 0],  # null
        [1, 1, 1, 1],
        [0, 0, 0, 1]
    ]
    expected_index_predictions = [
        [0],
        [1, 3],
        [-1],  # null
        [0, 1, 2, 3],
        [3]
    ]
    actual_index_predictions = inference_utils.one_hot_to_index_labels(one_hot_predictions)
    print(actual_index_predictions)

    for actual, expected in zip(actual_index_predictions, expected_index_predictions):
        assert len(actual) == len(expected)
        assert set(actual) == set(expected)


@pytest.mark.parametrize("set_custom_thresholds", [True, False])
@pytest.mark.usefixtures("multilabel_examples_without_labels")
@pytest.mark.usefixtures("class_labels")
def test_predict_multilabel_classes(
        set_custom_thresholds: bool,
        multilabel_examples_without_labels: List[InputMultilabelExample],
        class_labels: List[str]) -> None:
    """
    test that prediction yields a list of labeled examples
    """

    inference_config = configs.read_config_for_inference("test_data/inference_config.json")
    if set_custom_thresholds:
        thresholds = [0.5] * inference_config.num_labels
    else:
        thresholds = None
    examples_with_labels = inference_utils.predict_multilabel_classes(
        inference_config.model_config,
        inference_config.class_labels,
        inference_config.max_length,
        multilabel_examples_without_labels,
        thresholds
    )

    assert type(examples_with_labels) == list
    assert len(examples_with_labels) == len(multilabel_examples_without_labels)
    for example in examples_with_labels:
        assert example.labels is not None
        assert [lab in class_labels for lab in example.labels]
        assert len(example.labels) == len(example.logits)
