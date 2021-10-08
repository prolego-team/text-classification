"""
unit tests for text_classification/inference_utils.py
"""

from typing import List

import pytest
import numpy as np

from text_classification import inference_utils, configs
from text_classification.dataset_utils import (
    InputMultilabelExample,
    MultilabelDataset,
    OutputMultilabelExample)


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
        [],  # null
        [0, 1, 2, 3],
        [3]
    ]
    actual_index_predictions = inference_utils.one_hot_to_index_labels(one_hot_predictions)

    for actual, expected in zip(actual_index_predictions, expected_index_predictions):
        assert len(actual) == len(expected)
        assert set(actual) == set(expected)


@pytest.mark.parametrize("threshold_is_list", [True, False])
@pytest.mark.usefixtures("class_labels")
@pytest.mark.usefixtures("input_multilabel_examples_without_labels")
def test_MultilabelPredictor(
        threshold_is_list: bool,
        class_labels: List[str],
        input_multilabel_examples_without_labels: List[InputMultilabelExample]) -> None:

    # create MultilabelPredictor instance
    inference_config = configs.read_config_for_inference("test_data/inference_config.json")
    predictor = inference_utils.MultilabelPredictor(inference_config.model_config, class_labels)

    # test logits_to_predicted_labels
    threshold = 0.5
    if threshold_is_list:
        threshold = [threshold] * len(class_labels)
    logits = np.array([[0.9, 0.9, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0],  # null
                       [0.9, 0.9, 0.9, 0.0],
                       [0.9, 0.0, 0.0, 0.0]])
    expected_out_labels = [["Label 0", "Label 1"],
                           [],  # null
                           ["Label 0", "Label 1", "Label 2"],
                           ["Label 0"]]
    expected_out_logits = [[0.9, 0.9],
                           [],  # null
                           [0.9, 0.9, 0.9],
                           [0.9]]
    labels, logits = predictor.logits_to_predicted_labels(logits, threshold)

    assert np.all(labels == expected_out_labels)
    assert np.all(logits == expected_out_logits)

    # test create_dataset
    test_dataset = predictor.create_dataset(
        input_multilabel_examples_without_labels, inference_config.max_length)
    assert type(test_dataset) == MultilabelDataset

    # test predict_proba
    logits = predictor.predict_proba(test_dataset)
    assert type(logits) == np.ndarray
    assert logits.shape[0] == len(test_dataset)
    assert logits.shape[1] == predictor.num_labels

    # test __call__
    output_examples = predictor(
        input_multilabel_examples_without_labels,
        inference_config.max_length,
        threshold)

    assert type(output_examples) == list
    assert len(output_examples) == len(input_multilabel_examples_without_labels)
    for example in output_examples:
        assert type(example) == OutputMultilabelExample
        assert type(example.labels) == list
        assert [lab in class_labels for lab in example.labels]
        assert type(example.logits) == list
        assert len(example.labels) == len(example.logits)
