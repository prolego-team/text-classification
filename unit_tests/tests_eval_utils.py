"""
unit tests for eval_utils.py
"""


from typing import List

import numpy as np
import pytest

from text_classification import dataset_utils, eval_utils


@pytest.mark.usefixtures("input_multilabel_examples")
@pytest.mark.usefixtures("output_multilabel_examples")
@pytest.mark.usefixtures("class_labels")
def test_multilabel_confusion_matrix(
        input_multilabel_examples: List[dataset_utils.InputMultilabelExample],
        output_multilabel_examples: List[dataset_utils.OutputMultilabelExample],
        class_labels: List[str]) -> None:
    """
    test that multilabel_confusion_matrix returns a confusion matrix
    for each class, and that the values in the confusion matrices
    match expected
    """
    confusion_matrices = eval_utils.multilabel_confusion_matrix(
        input_multilabel_examples, output_multilabel_examples, class_labels)
    for k, v in confusion_matrices.items():
        assert k in class_labels
    assert np.all(v == np.array([[0, 0],
                                 [0, len(input_multilabel_examples)]]))


@pytest.mark.usefixtures("input_multilabel_examples")
@pytest.mark.usefixtures("output_multilabel_examples")
def test_multilabel_precision_recall(
        input_multilabel_examples: List[dataset_utils.InputMultilabelExample],
        output_multilabel_examples: List[dataset_utils.OutputMultilabelExample]
    ) -> None:
    """
    test that multilabel_precision_recall returns expected precision and
    recall scores on example data
    """

    precision, recall = eval_utils.multilabel_precision_recall(
        input_multilabel_examples,
        output_multilabel_examples
    )
    # TP: 40, FP: 0, FN: 0
    assert precision == 40 / (40 + 0)
    assert recall == 40 / (40 + 0)

    # prediction is always missing Label 3
    output_multilabel_examples = [
        dataset_utils.OutputMultilabelExample(
            e.guid, e.text, ["Label 0", "Label 1", "Label 2"], [1.0] * 3)
        for e in output_multilabel_examples]
    # TP: 30, FP: 0, FN: 10
    precision, recall = eval_utils.multilabel_precision_recall(
        input_multilabel_examples,
        output_multilabel_examples
    )
    assert precision == 30 / (30 + 0)
    assert recall == 30 / (30 + 10)

    # prediction always has an additional label, Label 4
    output_multilabel_examples = [
        dataset_utils.OutputMultilabelExample(
            e.guid, e.text, ["Label 0", "Label 1", "Label 2", "Label 3", "Label 4"], [1.0] * 4)
        for e in output_multilabel_examples
    ]
    # TP: 40, FP: 10, FN: 0
    precision, recall = eval_utils.multilabel_precision_recall(
        input_multilabel_examples,
        output_multilabel_examples
    )
    assert precision == 40 / (40 + 10)
    assert recall == 40 / (40 + 0)

    # prediction is always missing a label AND has an additional label
    output_multilabel_examples = [
        dataset_utils.OutputMultilabelExample(
            e.guid, e.text, ["Label 0", "Label 1", "Label 2", "Label 4"], [1.0] * 4)
        for e in output_multilabel_examples
    ]
    # TP: 30, FP: 10, FN: 10
    precision, recall = eval_utils.multilabel_precision_recall(
        input_multilabel_examples,
        output_multilabel_examples
    )
    assert precision == 30 / (30 + 10)
    assert recall == 30 / (30 + 10)
