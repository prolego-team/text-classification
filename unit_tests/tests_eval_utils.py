"""
unit tests for eval_utils.py
"""


from typing import List

import numpy as np
import pytest
from sklearn.base import _pprint

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
