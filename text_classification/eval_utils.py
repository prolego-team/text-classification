"""
utility functions for evaluating the performance of a trained model
"""

from typing import List, Dict

import numpy as np
from sklearn.metrics import multilabel_confusion_matrix as sklearn_multilabel_confusion_matrix

from text_classification import dataset_utils


def multilabel_confusion_matrix(
        test_examples: List[dataset_utils.InputMultilabelExample],
        prediction_examples: List[dataset_utils.OutputMultilabelExample],
        class_labels: List[str]) -> Dict[str, np.ndarray]:
    """
    Generate a set of confusion matrices, one for each class in
    class_labels.
    Uses scikit-learn multilabel_confusion_matrix under the hood.
    """

    # convert examples to one-hot encodings
    test_one_hot_encodings = [dataset_utils.to_one_hot(example, class_labels)
                              for example in test_examples]
    prediction_one_hot_encodings = [dataset_utils.to_one_hot(example, class_labels)
                                    for example in prediction_examples]

    confusion_matrices = sklearn_multilabel_confusion_matrix(
        test_one_hot_encodings, prediction_one_hot_encodings)

    # put into a dictionary for readability
    out = {class_label: confusion_matrix
           for class_label, confusion_matrix in zip(class_labels, confusion_matrices)}

    return out
