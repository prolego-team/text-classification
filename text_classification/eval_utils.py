"""
utility functions for evaluating the performance of a trained model
"""

from typing import List, Dict, Tuple

import numpy as np
from sklearn.metrics import multilabel_confusion_matrix as sklearn_multilabel_confusion_matrix

from . import dataset_utils


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


def multilabel_precision_recall(
        test_examples: List[dataset_utils.InputMultilabelExample],
        prediction_examples: List[dataset_utils.OutputMultilabelExample]
    ) -> Tuple[float, float]:
    """
    Compute precision and recall, microaveraged across all classes
    """
    tp_count = 0
    fp_count = 0
    fn_count = 0
    for pred_example, true_example in zip(prediction_examples, test_examples):
        all_labels = list(set(pred_example.labels + true_example.labels))
        for label in all_labels:
            if label in true_example.labels:
                if label in pred_example.labels:
                    # true positive
                    tp_count += 1
                else:
                    # false negative
                    fn_count += 1
            else:
                # false positive
                fp_count += 1

    precision = tp_count / (tp_count + fp_count)
    recall = tp_count / (tp_count + fn_count)

    return precision, recall
