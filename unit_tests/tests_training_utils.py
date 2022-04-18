"""
unit tests for text_classification/training_utils.py
"""

import os
import shutil
from tempfile import mkdtemp

import pytest
import numpy as np
from transformers import EvalPrediction

from text_classification import configs, training_utils, dataset_utils, model_utils


def test_compute_metrics() -> None:
    """
    test compute_metrics computes f1, precision and recall for each label
    and full label accuracy and hamming loss
    """
    class_labels = ["alice-in-wonderland", "frankenstein"]
    label_ids = np.array([
        [0, 1],
        [1, 0],
        [1, 1],
        [0, 0]
    ])
    predictions = np.array([
        [1, 1],
        [0, 1],
        [1, 0],
        [0, 1]
    ])

    eval_prediction = EvalPrediction(predictions, label_ids)
    compute_metrics = training_utils.build_compute_metrics(class_labels)
    all_metrics = compute_metrics(eval_prediction)
    assert all_metrics["full_label_accuracy"] == 0 / 8
    assert all_metrics["hamming_loss"] == 5 / 8
    assert all_metrics["alice-in-wonderland_precision"] == 1 / 2
    assert all_metrics["frankenstein_precision"] == 1 / 3
    assert all_metrics["alice-in-wonderland_recall"] == 1 / 2
    assert all_metrics["frankenstein_recall"] == 1 / 2


# @pytest.mark.skip(reason="long running unit test")
@pytest.mark.usefixtures("multilabel_dataset")
@pytest.mark.usefixtures("num_labels")
@pytest.mark.parametrize("do_eval", [True, False])
@pytest.mark.parametrize("do_class_weights", [True, False])
def test_train_multilabel_classifier(
        multilabel_dataset: dataset_utils.MultilabelDataset,
        num_labels: int,
        do_eval: bool,
        do_class_weights: bool) -> None:
    """
    test that model is trained and saved
    """

    # load model
    model_config = configs.ModelConfig("roberta-base", "main", None, "multilabel")

    # set up training arguments
    tmp_dir = mkdtemp()
    training_arguments = {
        "do_train": True,
        "num_train_epochs": 1.0,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 16,
        "do_predict": False,
        "block_size": 128,
        "output_dir": tmp_dir
    }

    class_labels = ["alice-in-wonderland", "frankenstein"]

    # run training
    eval_dataset = multilabel_dataset if do_eval else None
    training_utils.train_multilabel_classifier(
        multilabel_dataset,
        eval_dataset,
        model_config,
        num_labels,
        training_arguments,
        class_labels,
        append_eval_results=False,
        use_fast=model_utils.USE_FAST_TOKENIZER,
        do_eval=do_eval,
        do_class_weights=do_class_weights
    )

    # check that model was written to output dir
    assert os.path.exists(os.path.join(tmp_dir, "pytorch_model.bin"))
    # if do_eval, check that eval_results.txt was written to output dir
    if do_eval:
        assert os.path.exists(os.path.join(tmp_dir, "eval_results.txt"))

    # clean up
    shutil.rmtree(tmp_dir)
