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


def test_compute_multilabel_accuracy() -> None:
    """
    test the compute_multilabel_accuracy computes the
    expected accuracy results
    """
    label_ids = np.array([[0, 0, 1, 1],
                          [0, 0, 1, 0],
                          [1, 0, 0, 0]])
    predictions = np.zeros(label_ids.shape)
    expected_accuracy = 8 / 12

    eval_prediction = EvalPrediction(predictions, label_ids)
    accuracy = training_utils.compute_multilabel_accuracy(eval_prediction)

    assert type(accuracy) == dict
    assert accuracy['accuracy'] == pytest.approx(expected_accuracy)


# @pytest.mark.skip(reason="long running unit test")
@pytest.mark.usefixtures("multilabel_dataset")
@pytest.mark.usefixtures("num_labels")
@pytest.mark.parametrize("do_eval", [True, False])
@pytest.mark.parametrize("do_class_weights", [True, False])
@pytest.mark.parametrize("do_focal_loss", [True, False])
def test_train_multilabel_classifier(
        multilabel_dataset: dataset_utils.MultilabelDataset,
        num_labels: int,
        do_eval: bool,
        do_class_weights: bool,
        do_focal_loss: bool) -> None:
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

    # run training
    eval_dataset = multilabel_dataset if do_eval else None
    training_utils.train_multilabel_classifier(
        multilabel_dataset,
        eval_dataset,
        model_config,
        num_labels,
        training_arguments,
        use_fast=model_utils.USE_FAST_TOKENIZER,
        do_eval=do_eval,
        do_class_weights=do_class_weights,
        do_focal_loss=do_focal_loss,
        focal_loss_gamma=1.0 if do_focal_loss else None
    )

    # check that model was written to output dir
    assert os.path.exists(os.path.join(tmp_dir, "pytorch_model.bin"))
    # if do_eval, check that eval_results.txt was written to output dir
    if do_eval:
        assert os.path.exists(os.path.join(tmp_dir, "eval_results.txt"))

    # clean up
    shutil.rmtree(tmp_dir)
