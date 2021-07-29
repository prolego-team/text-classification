"""
unit tests for text_classification/training_utils.py
"""

import os
import shutil
from tempfile import mkdtemp

import pytest
import numpy as np
from transformers import EvalPrediction, AutoTokenizer

from text_classification import configs, training_utils, dataset_utils, model_utils


@pytest.fixture
def num_labels() -> int:
    return 4


@pytest.fixture
def multilabel_dataset(num_labels: int) -> dataset_utils.MultilabelDataset:
    """
    create a dummy multilabel dataset
    """
    labels = ["Label " + str(i) for i in range(num_labels)]
    examples = [dataset_utils.InputMultilabelExample(i, "Text " + str(i), labels)
                for i in range(10)]
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    max_length = 128
    return dataset_utils.MultilabelDataset(
        examples,
        labels,
        tokenizer,
        max_length,
        predict=False
    )


def test_compute_multilabel_accuracy() -> None:
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
    model_config = configs.ModelConfig("roberta-base", "main", None, "classification")
    model, _ = model_utils.load_pretrained_model_and_tokenizer(
        model_config, num_labels, use_fast=False)

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
        model,
        training_arguments,
        do_eval,
        do_class_weights
    )

    # check that model was written to output dir
    assert os.path.exists(os.path.join(tmp_dir, "pytorch_model.bin"))
    # if do_eval, check that eval_results.txt was written to output dir
    if do_eval:
        assert os.path.exists(os.path.join(tmp_dir, "eval_results.txt"))

    # clean up
    shutil.rmtree(tmp_dir)
