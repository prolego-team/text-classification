"""
unit tests for text_classification/configs.py
"""

import os
from tempfile import mkdtemp
import shutil

from text_classification import configs


def test_read_config_for_training() -> None:
    """
    test output types returned by read_config_for_training
    """
    training_config_filepath = "test_data/training_config.json"
    training_config = configs.read_config_for_training(training_config_filepath)

    assert type(training_config) == configs.TrainingConfig
    assert type(training_config.model_config) == configs.ModelConfig


def test_read_config_for_inference() -> None:
    """
    test output types returned by read_config_for_inference
    """
    inference_config_filepath = "test_data/inference_config.json"
    inference_config = configs.read_config_for_inference(inference_config_filepath)

    assert type(inference_config) == configs.InferenceConfig
    assert type(inference_config.model_config) == configs.ModelConfig
    assert inference_config.num_labels == len(inference_config.class_labels)


def test_save_config_for_inference() -> None:
    """
    test that an output file is created by save_config_for_inference
    """

    inference_config = configs.read_config_for_inference("test_data/inference_config.json")

    tmp_dir = mkdtemp()
    out_filepath = os.path.join(tmp_dir, "inference_config.json")

    configs.save_config_for_inference(inference_config, out_filepath)

    assert os.path.exists(out_filepath)

    # check that the created inference config contains identical content
    # as the original
    new_inference_config = configs.read_config_for_inference(out_filepath)
    assert new_inference_config.num_labels == inference_config.num_labels
    for attribute in inference_config.model_config.__dict__:
        assert getattr(new_inference_config.model_config, attribute) == \
            getattr(inference_config.model_config, attribute)

    # clean up
    shutil.rmtree(tmp_dir)
