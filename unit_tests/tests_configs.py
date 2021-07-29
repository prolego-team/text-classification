"""
unit tests for text_classification/configs.py
"""

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
