"""
configuration classes and utilities for models, training, and inference
"""

from dataclasses import dataclass
from typing import List, Optional
import json


@dataclass
class ModelConfig():
    """
    Contains some basic information about a model:
       model_name_or_dirpath: Path to base model, can be either a
          filesystem location or location in huggingface.co
       revision: SHA revision ID for a HuggingFace model
       saved_model_dirpath: Path to save the output model following
          training. Set to None if not relevant
       task_name: currently unused
    """
    model_name_or_dirpath: str
    revision: str
    saved_model_dirpath: Optional[str]
    task_name: str


@dataclass
class TrainingConfig():
    """
    configuration settings for model training workflow
    """
    model_config: ModelConfig


@dataclass
class InferenceConfig():
    """
    configuration settings for model inference workflow
    """
    model_config: ModelConfig
    class_labels: List[str]
    max_length: int

    def __post_init__(self):
        """
        this is run following initialization
        """
        # derive the number of labels from the list of class labels
        self.num_labels = len(self.class_labels)


def training_config_from_dict(training_dict: dict) -> TrainingConfig:
    """
    Create a training config from a dictionary of training configurations.
    See unit_tests/tests_configs.py::test_training_config_from_dict
    for an example of the expected input.
    """

    # create model config
    model_data = training_dict.get("model")
    model_config = ModelConfig(**model_data)

    return TrainingConfig(model_config)


def read_config_for_training(training_config_filepath: str) -> TrainingConfig:
    """
    Create a training config from a json file containing training configurations.
    See test_data/training_config.json for an example of the expected input.
    """

    with open(training_config_filepath, "r") as f:
        data = json.load(f)

    return training_config_from_dict(data)


def inference_config_from_dict(inference_dict: dict) -> InferenceConfig:
    """
    Create an inference config from a dictionary of inference configurations.
    """

    # create model config
    model_data = inference_dict.get("model")
    model_config = ModelConfig(**model_data)

    class_labels = [str(label) for label in inference_dict.get("class_labels")]
    max_length = int(inference_dict.get("max_length"))

    return InferenceConfig(model_config, class_labels, max_length)


def read_config_for_inference(inference_config_filepath: str) -> InferenceConfig:
    """
    Create an inference config from a json file containing inference configurations.
    See test_data/inference_config.json for an example of the expected input.
    """

    with open(inference_config_filepath, "r") as f:
        data = json.load(f)

    return inference_config_from_dict(data)


def save_config_for_inference(
        inference_config: InferenceConfig,
        inference_config_filepath: str) -> None:
    """
    write inference_config to a file
    """

    inference_data = {
        "class_labels": inference_config.class_labels,
        "max_length": inference_config.max_length,
        "model": vars(inference_config.model_config)
    }

    with open(inference_config_filepath, "w") as f:
        json.dump(inference_data, f, indent=4, sort_keys=True)
