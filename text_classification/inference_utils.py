"""
Utilities for running inference (i.e., prediction) on a
trained model
"""

from typing import List, Union, Tuple
from tempfile import mkdtemp
import shutil

import numpy as np
import torch
from transformers.training_args import TrainingArguments

from . import configs, dataset_utils, model_utils


def one_hot_to_index_labels(one_hot_predictions: List[List[int]]) -> List[List[int]]:
    """
    Convert one-hot encoded predictions to a list of indices indicating
    the positions where the one-hot vectors are 1. Null entries (zeros
    for all positions) return empty lists
    """
    one_hot_predictions_array = np.array(one_hot_predictions)
    get_label_indices = lambda one_hot: np.where(one_hot == 1)[0].tolist()
    prediction_indices = [get_label_indices(row) if sum(row) > 0 else []
                          for row in one_hot_predictions_array]
    return prediction_indices


class MultilabelPredictor:
    """
    A MultilabelPredictor is used to run the prediction workflow
    over a set of examples using a trained multilabel classification model.
    """
    def __init__(
            self,
            model_config: configs.ModelConfig,
            class_list: List[str],
            use_fast_tokenizer: bool = model_utils.USE_FAST_TOKENIZER,
            dataloader_pin_memory: bool = True) -> None:
        """
        compute number of labels from class_list and load model and tokenizer
        """
        self.class_list = class_list
        self.num_labels = len(self.class_list)
        self.dataloader_pin_memory = dataloader_pin_memory

        self.model, self.tokenizer = model_utils.load_pretrained_model_and_tokenizer(
            model_config,
            self.num_labels,
            use_fast=use_fast_tokenizer
        )

    def create_dataset(
            self,
            examples: List[dataset_utils.InputMultilabelExample],
            max_length: int) -> dataset_utils.MultilabelDataset:
        """
        create a multilabel dataset for prediction from a list
        of input multilabel examples
        """
        test_dataset = dataset_utils.MultilabelDataset(
            examples,
            self.class_list,
            self.tokenizer,
            max_length,
            predict=True
        )
        return test_dataset

    def predict_proba(self, test_dataset: dataset_utils.MultilabelDataset) -> np.array:
        """
        predict multi-label classes for a test dataset

        returns a [num_examples x num_classes] array of confidences (output
        of the final sigmoid activation) representing class membership
        scores for each class.
        """
        # set up trainer
        temp_dir = mkdtemp()
        training_args = TrainingArguments(
            output_dir=temp_dir,
            report_to="none",
            do_train=False,
            do_eval=False,
            do_predict=True,
            dataloader_pin_memory=self.dataloader_pin_memory
        )
        trainer = model_utils.MultilabelTrainer(
            model=self.model,
            args=training_args
        )
        trainer.set_class_weights(None)

        # make predictions
        predictions = trainer.predict(test_dataset=test_dataset).predictions
        confidences = torch.sigmoid(torch.tensor(predictions))
        if not self.dataloader_pin_memory:
            confidences = confidences.cpu()
        confidences = confidences.numpy()

        # clean up
        shutil.rmtree(temp_dir)

        torch.cuda.empty_cache()
        return confidences

    def confidences_to_predicted_labels(
            self,
            confidences: np.array,
            threshold: Union[float, List[float]]) -> Tuple[List[List[str]], List[List[float]]]:
        """
        Convert [# of examples x # of classes] array of confidences to lists of
        predicted labels and associated confidences for the predictions.
        Outputs:
           predicted_labels: for each example, list of string labels for which
              confidences are greater than threshold
           predicted_confidences: for each example, list of confidences corresponding to
              the labels in predicted_labels
        """

        if type(threshold) == float:
            threshold = [threshold] * len(self.class_list)
        one_hot_predictions = (confidences > np.array(threshold)).astype(int).tolist()

        # convert predictions to labels
        index_to_label_mapping = {i: lab for i, lab in enumerate(self.class_list)}
        prediction_indices = one_hot_to_index_labels(one_hot_predictions)
        predicted_labels = [[index_to_label_mapping[index] for index in indices] for indices in prediction_indices]

        # extract confidences for labels
        def extract_positive_class_confidences(row_confidences, indices):
            return list(row_confidences[indices])
        predicted_confidences = [extract_positive_class_confidences(row, indices)
                                 for row, indices in zip(confidences , prediction_indices)]

        return predicted_labels, predicted_confidences

    def __call__(
            self,
            examples: List[dataset_utils.InputMultilabelExample],
            max_length: int,
            threshold: Union[float, List[float]] = 0.5) -> List[dataset_utils.OutputMultilabelExample]:
        """
        run the prediction workflow:
            1) create prediction-ready dataset
            2) run inference to generate confidences
            3) convert confidences to predictions with class labels
            4) build list of output multilabel examples
        """

        # create prediction-ready dataset
        test_dataset = self.create_dataset(examples, max_length)

        # run inference and generate [# of examples x # of classes] array of confidences
        all_class_confidences = self.predict_proba(test_dataset)

        # convert confidences to list of predicted labels and associated confidences for predictions
        predicted_class_labels, predicted_class_confidences = self.confidences_to_predicted_labels(
            all_class_confidences, threshold)

        # build output examples
        output_examples = [dataset_utils.OutputMultilabelExample(guid, text, labels, confidences)
                           for guid, text, labels, confidences in zip(test_dataset.guids,
                                                                      test_dataset.texts,
                                                                      predicted_class_labels,
                                                                      predicted_class_confidences)]
        return output_examples
