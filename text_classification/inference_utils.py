from typing import List, Optional
from tempfile import mkdtemp

import numpy as np
import torch
from transformers.training_args import TrainingArguments

from text_classification import configs, dataset_utils, model_utils


def one_hot_to_index_labels(one_hot_predictions: List[List[int]]) -> List[List[int]]:
    """
    Convert one-hot encoded predictions to a list of indices indicating
    the positions where the one-hot vectors are 1. Null entries (zeros
    for all positions) are assigned an index of -1
    """
    one_hot_predictions_array = np.array(one_hot_predictions)
    get_label_indices = lambda one_hot: np.where(one_hot == 1)[0].tolist()
    prediction_indices = [get_label_indices(row) if sum(row) > 0 else [-1]
                          for row in one_hot_predictions_array]
    return prediction_indices


def predict_multilabel_classes(
        model_config: configs.ModelConfig,
        class_list: List[str],
        max_length: int,
        examples: List[dataset_utils.InputMultilabelExample],
        thresholds: Optional[List[float]]) -> List[dataset_utils.InputMultilabelExample]:
    """
    generate predictions from a multi-label classification model
    """

    # load model and tokenizer
    num_labels = len(class_list)
    model, tokenizer = model_utils.load_pretrained_model_and_tokenizer(
        model_config,
        num_labels
    )

    # create dataset
    test_dataset = dataset_utils.MultilabelDataset(
        examples,
        class_list,
        tokenizer,
        max_length,
        predict=True
    )

    # set up trainer
    temp_dir = mkdtemp()
    training_args = TrainingArguments(
        output_dir=temp_dir,
        do_train=False,
        do_eval=False,
        do_predict=True
    )
    trainer = model_utils.MultilabelTrainer(
        model=model,
        args=training_args
    )
    trainer.set_class_weights(None)

    # make predictions
    predictions = trainer.predict(test_dataset=test_dataset).predictions
    out = torch.sigmoid(torch.tensor(predictions)).numpy()
    if thresholds is None:
        # just use a sensible 0.5 cutoff
        thresholds = [0.5] * num_labels
    one_hot_predictions = (out > np.array(thresholds)).astype(int).tolist()

    # convert predictions to labels
    prediction_indices = one_hot_to_index_labels(one_hot_predictions)
    index_to_label = {i: lab for i, lab in enumerate(class_list)}
    index_to_label[-1] = "null"
    prediction_labels = [index_to_label[index] for indices in prediction_indices for index in indices]

    output_examples = [
        dataset_utils.InputMultilabelExample(guid, text, labels)
        for guid, text, labels in zip(test_dataset.guids, test_dataset.texts, prediction_labels)
    ]

    torch.cuda.empty_cache()

    return output_examples
