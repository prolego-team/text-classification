import os
from typing import Dict, Optional

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    HfArgumentParser,
    TrainingArguments,
    EvalPrediction,
    set_seed
)

from text_classification import dataset_utils, model_utils


def compute_multilabel_accuracy(
        prediction: EvalPrediction,
        threshold: float = 0.5) -> Dict[str, float]:
    """
    accuracy = # of correct predictions / # of total predictions
    """
    # apply activation
    scores = torch.sigmoid(torch.tensor(prediction.predictions)).numpy()
    preds = (scores > threshold).astype(int)
    return {"accuracy": np.sum(preds == prediction.label_ids) / (preds.shape[0] * preds.shape[1])}


def train_multilabel_classifier(
        train_dataset: dataset_utils.MultilabelDataset,
        eval_dataset: Optional[dataset_utils.MultilabelDataset],
        model: AutoModelForSequenceClassification,
        training_arguments: dict,
        do_eval: bool = True,
        do_class_weights: bool = False) -> None:
    """
    training loop for multi-label classification
    training_arguments: "output_dir" is a required key in the dictionary
    """

    if do_class_weights:
        class_weights = dataset_utils.compute_class_weights(train_dataset.labels)
    else:
        class_weights = None

    # set up trainer
    training_arguments["do_eval"] = do_eval
    parser = HfArgumentParser(TrainingArguments)
    training_args = parser.parse_dict(training_arguments)[0]
    trainer = model_utils.MultilabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_multilabel_accuracy
    )
    trainer.set_class_weights(class_weights)

    # set seed for training
    set_seed(training_args.seed)

    # train and save model
    trainer.train()
    trainer.save_model(training_args.output_dir)

    # eval and write results to txt file
    if do_eval:
        eval_results = trainer.evaluate()
        eval_filepath = os.path.join(training_args.output_dir, "eval_results.txt")
        with open(eval_filepath, "w") as f:
            for k, v in sorted(eval_results.items()):
                f.write("%s = %s\n" % (k, str(v)))

    # release GPU memory
    torch.cuda.empty_cache
