"""
Utilities for running model training
"""

import os
from typing import Dict, Optional, List
from datetime import datetime

import numpy as np
import torch
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    EvalPrediction,
    set_seed
)

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, hamming_loss

from . import configs, dataset_utils, model_utils


def build_compute_metrics(class_labels):
    """
    pass class_labels through to compute_metrics
    """
    def compute_metrics(
        prediction: EvalPrediction,
        threshold: float = 0.5) -> Dict[str, float]:
        """
        Compute f1, precision, recall for each label
        Compute full label accuracy and hamming loss (fraction of wrong labels: total labels)
        """
        scores = torch.sigmoid(torch.tensor(prediction.predictions)).numpy()
        preds = (scores > threshold).astype(int)
        labels = prediction.label_ids
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
        hamming_loss_metric = hamming_loss(labels, preds)
        accuracy = accuracy_score(labels, preds)
        per_label_metrics = {}
        for idx in range(len(class_labels)):
            per_label_metrics[str(class_labels[idx]) + "_precision"] = precision[idx]
            per_label_metrics[str(class_labels[idx]) + "_recall"] = recall[idx]
            per_label_metrics[str(class_labels[idx]) + "_f1"] = f1[idx]
        per_label_metrics["full_label_accuracy"] = accuracy
        per_label_metrics["hamming_loss"] = hamming_loss_metric
        return per_label_metrics
    
    return compute_metrics


def train_multilabel_classifier(
        train_dataset: dataset_utils.MultilabelDataset,
        eval_dataset: Optional[dataset_utils.MultilabelDataset],
        model_config: configs.ModelConfig,
        num_labels: int,
        training_arguments: dict,
        class_labels: List[str],
        append_eval_results: bool,
        use_fast: bool = model_utils.USE_FAST_TOKENIZER,
        do_eval: bool = True,
        do_class_weights: bool = False) -> None:
    """
    training loop for multi-label classification
    Notes:
       training_arguments: "output_dir" is a required key in the dictionary
       do_eval: if True, evaluation during training (every logging_steps steps)
          and at the end of training is performed. Note that if eval_dataset is None,
          do_eval is automatically reset to False
    """

    if do_class_weights:
        class_weights = dataset_utils.compute_class_weights(train_dataset.labels)
    else:
        class_weights = None

    # build training args
    do_eval = (do_eval) and (eval_dataset is not None)  # if eval_dataset is None, set do_eval to False
    training_arguments["do_eval"] = do_eval  # do evaluation every logging_steps steps
    training_arguments["report_to"] = training_arguments.get("report_to", "none")
    parser = HfArgumentParser(TrainingArguments)
    training_args = parser.parse_dict(training_arguments)[0]

    # set seed for training
    set_seed(training_args.seed)

    # load model and tokenizer
    model, tokenizer = model_utils.load_pretrained_model_and_tokenizer(
        model_config, num_labels, use_fast)

    # define compute metrics method and use class_labels
    compute_metrics = build_compute_metrics(class_labels)

    # set up trainer
    trainer = model_utils.MultilabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )
    trainer.set_class_weights(class_weights)

    # train and save model and tokenizer
    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    # eval and write results to txt file
    if do_eval:
        eval_results = trainer.evaluate()
        eval_filepath = os.path.join(training_args.output_dir, "eval_results.txt")
        if append_eval_results:
            with open(eval_filepath, "a") as f:
                # default to "w", use "a" when appending results
                timestamp = datetime.timestamp(datetime.now())
                timestamp_string = "Timestamp: " + str(datetime.fromtimestamp(timestamp))
                f.write("%s\n" % (timestamp_string))
                for k, v in sorted(eval_results.items()):
                    f.write("%s = %s\n" % (k, str(v)))
        else:
            with open(eval_filepath, "w") as f:
                for k, v in sorted(eval_results.items()):
                    f.write("%s = %s\n" % (k, str(v)))

    # release GPU memory
    torch.cuda.empty_cache()
