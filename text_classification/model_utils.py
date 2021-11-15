"""
Utilities for training models
"""

from typing import Tuple, Optional, List
from copy import deepcopy

import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer
)

from . import configs


USE_FAST_TOKENIZER = False


def load_pretrained_model_and_tokenizer(
        model_config: configs.ModelConfig,
        num_labels: int,
        use_fast: bool = USE_FAST_TOKENIZER) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    load a pretrained transformers model and associated tokenizer
    """

    config = AutoConfig.from_pretrained(
        model_config.model_name_or_dirpath,
        revision=model_config.revision,
        num_labels=num_labels,
        finetuning_task="classification",
        # hidden_dropout_prob=0.1,
        # attention_probs_dropout_prob=0.1
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_dirpath,
        revision=model_config.revision,
        config=config
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_dirpath,
        revision=model_config.revision,
        use_fast=use_fast
    )

    return model, tokenizer


class MultilabelTrainer(Trainer):
    """
    A multi-label trainer just overrides the default compute_loss
    function with binary cross-entropy loss.
    Also includes support for weighted loss to address class imbalance.
    """
    def set_class_weights(self, class_weights: Optional[List[float]]):
        """
        add class weights
        Note: This needs to be run before model training. If class weights
        are not required, pass None as the input.
        """
        # TODO: There's probably a cleaner way to do this, e.g., modifying the __init__
        # to take class_weights as an optional input.
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        override default loss function with binary cross-entropy loss
        for multi-label classification
        """
        # the eval step expects inputs.labels to exist, but compute_loss
        # pops "labels", so we need to make a copy before extracting labels
        inputs_copy = deepcopy(inputs)
        labels = inputs_copy.pop("labels")
        outputs = model(**inputs_copy)
        logits = outputs.logits
        if self.class_weights is not None:
            class_weights = torch.FloatTensor(self.class_weights)
            if torch.cuda.is_available():
                class_weights = class_weights.to("cuda")
            loss_function = torch.nn.BCEWithLogitsLoss(
                pos_weight=class_weights
            )
        else:
            loss_function = torch.nn.BCEWithLogitsLoss()

        loss = loss_function(logits.view(-1, self.model.config.num_labels),
                             labels.float().view(-1, self.model.config.num_labels))

        if return_outputs:
            return (loss, outputs)
        else:
            return loss
