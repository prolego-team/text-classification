"""
Utilities for training models
"""

from typing import Tuple, Optional, List
from copy import deepcopy

import torch
from torch.nn import Module, functional
from torch.autograd import Variable
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer
)

from . import configs


USE_FOCAL_LOSS = False
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


class FocalLoss(Module):
    """
    from: https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    """
    def __init__(self, gamma: float = 0):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        logpt = functional.logsigmoid(input)
        logpt = logpt[target == 1]
        pt = Variable(logpt.data.exp())

        loss = -1 * (1 - pt) ** self.gamma * logpt

        return loss.mean()


class MultilabelTrainer(Trainer):
    """
    A multi-label trainer just overrides the default compute_loss
    function with binary cross-entropy loss.
    Also includes support for weighted loss to address class imbalance.
    """
    def __init__(self, *args, **kwargs):
        self.class_weights = None
        self.focal_loss_gamma = 0.5
        self.loss_function = torch.nn.BCEWithLogitsLoss()

        # override defaults if provided, and remove from kwargs

        if "class_weights" in kwargs:
            if kwargs["class_weights"] is not None:
                # modify loss function to use class weights
                self.class_weights = torch.FloatTensor(kwargs["class_weights"])
                if torch.cuda.is_available():
                    self.class_weights = self.class_weights.to("cuda")
                self.loss_function = torch.nn.BCEWithLogitsLoss(
                    pos_weight=self.class_weights
                )
            kwargs.pop("class_weights")

        if "focal_loss_gamma" in kwargs:
            if kwargs["focal_loss_gamma"] is not None:
                self.focal_loss_gamma = kwargs["focal_loss_gamma"]
            kwargs.pop("focal_loss_gamma")

        if "do_focal_loss" in kwargs:
            if kwargs["do_focal_loss"]:
                self.loss_function = FocalLoss(gamma=self.focal_loss_gamma)
            kwargs.pop("do_focal_loss")

        # init Trainer
        super().__init__(*args, **kwargs)

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
        loss = self.loss_function(logits.view(-1, self.model.config.num_labels),
                                  labels.float().view(-1, self.model.config.num_labels))

        if return_outputs:
            return (loss, outputs)
        else:
            return loss
