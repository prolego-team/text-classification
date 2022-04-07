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
    Focal loss places greater emphasis on mis-classified examples.
    It may be beneficial in cases where class imbalance exists and
    hard-to-classify examples are primarily found in the minory class.
    Original citation: https://arxiv.org/pdf/1708.02002.pdf
    """
    def __init__(self, gamma: float = 0):
        """
        From  the focal loss paper:
           gamma > 0 reduces the relative loss for well-classified examples,
              putting more focus on hard, misclassified examples.
        When gamma = 0, this is just binary cross entropy loss.
        """
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        """
        cross entropy loss:
          CE = -1 * log(pt)
            ---> pt = e^(-1 * CE)
        focal loss = -1 * (1 - pt) ** gamma * log(pt)
                   = CE * (1 - pt) ** gamma
        """
        CE = functional.binary_cross_entropy_with_logits(input, target, reduction="none")
        pt = torch.exp(-1 * CE)
        loss = CE * (1 - pt) ** self.gamma

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

        # override defaults if provided
        if kwargs.get("class_weights") is not None:
            # modify loss function to use class weights
            self.class_weights = torch.FloatTensor(kwargs["class_weights"])
            if torch.cuda.is_available():
                self.class_weights = self.class_weights.to("cuda")
            self.loss_function = torch.nn.BCEWithLogitsLoss(
                pos_weight=self.class_weights
            )
        if kwargs.get("focal_loss_gamma"):
            # set focal loss gamma (only used if "do_focal_loss" is True)
            self.focal_loss_gamma = kwargs["focal_loss_gamma"]
        if kwargs.get("do_focal_loss"):
            # use focal loss to place greater weight on misclassified examples
            self.loss_function = FocalLoss(gamma=self.focal_loss_gamma)

        # reconstruct kwargs without custom fields
        custom_key_names = ["class_weights", "do_focal_loss", "focal_loss_gamma"]
        kwargs = {k: v for k, v in kwargs.items() if k not in custom_key_names}

        # init Trainer
        super().__init__(*args, **kwargs)

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
