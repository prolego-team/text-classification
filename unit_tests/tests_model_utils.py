"""
unit tests for text_classification/model_utils.py
"""

import pytest
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaTokenizerFast

from text_classification import model_utils, configs


@pytest.mark.parametrize("use_fast", [True, False])
def test_load_pretrained_model_and_tokenizer(use_fast: bool) -> None:
    """
    test types returned by load_pretrained_model_and_tokenizer
    """

    model_config = configs.ModelConfig("roberta-base", "main", None, "classification")
    num_labels = 5

    model, tokenizer = model_utils.load_pretrained_model_and_tokenizer(
        model_config,
        num_labels,
        use_fast
    )

    assert type(model) == RobertaForSequenceClassification
    if use_fast:
        assert type(tokenizer) == RobertaTokenizerFast
    else:
        assert type(tokenizer) == RobertaTokenizer
