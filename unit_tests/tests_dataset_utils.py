"""
unit tests for dataset_utils.py
"""

import os
import shutil
from typing import List
from tempfile import mkdtemp

import pytest
from transformers import AutoTokenizer, BatchEncoding

from text_classification import dataset_utils


@pytest.mark.parametrize("predict", [True, False])
@pytest.mark.usefixtures("multilabel_examples")
@pytest.mark.usefixtures("class_labels")
def test_MultilabelDataset(
        multilabel_examples: List[dataset_utils.InputMultilabelExample],
        class_labels: List[str],
        predict: bool) -> None:
    """
    test attributes of MultilabelDataset class
    """
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    max_length = 128
    dataset = dataset_utils.MultilabelDataset(
        multilabel_examples,
        class_labels,
        tokenizer,
        max_length,
        predict
    )

    # dataset.num_labels
    assert type(dataset.num_labels) == int
    assert dataset.num_labels == len(class_labels)
    n_examples = len(multilabel_examples)
    # dataset.texts
    assert type(dataset.texts) == list
    assert len(dataset.texts) == n_examples
    # dataset.labels (one-hot encoded labels)
    assert type(dataset.labels) == list
    assert len(dataset.labels) == n_examples
    for label in dataset.labels:
        assert len(label) == len(class_labels)
        for one_hot in label:
            assert one_hot in [0, 1]
    # dataset.encodings (tokenized text)
    assert type(dataset.encodings) == BatchEncoding


def test_compute_class_weights() -> None:
    """
    test that compute_class_weights returns expected results
    using some dummy data
    """
    one_hot_labels = [[0, 0, 0, 1],
                      [0, 1, 1, 0],
                      [1, 1, 0, 0]]
    pos_counts = [1, 2, 1, 1]
    neg_counts = [2, 1, 2, 2]
    expected_class_weights = [neg / pos for pos, neg in zip(pos_counts, neg_counts)]

    actual_class_weights = dataset_utils.compute_class_weights(one_hot_labels)
    for expected, actual in zip(expected_class_weights, actual_class_weights):
        assert actual == pytest.approx(expected)


@pytest.mark.usefixtures("multilabel_examples")
def test_multilabel_examples_to_tsv(
        multilabel_examples: List[dataset_utils.InputMultilabelExample]) -> None:
    """
    test that output tsv file is created by multilabel_exampels_to_tsv
    """
    tmp_dir = mkdtemp()
    out_tsv_filepath = os.path.join(tmp_dir, "out.tsv")
    dataset_utils.multilabel_examples_to_tsv(
        multilabel_examples,
        out_tsv_filepath
    )
    assert os.path.exists(out_tsv_filepath)

    # clean up temporary directory
    shutil.rmtree(tmp_dir)


@pytest.mark.parametrize("generate_guids", [True, False])
@pytest.mark.parametrize("tsv_filename", ["multilabel_examples.tsv",
                                          "multilabel_examples_without_labels.tsv"])
def test_tsv_to_multilabel_examples(
        generate_guids: bool,
        tsv_filename: str) -> None:
    """
    test that tsv is parsed into multilabel examples
    """
    tsv_filepath = os.path.join("test_data", tsv_filename)
    multilabel_examples = dataset_utils.tsv_to_multilabel_examples(
        tsv_filepath, generate_guids=generate_guids)
    for example in multilabel_examples:
        assert type(example) == dataset_utils.InputMultilabelExample


@pytest.mark.parametrize("example_dictionaries", ["multilabel_example_dictionaries",
                                                  "multilabel_example_dictionaries_without_labels"])
@pytest.mark.parametrize("generate_guids", [True, False])
def test_dictionaries_to_multilabel_examples(
        example_dictionaries,
        generate_guids,
        request) -> None:
    """
    test that a list of multilabel examples are created
    """
    example_dictionaries = request.getfixturevalue(example_dictionaries)
    multilabel_examples = dataset_utils.dictionaries_to_multilabel_examples(
        example_dictionaries, generate_guids=generate_guids)
    for example in multilabel_examples:
        assert type(example) == dataset_utils.InputMultilabelExample


@pytest.mark.parametrize("examples", ["multilabel_examples",
                                      "multilabel_examples_without_labels"])
def test_sorted_class_labels(
        examples: List[dataset_utils.InputMultilabelExample],
        request) -> None:
    """
    test that class labels are extracted from a list of examples
    """
    examples = request.getfixturevalue(examples)
    labels = dataset_utils.sorted_class_labels(examples)

    assert type(labels) == list
