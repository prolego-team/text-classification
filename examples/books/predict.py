"""
Example script to run inference on a trained model to classify sentences by book.

Usage:
    python -m examples.books.predict <args>
"""

import click

from text_classification import configs, inference_utils
from text_classification import dataset_utils


@click.command()
@click.argument("data_txt_filepath", type=click.Path(exists=True))
@click.argument("inference_config_filepath", type=click.Path(exists=True))
@click.argument("output_tsv_filepath", type=click.Path())
def main(**kwargs):
    """
    Run inference using a trained transformers model to predict the originating novel
    for a list of sentences.

    \b
    DATA_TXT_FILEPATH: Path to a .txt file containing sentences (one per row) for inference.
    INFERENCE_CONFIG_FILEPATH: Path to an inference config .json file. See test_data/inference_config.json
       for an example.
    OUTPUT_FILEPATH: Path to a .tsv file where sentences and predictions should be written. \f
    """

    # load data and convert to multilabel examples
    with open(kwargs["data_txt_filepath"], "r") as f:
        sentences = f.readlines()
    examples = []
    for i, sentence in enumerate(sentences):
        example = dataset_utils.InputMultilabelExample(i, sentence, None)
        examples.append(example)

    # read inference config
    inference_config = configs.read_config_for_inference(kwargs["inference_config_filepath"])

    # run inference
    prediction_examples = inference_utils.predict_multilabel_classes(
        inference_config.model_config,
        inference_config.class_labels,
        inference_config.max_length,
        examples,
        thresholds=None
    )

    # save output to a tsv file
    dataset_utils.multilabel_examples_to_tsv(
        prediction_examples,
        kwargs["output_tsv_filepath"]
    )


if __name__ == "__main__":
    main()
