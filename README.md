# text-classification

Build a transformers model to classify text, built on top of HuggingFace transformers and pytorch libraries.

Developed using python 3.9.5

## Getting Started

Clone the git repo:

        git clone https://github.com/prolego-team/text-classification.git

Create a virtual environment and install package dependencies using `pip` (v. 21.2.1):

        cd text-classification
        python3 -m venv .venv
        source .venv/bin/activate
        pip install -r requirements.txt

The environment only needs to be set up once. After it has been created, it can be activated using the command:

        cd text-classification
        source .venv/bin/activate

Test set-up by running the unit tests and ensuring that they all pass:

        python -m pytest unit_tests

## Examples

### Classify Sentences from Books

Build a multi-label classification model to classify sentences as belonging to one or more books.

To train:

        python -m examples.books.train <args> <optional_args>

A trained model and inference config json file will be created. To run inference:

        python -m examples.books.predict <args>

Predictions for each sentence will be written to an output .tsv file.