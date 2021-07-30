"""
shared utility methods
"""

import pandas as pd


def dataframe_to_tsv(dataframe: pd.DataFrame, tsv_filepath: str) -> None:
    """
    Save a pandas dataframe to a tab-separated file.
    """
    dataframe.to_csv(tsv_filepath, sep="\t", index=False)


def tsv_to_dataframe(tsv_filepath: str) -> pd.DataFrame:
    """
    Read a tab-separated file into a dataframe.
    """
    dataframe = pd.read_csv(tsv_filepath, sep="\t")
    return dataframe
