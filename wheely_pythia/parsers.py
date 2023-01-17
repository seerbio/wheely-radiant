"""
`parsers`: module for Pythia results parsing functions
"""

from pyspark.sql import SparkSession as _SparkSession
from wheely.mammoth import PsmDataset as _PsmDataset
from wheely.mammoth.utils import listify as _listify


def read_pythia_features(scored_files, spark=None) -> _PsmDataset:
    """
    Read scored PSMs from Pythia `.psm.scored` files.

    Parameters
    ----------
    scored_files : str or tuple of str
        Paths or URIs specifying a collection of PSMs in Pythia's `.scored` (HDF) format.
    spark : SparkSession (optional)
        If `None`, creates a default session.

    Returns
    -------
    PsmDataset
        A :py:class:`wheely.mammoth.dataset.PsmDataset` object containing the parsed PSMs.
    """
    if not spark:
        spark = _SparkSession.builder.getOrCreate()

    file_paths = [str(p) for p in _listify(scored_files)]

    raise NotImplementedError("TODO")  # TODO
