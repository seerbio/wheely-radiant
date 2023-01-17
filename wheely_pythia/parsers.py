"""
`parsers`: module for Pythia results parsing functions
"""
import logging as _logging
from typing import Optional as _Optional

import h5py as _h5
import pandas as _pd
from pyspark.sql import SparkSession as _SparkSession
from pyspark.sql.functions import col as _col
from wheely.mammoth import PsmDataset as _PsmDataset
from wheely.mammoth.utils import listify as _listify


def read_pythia_features(
    scored_files,
    spark: _Optional[_SparkSession] = None,
    num_partitions: _Optional[int] = None,
) -> _PsmDataset:
    """
    Read scored PSMs from Pythia `.psm.scored` files.

    Parameters
    ----------
    scored_files : str or tuple of str
        Paths or URIs specifying a collection of PSMs in Pythia's `.scored` (HDF) format.
    spark : :py:class:`pyspark.sql.SparkSession` (optional)
        If `None`, creates a default session.
    num_partitions: int (optional)
        The number of partitions the list of files should be split into for reading.
        If unset (`None`), falls back to the Spark context's default.

    Returns
    -------
    PsmDataset
        A :py:class:`wheely.mammoth.dataset.PsmDataset` object containing the parsed PSMs.
    """
    if not spark:
        spark = _SparkSession.builder.getOrCreate()

    file_paths = [str(p) for p in _listify(scored_files)]

    # Distribute the file paths
    files_rdd = spark.sparkContext.parallelize(
        file_paths, numSlices=num_partitions
    )

    # Read the files in parallel and concatenate to one large RDD
    psms_rdd = files_rdd.flatMap(read_pythia_scored_rows)

    # Create the PySpark SQL DataFrame and assign a boolean "target" col
    psms_df = spark.createDataFrame(psms_rdd).withColumn(
        "target", _col("isDecoy") == 0
    )

    _logging.debug("Read dataframe with columns: %s", psms_df.columns)

    # Allow for typo in Pythia
    mean_abs_ppm_col = (
        "meanAbsolueErrorPPM"
        if "meanAbsolueErrorPPM" in psms_df.columns
        else "meanAbsoluteErrorPPM"
    )

    return _PsmDataset(
        psms_df,
        target_column="target",
        spectrum_columns=["filename", "scanNumber"],
        score_columns=[
            "cosine_similarity",
            "klDivergence",
            "Score",
            "hyperscore",
            "deltaScore",
            "meanErrorPPM",
            mean_abs_ppm_col,
            "leftOverRawScanIntensity",
            "extractedIonCount",
            "aCount",
            "bCount",
            "yCount",
            "b2Count",
            "y2Count",
            "yNH3Count",
            "yH2OCount",
            "bNH3Count",
            "bH2OCount",
            "scanRank",
        ],
        peptide_column="peptideId",
        protein_column="fastaDescriptions",
        protein_delim=";",
    )


def read_pythia_scored_rows(path) -> iter:
    """
    Read a single Pythia `.scored` file and return an iterator over its rows.
    Suitable for e.g. flat mapping with Spark over an RDD of file paths.
    """
    return read_pythia_scored_file(path).itertuples()


def read_pythia_scored_file(path) -> _pd.DataFrame:
    """
    Read a single Pythia `.scored` file as a :py:class:`pandas.DataFrame`
    """

    with _h5.File(path, "r") as f:
        df = _pd.DataFrame(f["psmResultsScoredDataset"][()])

    df["fastaDescriptions"] = df["fastaDescriptions"].str.decode("utf-8")

    df["peptideSequence"] = df["peptideSequence"].str.decode("utf-8")

    df["modificationString"] = df["modificationString"].str.decode("utf-8")

    previous_residue: str = (
        df["previousResidue"].values.tobytes().decode("utf-8")
    )

    df["previousResidue"] = [*previous_residue]

    post_residue: str = df["postResidue"].values.tobytes().decode("utf-8")
    df["postResidue"] = [*post_residue]

    df["peptideLength"] = df.peptideSequence.str.len()

    df["filename"] = path

    return df
