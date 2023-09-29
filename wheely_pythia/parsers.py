"""
`parsers`: module for Pythia results parsing functions
"""
import logging as _logging
from typing import Optional as _Optional

import pandas as _pd
from pyspark.sql import SparkSession as _SparkSession
from pyspark.sql.functions import (
    col as _col,
    concat as _concat,
    input_file_name as _input_file_name,
    lit as _lit,
)
from wheely.mammoth import PsmDataset as _PsmDataset
from wheely.mammoth.utils import listify as _listify


def read_pythia_features(
    scored_files,
    spark: _Optional[_SparkSession] = None,
    num_partitions: _Optional[int] = None,
    **kwargs,
) -> _PsmDataset:
    """
    Read scored PSMs from Pythia `.psm.scored` files.

    Parameters
    ----------
    scored_files : str or tuple of str
        Paths or URIs specifying a collection of PSMs in Pythia's `.prq.pythiaDIA` format, or
        `.scored` (HDF) format (to be deprecated). Note: all file paths must be in the same
        format.
    spark : :py:class:`pyspark.sql.SparkSession` (optional)
        If `None`, creates a default session.
    num_partitions: int (optional)
        (Used only when reading HDF format)
        The number of partitions the list of files should be split into for reading.
        If unset (`None`), falls back to the Spark context's default.

    Any other keyword arguments are passe to `read_pythia_parquet`, or ignored if reading HDF.

    Returns
    -------
    PsmDataset
        A :py:class:`wheely.mammoth.dataset.PsmDataset` object containing the parsed PSMs.
    """
    if not spark:
        spark = _SparkSession.builder.getOrCreate()

    file_paths = [str(p) for p in _listify(scored_files)]

    num_hdf = len(
        list(filter(lambda f: f.lower().endswith(".scored"), file_paths))
    )
    if num_hdf == 0:
        return read_pythia_parquet(file_paths, spark=spark, **kwargs)
    elif num_hdf != len(file_paths):
        raise ValueError(
            "Can't read a mix of formats! Only some locations ended in '.psm.scored'"
        )

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


def read_pythia_parquet(
    locations,
    spark: _Optional[_SparkSession] = None,
) -> _PsmDataset:
    """
    Read scored PSMs from Pythia `.psm.scored` files.

    Parameters
    ----------
    locations : str or tuple of str
        Paths or URIs specifying a collection of PSMs in Pythia's `.prq.pythiaDIA` format.
    spark : :py:class:`pyspark.sql.SparkSession` (optional)
        If `None`, creates a default session.

    Returns
    -------
    PsmDataset
        A :py:class:`wheely.mammoth.dataset.PsmDataset` object containing the parsed PSMs.
    """
    if not spark:
        spark = _SparkSession.builder.getOrCreate()

    file_paths = [str(p) for p in _listify(locations)]

    psms_df = (
        spark.read.parquet(*file_paths)
        .withColumn("filename", _input_file_name())
        .withColumn(
            "precursor",
            _concat(_col("peptideWithMods"), _lit("+"), _col("charge")),
        )
        .withColumn("target", ~_col("isDecoy").astype("boolean"))
    )

    _logging.debug("Read dataframe with columns: %s", psms_df.columns)

    return _PsmDataset(
        psms_df,
        target_column="target",
        spectrum_columns=["filename", "scanNumber"],
        score_columns=[
            "charge",
            "cosineSim",
            "discScore",
            "discScoreMax",
            "discScoreMean",
            "discScoreMedian",
            "discScoreMin",
            "discScoreStDev",
            "fractionFound",
            "frameCandidateCount",
            "frameError",
            "frameFStat",
            "frameRankDiscScore",
            "frameRankScore",
            "ionsFound",
            "isotopeFoundCount",
            "klDiv",
            "missedCleavages",
            "monoIsoOffset",
            "ms1CosineSim",
            "mz",
            "mzFound",
            "peptideSize",
            "ppmDiffMs1",
            "rescore",
            "score",
            "scoreMax",
            "scoreMean",
            "scoreMedian",
            "scoreStDev",
        ],
        peptide_column="precursor",
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
    # Optional dependency; only import if we know we need to use it
    import h5py as _h5

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
