"""
`parsers`: module for Pythia results parsing functions
"""
import logging as _logging
from typing import (
    Callable as _Callable,
    Dict as _Dict,
    Iterable as _Iterable,
    Optional as _Optional,
    Union as _Union,
)

import pandas as _pd
from pyspark.sql import (
    Column as _Column,
    SparkSession as _SparkSession,
)
import pyspark.sql.functions as _fns
from pyspark.sql.functions import (
    col as _col,
    lit as _lit,
)
from wheely.mammoth import PsmDataset as _PsmDataset
from wheely.mammoth.utils import listify as _listify
from wheely_pythia import pythia_scores_default

_logger = _logging.getLogger(__name__)


def read_pythia_features(
    location,
    spark: _Optional[_SparkSession] = None,
    num_partitions: _Optional[int] = None,
    **kwargs,
) -> _PsmDataset:
    """
    Read scored PSMs from Pythia `.psm.scored` files.

    Parameters
    ----------
    location : str or tuple of str
        Paths or URIs specifying a collection of PSMs in Pythia's `.prq.pythiaDIA` format, or
        `.scored` (HDF) format (to be deprecated). Note: all file paths must be in the same
        format.
    spark : :py:class:`pyspark.sql.SparkSession` (optional)
        If `None`, creates a default session.
    num_partitions: int (optional)
        (Used only when reading HDF format)
        The number of partitions the list of files should be split into for reading.
        If unset (`None`), falls back to the Spark context's default.

    Any other keyword arguments are passed to `read_pythia_parquet`, or ignored if reading HDF.

    Returns
    -------
    PsmDataset
        A :py:class:`wheely.mammoth.dataset.PsmDataset` object containing the parsed PSMs.
    """
    if not spark:
        spark = _SparkSession.builder.getOrCreate()

    file_paths = [str(p) for p in _listify(location)]

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

    _logger.debug("Read dataframe with columns: %s", psms_df.columns)

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
    location,
    score_columns: _Optional[
        _Union[
            str,
            _Iterable[str],
            _Dict[str, _Column],
            _Callable[[], _Union[str, _Iterable[str], _Dict[str, _Column]]],
        ]
    ] = None,
    spark: _Optional[_SparkSession] = None,
) -> _PsmDataset:
    """
    Read scored PSMs from Pythia `.pythiaDIA` files.

    Parameters
    ----------
    location : str or iterable of str
        Paths or URIs specifying a collection of PSMs in Pythia's `.prq.pythiaDIA` format.
    score_columns : str, list of str, dict of `{name: pyspark.sql.Column}`, or `callable` specifying
                    the `score_columns` of the returned dataset. See also `pythia_scores_default()`
                    and `pythia_scores_svm()` which return collections compatible with this parameter.
                    If a callable, it must accept no arguments and produce a suitable value.
    spark : :py:class:`pyspark.sql.SparkSession` (optional)
        If `None`, creates a default session.

    Returns
    -------
    PsmDataset
        A :py:class:`wheely.mammoth.dataset.PsmDataset` object containing the parsed PSMs.
    """
    if not spark:
        spark = _SparkSession.builder.getOrCreate()

    file_paths = [str(p) for p in _listify(location)]

    psms_df = (
        spark.read.parquet(*file_paths)
        .withColumn("filename", _fns.input_file_name())
        .withColumn(
            "precursor",
            _fns.concat(
                _col("peptideStringWithMods"), _lit("+"), _col("charge")
            ),
        )
        .withColumn("target", ~_col("isDecoy").astype("boolean"))
    )

    _logger.debug("Read dataframe with columns: %s", psms_df.columns)

    if score_columns is None:
        score_columns = pythia_scores_default

    if callable(score_columns):
        score_columns = score_columns()

    if isinstance(score_columns, _Dict):
        psms_df = psms_df.withColumns(score_columns)

        score_columns = score_columns.keys()

    score_columns = _listify(score_columns)

    return _PsmDataset(
        psms_df,
        target_column="target",
        spectrum_columns=["filename", "precursor", "scanNumber"],
        score_columns=score_columns,
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
