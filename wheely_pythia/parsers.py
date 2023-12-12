"""
`parsers`: module for Pythia results parsing functions
"""
import logging as _logging
import struct as _struct
from typing import (
    Dict as _Dict,
    Iterable as _Iterable,
    List as _List,
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
    location,
    score_columns: _Optional[
        _Union[str, _Iterable[str], _Dict[str, _Column]]
    ] = None,
    spark: _Optional[_SparkSession] = None,
) -> _PsmDataset:
    """
    Read scored PSMs from Pythia `.pythiaDIA` files.

    Parameters
    ----------
    location : str or iterable of str
        Paths or URIs specifying a collection of PSMs in Pythia's `.prq.pythiaDIA` format.
    score_columns : str, list of str, or dict of `{name: pyspark.sql.Column}` specifying the
                    `score_columns` of the returned dataset. See also `pythia_scores_default()` and
                    `pythia_scores_svm()` which return collections compatible with this parameter.
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

    _logging.debug("Read dataframe with columns: %s", psms_df.columns)

    if score_columns is None:
        score_columns = pythia_scores_default()

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


def pythia_scores_default() -> _List[str]:
    """
    Returns
    -------
    The default set of score columns from Pythia, excluding the output of its NN classifier.
    """
    return [
        "discriminateScore",  # Moved to first, as this is the "primary" score
        "b2Corr",
        "b2b3CosineSimSum",
        "b3Corr",
        "charge",
        # "classifierScore",                # From classifier
        "cosineSim100MS1",
        "cosineSim100MS1Iso1",
        "cosineSim100MS1Iso2",
        "cosineSim20MS1",
        "cosineSim45MS1",
        "cosineSimSpectrum",
        "cosineSimSum100",
        "cosineSimSum20",
        "cosineSimSum45",
        # "decoyRatio",                     # From classifier
        "iRTPredicted",
        "klDivSpectrum",
        "klDivSum",
        "mass",
        "matrixError",
        "matrixPVal",
        "matrixWeight",
        "peakShapeRatio1",
        "peakShapeRatio2",
        "peakShapeRatio3",
        # "qValue",                         # From classifier
        "scanIonCount",
        "scanNumberCandidateCount",
        "scanTime",
        "scanTimePredicted",
        "theoFragmentCount",
    ]


def pythia_score_classifier() -> str:
    """
    Returns
    -------
    The name of Pythia's NN classifier score column.
    """
    return "classifierScore"


def pythia_scores_svm(n_vec_scores=12) -> _Dict[str, _Column]:
    """
    Create a set of scores particularly suited to applying SVM rescoring to PythiaDIA results.

    Parameters
    ----------
    n_vec_scores The number of scores to decode from each vector-typed score column

    Returns
    -------
    A dict mapping column name to a PySpark column, representing the computation of individual scoring features.
    """
    # Take a list of all known scores; comment out those that aren't directly usable
    pythia_scores = [
        "discriminateScore",  # Moved to first, as this is the "primary" score
        "b2Corr",
        "b2b3CosineSimSum",
        "b3Corr",
        # 'charge',                         # Reencoded below
        # 'classifierScore',                # From classifier
        "cosineSim100MS1",
        "cosineSim100MS1Iso1",
        "cosineSim100MS1Iso2",
        "cosineSim20MS1",
        "cosineSim45MS1",
        # "cosineSimShadowsToAnchorVec",
        "cosineSimSpectrum",
        "cosineSimSum100",
        "cosineSimSum20",
        "cosineSimSum45",
        # 'cosineSimToAnchorVec',
        # 'decoyRatio',                     # From classifier
        "iRTPredicted",
        # 'intensityFoundMaxVec',
        # 'isDecoy',
        "klDivSpectrum",
        "klDivSum",
        "mass",
        "matrixError",
        "matrixPVal",
        "matrixWeight",
        # 'mzFoundMeanVec',
        # 'mzFoundStDevVec',
        # 'mzSearchedVec',
        "peakShapeRatio1",
        "peakShapeRatio2",
        "peakShapeRatio3",
        # 'peptideStringWithMods',
        # 'proteinGroup',
        # 'qValue',                         # From classifier
        "scanIonCount",
        # "scanNumber",
        "scanNumberCandidateCount",
        # 'scanTime',                       # Reencoded below
        # 'scanTimePredicted',              # Reencoded below
        # 'targetKey',
        "theoFragmentCount",
        # 'theoIntensityVec',
    ]

    # Now we construct additional scores from some
    # that we don't use directly

    addl_scores = {
        "absDeltaScanTime": _fns.abs(
            _fns.col("scanTime") - _fns.col("scanTimePredicted")
        ),
        # 1-hot encoding for charge
        **{
            f"charge{i}": _fns.when(
                _fns.col("charge") == i, _fns.lit(1.0)
            ).otherwise(0.0)
            for i in [1, 2, 3, 4]
        },
    }

    # Now add the various scores that must be parsed from arrays;
    # dict value is default value
    arr_scores = {
        "cosineSimToAnchorVec": 0.0,
        # "intensityFoundMaxVec": 0.0,        # Reencoded below
        # "mzFoundMeanVec": 0.0,              # Reencoded below
        "mzFoundStDevVec": 0.1,
        # "mzSearchedVec": 0.0,               # Reencoded below
        # "theoIntensityVec": 0.0,            # Reencoded below
    }

    to_double_array = _fns.udf(
        lambda b: _struct.unpack("<" + "d" * int(len(b) / 8), b),
        returnType="Array<double>",
    )

    for score, default in arr_scores.items():
        arr_col = to_double_array(score)

        for i in range(n_vec_scores):
            val = _fns.coalesce(arr_col.getItem(i), _fns.lit(default))
            if default > 0.0:
                val = _fns.least(val, _fns.lit(default))
            addl_scores[f"{score}_{i}"] = val

    # Convert this pair of arrays to an individual mass deltas
    _foundmz = to_double_array("mzFoundMeanVec")
    _theomz = to_double_array("mzSearchedVec")
    _max_mz_delta = 0.05
    for i in range(n_vec_scores):
        addl_scores[f"absDeltaMz_{i}"] = _fns.when(
            # Handle peaks that weren't found
            (_foundmz.getItem(i) == 0.0) | _fns.isnull(_foundmz.getItem(i)),
            _fns.lit(_max_mz_delta),
        ).otherwise(
            _fns.least(
                _fns.abs(_foundmz.getItem(i) - _theomz.getItem(i)),
                _fns.lit(_max_mz_delta),
            )
        )

    # Convert this pair of arrays to an individual absolute log ratios
    _foundint = to_double_array("intensityFoundMaxVec")
    _totfoundint = _fns.aggregate(_foundint, _fns.lit(0.0), lambda a, b: a + b)
    _theoint = to_double_array("theoIntensityVec")
    _tottheoint = _fns.aggregate(_theoint, _fns.lit(0.0), lambda a, b: a + b)
    _max_log_inten_ratio = 4.0
    for i in range(n_vec_scores):
        addl_scores[f"absLogNormIntenRatio_{i}"] = _fns.least(
            _fns.lit(10.0),
            _fns.abs(
                _fns.log(
                    10.0,
                    (_foundint.getItem(i) / _totfoundint)
                    / (_theoint.getItem(i) / _tottheoint),
                )
            ),
        )

    return {
        **{c: _fns.col(c) for c in pythia_scores},
        **addl_scores,
    }


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
