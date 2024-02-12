"""
`parsers`: module for Pythia results parsing functions
"""

import logging as _logging
import struct as _struct
from typing import (
    Callable as _Callable,
    Dict as _Dict,
    Iterable as _Iterable,
    Optional as _Optional,
    Union as _Union,
    cast as _cast,
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
from wheely.mammoth.spectra import SpectraDataset as _SpectraDataset
from wheely.mammoth.spectra.utils import (
    lists_to_peaklist as _lists_to_peaklist,
)

from .dataset import (
    PythiaDataset as _PythiaDataset,
    PythiaSpectraDataset as _PythiaSpectraDataset,
)
from .scoring import get_scheme as _get_scoring_scheme

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
    else:
        return read_pythia_hdf(file_paths, spark=spark, **kwargs)


def read_pythia_hdf(location, spark, num_partitions=None):
    file_paths = location

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
    scoring: _Optional[
        _Union[
            str,
            _Iterable[str],
            _Dict[str, _Column],
            _Callable[[], _Union[str, _Iterable[str], _Dict[str, _Column]]],
        ]
    ] = None,
    read_spectra: bool = False,
    spark: _Optional[_SparkSession] = None,
) -> _PsmDataset:
    """
    Read scored PSMs from Pythia `.pythiaDIA` files.

    Parameters
    ----------
    location : str or iterable of str
        Paths or URIs specifying a collection of PSMs in Pythia's `.prq.pythiaDIA` format.
    scoring : str, list of str, dict of `{name: pyspark.sql.Column}`, or `callable` specifying the
              `score_columns` of the returned dataset. See also `pythia_scores_default()` and
              `pythia_scores_svm()` which return collections compatible with this parameter. If a
              str, it will be treated as the name of a registered scoring scheme (see
              `wheely_pythia.scoring`), or if no such scheme exists, the name of a single column.
              An error will occur if no matches are found in the scheme registry or in the specified
              files. If a callable, it must accept a set of column names as positional arguments and
              return a suitable value.
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

    psms_df = spark.read.parquet(*file_paths)

    _logger.debug("Read dataframe with columns: %s", psms_df.columns)

    addl_cols = {
        "filename": _fns.input_file_name(),
        "target": ~_col(
            "IsDecoy" if "IsDecoy" in psms_df.columns else "isDecoy"
        ).astype("boolean"),
    }

    if read_spectra:
        addl_cols["peaklist"] = parse_peaklist(psms_df.columns)

    charge_col: str = "charge"
    if charge_col not in psms_df.columns:
        if "Charge" in psms_df.columns:
            charge_col = "Charge"

    addl_cols["mz"] = (
        _col("Mass") + _fns.lit(1.007276) * _col(charge_col)
    ) / _col(charge_col)

    psms_df = psms_df.withColumns(addl_cols)

    if scoring is None:
        scoring = "default"

    if isinstance(scoring, str):
        # Check if this is a defined scoring scheme
        try:
            scoring = _get_scoring_scheme(scoring)
        except KeyError:
            scoring = [scoring]

    if callable(scoring):
        scoring = scoring(*psms_df.columns)

    _logger.debug("Got scoring scheme: %s", scoring)

    if isinstance(scoring, _Dict):
        psms_df = psms_df.withColumns(scoring)

        scoring = scoring.keys()
    else:
        # Ensure we have an iterable, not a single string
        scoring = _listify(scoring)

    _logger.debug("Proceeding with score_columns: %s", scoring)

    assert all(
        s in psms_df.columns for s in scoring
    ), f"Missing scoring columns! Could not find: {list(set(scoring) - set(psms_df.columns))} in {list(psms_df.columns)}"

    # Drop any vector-typed columns (really, binary blobs). Any access to these columns can be
    # performed by passing appropriate `pyspark.sql.Column`s (in a dict) to the `scoring` parameter.
    # For now, we simply recognize these columns by their name suffix. # TODO: check schema instead
    psms_df = psms_df.drop(
        *(c for c in psms_df.columns if c not in scoring and c.endswith("Vec"))
    )

    col_semantics = _get_col_semantics(psms_df.columns, charge_col=charge_col)

    if read_spectra:
        return _PythiaSpectraDataset(
            psms_df,
            target_column="target",
            score_columns=scoring,
            **col_semantics,
            protein_delim=";",
        )
    else:
        return _PythiaDataset(
            psms_df,
            target_column="target",
            score_columns=scoring,
            **col_semantics,
            protein_delim=";",
        )


def parse_peaklist(columns, n_peaks=12):
    if "mzFoundMeanVec" in columns:
        # _to_float_array = _fns.udf(lambda b: _struct.unpack("<" + "f" * int(len(b) / 4), b), returnType="Array<float>")
        _to_double_array = _fns.udf(
            lambda b: _struct.unpack("<" + "d" * int(len(b) / 8), b),
            returnType="Array<double>",
        )

        return _lists_to_peaklist(
            _to_double_array("mzFoundMeanVec"),
            _to_double_array("intensityFoundMaxVec"),
        )
    else:
        return _lists_to_peaklist(
            _fns.array(*[f"MzFoundMean{i + 1}" for i in range(n_peaks)]).alias(
                "MzFoundMeanVec"
            ),
            _fns.array(
                *[f"IntensityFoundMax{i + 1}" for i in range(n_peaks)]
            ).alias("IntensityFoundMaxVec"),
        )


def _get_col_semantics(columns, charge_col="charge"):
    if "discriminateScore" not in columns:
        return dict(
            spectrum_columns=[
                "filename",
                "PeptideStringWithMods",
                charge_col,
                "ScanNumber",
            ],
            rt_column="ScanTime",
            peptide_column="PeptideStringWithMods",
            protein_column="ProteinGroup",
        )
    else:
        # Support legacy files with deprecated column names
        return dict(
            spectrum_columns=[
                "filename",
                "peptideStringWithMods",
                charge_col,
                "scanNumber",
            ],
            rt_column="scanTime",
            peptide_column="peptideStringWithMods",
            protein_column="proteinGroup",
        )


def read_pythia_spectra(
    psms: _PsmDataset,
    **kwargs,
) -> _SpectraDataset:
    # Try to short-circuit by reannotating known columns
    if any(c in psms.data.columns for c in ["mzFoundMeanVec", "MzFoundMean1"]):
        pass_thru_dset = _PythiaSpectraDataset(
            psms.data.withColumn(
                "peaklist", parse_peaklist(psms.data.columns)
            ),
            target_column=psms.target_column,
            score_columns=psms.score_columns,
            protein_delim=psms.protein_delim,
            **_get_col_semantics(
                psms.data.columns,
                charge_col=(
                    "charge" if "charge" in psms.data.columns else "Charge"
                ),
            ),
        )
        if all(c in psms.data.columns for c in pass_thru_dset.columns):
            _logger.info("Using pass-through spectra from Pythia")
            return pass_thru_dset

    assert (
        "filename" in psms.data.columns
    ), "Did not find `filename` column for reading Pythia spectra!"

    location = (
        psms.data.select(_col("filename").alias("__location"))
        .dropDuplicates()
        .toPandas()["__location"]
        .values
    )

    _logger.info(
        f"Reading Pythia spectra from {len(location)} location(s): {location[:3]}{'…' if len(location) > 3 else ''}"
    )

    return _cast(
        _SpectraDataset,
        read_pythia_features(
            location,
            spark=psms.data.sparkSession,
            read_spectra=True,
            **kwargs,
        ),
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
