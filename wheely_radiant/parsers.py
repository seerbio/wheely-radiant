"""
`parsers`: module for results parsing functions
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
from wheely.mammoth.semantics import (
    NORMALIZED_RT_IN_SECONDS as _NORMALIZED_RT_IN_SECONDS,
    RT_IN_SECONDS as _RT_IN_SECONDS,
    RT_START_IN_SECONDS as _RT_START_IN_SECONDS,
    RT_STOP_IN_SECONDS as _RT_STOP_IN_SECONDS,
    SCAN_NUMBER as _SCAN_NUMBER,
    THEORETICAL_MONO_MASS as _THEORETICAL_MONO_MASS,
    THEORETICAL_PRECURSOR_MZ as _THEORETICAL_PRECURSOR_MZ,
)
from wheely.mammoth.utils import listify as _listify
from wheely.mammoth.spectra import SpectraDataset as _SpectraDataset
from wheely.mammoth.spectra.utils import (
    lists_to_peaklist as _lists_to_peaklist,
)

from .dataset import (
    RadiantDataset as _RadiantDataset,
    RadiantIonMobilityDataset as _RadiantIonMobilityDataset,
    RadiantIonMobilitySpectraDataset as _RadiantIonMobilitySpectraDataset,
    RadiantSpectraDataset as _RadiantSpectraDataset,
)
from .scoring import get_scheme as _get_scoring_scheme

_logger = _logging.getLogger(__name__)


def read_radiant_features(
    location,
    spark: _Optional[_SparkSession] = None,
    *_,
    scoring: _Optional[
        _Union[
            str,
            _Iterable[str],
            _Dict[str, _Column],
            _Callable[[], _Union[str, _Iterable[str], _Dict[str, _Column]]],
        ]
    ] = None,
    read_spectra: bool = False,
    use_irt: bool = True,
    use_iim: _Optional[bool] = None,
) -> _PsmDataset:
    """
    Read scored PSMs from `.radiantDIA` files.

    Parameters
    ----------
    location : str or iterable of str
        Paths or URIs specifying a collection of PSMs in `.radiantDIA` format.
    spark : :py:class:`pyspark.sql.SparkSession` (optional)
        If `None`, creates a default session.
    scoring : str, list of str, dict of ``{name: pyspark.sql.Column}``, or ``callable`` specifying the
              ``score_columns`` of the returned dataset. See also ``radiant_scores_default()`` and
              ``radiant_scores_svm()`` which return collections compatible with this parameter. If a
              ``str``, it will be treated as the name of a registered scoring scheme (see
              :py:mod:`wheely_radiant.scoring`), or if no such scheme exists, the name of a single column.
              An error will occur if no matches are found in the scheme registry or in the specified
              files. If a callable, it must accept a set of column names as positional arguments and
              return a suitable value.
    read_spectra : bool
        If ``True``, include spectral information in the returned dataset and return a
        :py:class:`wheely.mammoth.spectra.SpectraDataset`.
    use_irt : bool
        If ``True``, the returned dataset will use empirical IRT values for the retention time column, ensuring that RTs
        across multiple files are well-aligned to a consistent RT space. Otherwise, the measured retention time in each
        file will be used directly.
    use_iim : bool, optional
        If ``True``, require and use ``IIMEmpirical`` for ion mobility. If
        ``False``, require and use ``IonMobilityFound``. If ``None``, use
        ``IonMobilityFound`` when present and otherwise omit ion mobility.

    Returns
    -------
    PsmDataset
        A :py:class:`wheely.mammoth.dataset.PsmDataset` object containing the parsed PSMs.
    """
    if _:
        raise TypeError("Unexpected positional arguments!")

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
        else:
            raise ValueError("Charge or charge column not found")

    psms_df = psms_df.withColumn(charge_col, _col(charge_col).cast("integer"))

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

    if "TotalIntensityLog" in scoring:
        scoring = [
            c if c != "TotalIntensityLog" else "__TotalIntensityLog"
            for c in scoring
        ]
        psms_df = psms_df.withColumn(
            "__TotalIntensityLog", _col("TotalIntensityLog")
        )

    # Drop any vector-typed columns (really, binary blobs). Any access to these columns can be
    # performed by passing appropriate `pyspark.sql.Column`s (in a dict) to the `scoring` parameter.
    # For now, we simply recognize these columns by their name suffix. # TODO: check schema instead
    psms_df = psms_df.drop(
        *(c for c in psms_df.columns if c not in scoring and c.endswith("Vec"))
    )

    col_semantics, semantics = _get_col_semantics(
        psms_df.columns,
        charge_col=charge_col,
        use_irt=use_irt,
        use_iim=use_iim,
    )

    semantics = dict(
        semantics or {},
        mz=_THEORETICAL_PRECURSOR_MZ,
    )

    _logger.debug("Using column semantics: %s", col_semantics)
    _logger.debug("Additional semantic tags: %s", semantics)

    if read_spectra:
        dataset_type = (
            _RadiantIonMobilitySpectraDataset
            if "ion_mobility_column" in col_semantics
            else _RadiantSpectraDataset
        )
        return dataset_type(
            psms_df,
            target_column="target",
            score_columns=scoring,
            **col_semantics,
            protein_delim=";",
            semantics=semantics,
        )
    else:
        dataset_type = (
            _RadiantIonMobilityDataset
            if "ion_mobility_column" in col_semantics
            else _RadiantDataset
        )
        return dataset_type(
            psms_df,
            target_column="target",
            score_columns=scoring,
            **col_semantics,
            protein_delim=";",
            semantics=semantics,
        )


def parse_peaklist(columns, n_peaks=12):
    if "mzSearchedVec" in columns:
        # _to_float_array = _fns.udf(lambda b: _struct.unpack("<" + "f" * int(len(b) / 4), b), returnType="Array<float>")
        _to_double_array = _fns.udf(
            lambda b: _struct.unpack("<" + "d" * int(len(b) / 8), b),
            returnType="Array<double>",
        )

        result = _lists_to_peaklist(
            _to_double_array("mzSearchedVec"),
            _to_double_array("intensityFoundMaxVec"),
        )
    else:
        result = _lists_to_peaklist(
            _fns.array(*[f"MzSearched{i + 1}" for i in range(n_peaks)]).alias(
                "MzSearchedVec"
            ),
            _fns.array(
                *[f"IntensityFoundMax{i + 1}" for i in range(n_peaks)]
            ).alias("IntensityFoundMaxVec"),
        )

    return _fns.filter(result, _is_valid_peak)


def _is_valid_peak(pk_col: _Column) -> _Column:
    return (pk_col.getItem(0) > 0) & (pk_col.getItem(1) > 0)


def _get_col_semantics(columns, charge_col=None, use_irt=True, use_iim=None):
    if "discriminateScore" not in columns:
        col_semantics = dict(
            spectrum_columns=[
                "filename",
                "PeptideStringWithMods",
                "Charge",
                "ScanNumber",
            ],
            charge_column=charge_col or "Charge",
            rt_column=(
                "IRTEmpirical"
                if "IRTEmpirical" in columns and use_irt
                else "ScanTime"
            ),
            peptide_column="PeptideStringWithMods",
            protein_column="ProteinGroup",
        )
        semantics = {
            "ScanNumber": _SCAN_NUMBER,
            "ScanTime": _RT_IN_SECONDS,
            "ScanTimeStart": _RT_START_IN_SECONDS,
            "ScanTimeEnd": _RT_STOP_IN_SECONDS,
            "IRTEmpirical": _NORMALIZED_RT_IN_SECONDS,
            "Mass": _THEORETICAL_MONO_MASS,
        }

        ion_mobility_column = _get_ion_mobility_column(columns, use_iim)
        if ion_mobility_column is not None:
            col_semantics["ion_mobility_column"] = ion_mobility_column

        return col_semantics, semantics
    else:
        # Support legacy files with deprecated column names
        col_semantics = dict(
            spectrum_columns=[
                "filename",
                "peptideStringWithMods",
                "charge",
                "scanNumber",
            ],
            charge_column=charge_col or "charge",
            rt_column="scanTime",
            peptide_column="peptideStringWithMods",
            protein_column="proteinGroup",
        )
        ion_mobility_column = _get_ion_mobility_column(columns, use_iim)
        if ion_mobility_column is not None:
            col_semantics["ion_mobility_column"] = ion_mobility_column
        return col_semantics, None


def _get_ion_mobility_column(columns, use_iim=None):
    requested = "IIMEmpirical" if use_iim else "IonMobilityFound"
    if requested in columns:
        return requested
    if use_iim is None:
        return None
    raise ValueError(
        f"Requested ion mobility column {requested!r} was not found in "
        f"Radiant results columns: {list(columns)}"
    )


def read_radiant_spectra(
    psms: _PsmDataset,
    use_irt: bool = True,
    use_iim: _Optional[bool] = None,
    **kwargs,
) -> _SpectraDataset:
    # Try to short-circuit by reannotating known columns
    if any(c in psms.data.columns for c in ["mzFoundMeanVec", "MzFoundMean1"]):
        col_semantics, semantics = _get_col_semantics(
            psms.data.columns, use_irt=use_irt, use_iim=use_iim
        )

        _logger.debug("Using column semantics: %s", col_semantics)
        _logger.debug("Additional semantic tags: %s", semantics)

        dataset_type = (
            _RadiantIonMobilitySpectraDataset
            if "ion_mobility_column" in col_semantics
            else _RadiantSpectraDataset
        )
        pass_thru_dset = dataset_type(
            psms.data.withColumn(
                "peaklist", parse_peaklist(psms.data.columns)
            ),
            target_column=psms.target_column,
            score_columns=psms.score_columns,
            protein_delim=psms.protein_delim,
            **col_semantics,
            semantics=semantics,
        )
        if all(
            c in pass_thru_dset.data.columns for c in pass_thru_dset.columns
        ):
            _logger.info("Using pass-through spectra from Radiant")
            return pass_thru_dset

    assert (
        "filename" in psms.data.columns
    ), "Did not find `filename` column for reading Radiant spectra!"

    location = (
        psms.data.select(_col("filename").alias("__location"))
        .dropDuplicates()
        .toPandas()["__location"]
        .values
    )

    _logger.info(
        f"Reading Radiant spectra from {len(location)} location(s): {location[:3]}{'…' if len(location) > 3 else ''}"
    )

    return _cast(
        _SpectraDataset,
        read_radiant_features(
            location,
            spark=psms.data.sparkSession,
            read_spectra=True,
            use_irt=use_irt,
            use_iim=use_iim,
            **kwargs,
        ),
    )
