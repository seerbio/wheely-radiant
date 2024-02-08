"""
`wheely_pythia.scoring` -- different scoring schemes for use with PythiaDIA
"""

import logging as _logging
import struct as _struct
from typing import (
    Dict as _Dict,
    Iterable as _Iterable,
    List as _List,
)

# Once the min supported version reaches 3.10, the standard library should
# be used like so -> from importlib.metadata import entry_points
from importlib_metadata import entry_points

from pyspark.sql import Column as _Column, functions as _fns

_logger = _logging.getLogger(__name__)


def pythia_scores_default(*columns) -> _List[str]:
    """
    Returns
    -------
    The default set of score columns from Pythia v1.0 and later, excluding the output of its NN
    classifier.
    """

    if "discriminateScore" in columns:
        return pythia_scores_default_v0(*columns)

    # Fall through to here when no arguments are supplied, making v1.0+ the default.
    return pythia_scores_default_v1(*columns)


def pythia_scores_default_v1(*columns) -> _List[str]:
    """
    Returns
    -------
    The default set of score columns from Pythia v1.0 and later, excluding the output of its NN
    classifier.
    """
    pythia_scores = [
        "DiscriminantScore",  # Moved to first, as this is the "primary" score
        "AllignedMaxIndexesCount",
        "AltTargetKeyIdCosineSimSumCharge1_1",
        "AltTargetKeyIdCosineSimSumCharge1_2",
        "AltTargetKeyIdCosineSimSumCharge1_3",
        "AltTargetKeyIdCosineSimSumCharge1_OG",
        "AltTargetKeyIdCosineSimSumCharge2_1",
        "AltTargetKeyIdCosineSimSumCharge2_2",
        "AltTargetKeyIdCosineSimSumCharge2_3",
        "AltTargetKeyIdCosineSimSumCharge2_OG",
        "AltTargetKeyIdCosineSimSumCharge3_1",
        "AltTargetKeyIdCosineSimSumCharge3_2",
        "AltTargetKeyIdCosineSimSumCharge3_3",
        "AltTargetKeyIdCosineSimSumCharge3_OG",
        "AltTargetKeyIdCosineSimSumCharge4_1",
        "AltTargetKeyIdCosineSimSumCharge4_2",
        "AltTargetKeyIdCosineSimSumCharge4_3",
        "AltTargetKeyIdCosineSimSumCharge4_OG",
        "AltTargetKeyIdTimeDeltaCharge1_1",
        "AltTargetKeyIdTimeDeltaCharge1_2",
        "AltTargetKeyIdTimeDeltaCharge1_3",
        "AltTargetKeyIdTimeDeltaCharge2_1",
        "AltTargetKeyIdTimeDeltaCharge2_2",
        "AltTargetKeyIdTimeDeltaCharge2_3",
        "AltTargetKeyIdTimeDeltaCharge3_1",
        "AltTargetKeyIdTimeDeltaCharge3_2",
        "AltTargetKeyIdTimeDeltaCharge3_3",
        "AltTargetKeyIdTimeDeltaCharge4_1",
        "AltTargetKeyIdTimeDeltaCharge4_2",
        "AltTargetKeyIdTimeDeltaCharge4_3"
        "AminoAcidCountA",
        "AminoAcidCountC",
        "AminoAcidCountD",
        "AminoAcidCountE",
        "AminoAcidCountF",
        "AminoAcidCountG",
        "AminoAcidCountH",
        "AminoAcidCountI",
        "AminoAcidCountK",
        "AminoAcidCountL",
        "AminoAcidCountM",
        "AminoAcidCountN",
        "AminoAcidCountP",
        "AminoAcidCountQ",
        "AminoAcidCountR",
        "AminoAcidCountS",
        "AminoAcidCountT",
        "AminoAcidCountV",
        "AminoAcidCountW",
        "AminoAcidCountY",
        "AminoAcidCountB",
        "AminoAcidCountJ",
        "AminoAcidCountO",
        "AminoAcidCountU",
        "AminoAcidCountX",
        "AminoAcidCountZ",
        "ChargeNorm",
        # 'ClassifierScore',                 # From classifier
        "ColumnApexIndexRatiosToAnchor1",
        "ColumnApexIndexRatiosToAnchor10",
        "ColumnApexIndexRatiosToAnchor11",
        "ColumnApexIndexRatiosToAnchor12",
        "ColumnApexIndexRatiosToAnchor2",
        "ColumnApexIndexRatiosToAnchor3",
        "ColumnApexIndexRatiosToAnchor4",
        "ColumnApexIndexRatiosToAnchor5",
        "ColumnApexIndexRatiosToAnchor6",
        "ColumnApexIndexRatiosToAnchor7",
        "ColumnApexIndexRatiosToAnchor8",
        "ColumnApexIndexRatiosToAnchor9",
        "CosineSim100MS1",
        "CosineSim100MS1Iso1",
        "CosineSim100MS1Iso2",
        "CosineSim100MS1PreMono",
        "CosineSim20MS1",
        "CosineSim45MS1",
        "CosineSimShadowsToAnchor1",
        "CosineSimShadowsToAnchor10",
        "CosineSimShadowsToAnchor11",
        "CosineSimShadowsToAnchor12",
        "CosineSimShadowsToAnchor2",
        "CosineSimShadowsToAnchor3",
        "CosineSimShadowsToAnchor4",
        "CosineSimShadowsToAnchor5",
        "CosineSimShadowsToAnchor6",
        "CosineSimShadowsToAnchor7",
        "CosineSimShadowsToAnchor8",
        "CosineSimShadowsToAnchor9",
        "CosineSimSpectrum",
        "CosineSimSpectrumCubed",
        "CosineSimSum100",
        "CosineSimSum20",
        "CosineSimSum45",
        "CosineSimSumBottom6",
        "CosineSimSumTop6",
        "CosineSimToAnchor1",
        "CosineSimToAnchor10",
        "CosineSimToAnchor11",
        "CosineSimToAnchor12",
        "CosineSimToAnchor2",
        "CosineSimToAnchor3",
        "CosineSimToAnchor4",
        "CosineSimToAnchor5",
        "CosineSimToAnchor6",
        "CosineSimToAnchor7",
        "CosineSimToAnchor8",
        "CosineSimToAnchor9",
        # 'DecoyRatio',           # From classifier
        "IRTPredicted",
        "IntensityFoundMax1",
        "IntensityFoundMax10",
        "IntensityFoundMax11",
        "IntensityFoundMax12",
        "IntensityFoundMax2",
        "IntensityFoundMax3",
        "IntensityFoundMax4",
        "IntensityFoundMax5",
        "IntensityFoundMax6",
        "IntensityFoundMax7",
        "IntensityFoundMax8",
        "IntensityFoundMax9",
        "KlDivSpectrum",
        "KlDivSpectrumCubeRoot",
        "Mass",
        "MzAccuracy1",
        "MzAccuracy10",
        "MzAccuracy11",
        "MzAccuracy12",
        "MzAccuracy2",
        "MzAccuracy3",
        "MzAccuracy4",
        "MzAccuracy5",
        "MzAccuracy6",
        "MzAccuracy7",
        "MzAccuracy8",
        "MzAccuracy9",
        "MzFoundMean1",
        "MzFoundMean10",
        "MzFoundMean11",
        "MzFoundMean12",
        "MzFoundMean2",
        "MzFoundMean3",
        "MzFoundMean4",
        "MzFoundMean5",
        "MzFoundMean6",
        "MzFoundMean7",
        "MzFoundMean8",
        "MzFoundMean9",
        "MzFoundStDev1",
        "MzFoundStDev10",
        "MzFoundStDev11",
        "MzFoundStDev12",
        "MzFoundStDev2",
        "MzFoundStDev3",
        "MzFoundStDev4",
        "MzFoundStDev5",
        "MzFoundStDev6",
        "MzFoundStDev7",
        "MzFoundStDev8",
        "MzFoundStDev9",
        "MzNorm",
        "MzPeakLengthsNorm1",
        "MzPeakLengthsNorm10",
        "MzPeakLengthsNorm11",
        "MzPeakLengthsNorm12",
        "MzPeakLengthsNorm2",
        "MzPeakLengthsNorm3",
        "MzPeakLengthsNorm4",
        "MzPeakLengthsNorm5",
        "MzPeakLengthsNorm6",
        "MzPeakLengthsNorm7",
        "MzPeakLengthsNorm8",
        "MzPeakLengthsNorm9",
        "MzSearched1",
        "MzSearched10",
        "MzSearched11",
        "MzSearched12",
        "MzSearched2",
        "MzSearched3",
        "MzSearched4",
        "MzSearched5",
        "MzSearched6",
        "MzSearched7",
        "MzSearched8",
        "MzSearched9",
        "PeakShapeRatio1",
        "PeakShapeRatio2",
        "PeakShapeRatio3",
        "PeptideLengthNorm",
        # 'PeptideStringWithMods',
        # 'ProteinGroup',
        # 'QValue',              # From classifier
        "ScanIonCount",
        #'ScanNumber',
        "ScanTime",
        "ScanTimeDelta",
        "ScanTimePd",
        "ScanTimePredicted",
        "ScanTimeRange",
        "ShadowsCosineSimSum",
        "ShadowsIntensityRatio1",
        "ShadowsIntensityRatio10",
        "ShadowsIntensityRatio11",
        "ShadowsIntensityRatio12",
        "ShadowsIntensityRatio2",
        "ShadowsIntensityRatio3",
        "ShadowsIntensityRatio4",
        "ShadowsIntensityRatio5",
        "ShadowsIntensityRatio6",
        "ShadowsIntensityRatio7",
        "ShadowsIntensityRatio8",
        "ShadowsIntensityRatio9",
        # 'TargetKey',
        "TheoFragmentCount",
        "TheoIntensity1",
        "TheoIntensity10",
        "TheoIntensity11",
        "TheoIntensity12",
        "TheoIntensity2",
        "TheoIntensity3",
        "TheoIntensity4",
        "TheoIntensity5",
        "TheoIntensity6",
        "TheoIntensity7",
        "TheoIntensity8",
        "TheoIntensity9",
        "TopBottomRatio",
        "TopBottomRatioNorm",
        "TotalIntensityLog",
    ]

    _missing = set(pythia_scores) - set(columns)
    if len(_missing) == len(pythia_scores):
        raise ValueError(
            f"Could not find any of the columns {pythia_scores} in {columns}"
        )
    elif len(_missing) > 0:
        _logger.warning(
            "The following scores are missing and will be ignored: %s",
            _missing,
        )

    # Preserve ordering of scores
    pythia_scores = [c for c in pythia_scores if c in columns]

    return pythia_scores


def pythia_scores_default_v0(*columns) -> _List[str]:
    """
    Returns
    -------
    The default set of score columns from Pythia (before v1.0), excluding the output of its NN
    classifier.
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


def pythia_score_classifier(*columns) -> str:
    """
    Returns
    -------
    The name of Pythia's NN classifier score column.
    """

    if "discriminateScore" in columns:
        return pythia_score_classifier_v0(*columns)

    # Fall through to here when no arguments are supplied, making v1.0+ the default.
    return pythia_score_classifier_v1(*columns)


def pythia_score_classifier_v1(*columns) -> str:
    """
    Returns
    -------
    The name of Pythia's NN classifier score column (v1.0 and later).
    """
    return "ClassifierScore"


def pythia_score_classifier_v0(*columns) -> str:
    """
    Returns
    -------
    The name of Pythia's NN classifier score column (before v1.0).
    """
    return "classifierScore"


def pythia_scores_svm(*columns) -> _Dict[str, _Column]:
    """
    Create a set of scores particularly suited to applying SVM rescoring to PythiaDIA results.

    Returns
    -------
    A dict mapping column name to a PySpark column, representing the computation of individual scoring features.
    """

    if "discriminateScore" in columns:
        return pythia_scores_svm_v0(*columns)

    # Fall through to here when no arguments are supplied, making v1.0+ the default.
    return pythia_scores_svm_v1(*columns)


def pythia_scores_svm_v1(*columns) -> _Dict[str, _Column]:
    """
    Create a set of scores particularly suited to applying SVM rescoring to PythiaDIA v1.0 and later results.

    Returns
    -------
    A dict mapping column name to a PySpark column, representing the computation of individual scoring features.
    """
    # Take a list of all known scores; comment out those that aren't directly usable
    pythia_scores = [
        "DiscriminantScore",  # Moved to first, as this is the "primary" score
        "AllignedMaxIndexesCount",
        "AltTargetKeyIdCosineSimSumCharge1_1",
        "AltTargetKeyIdCosineSimSumCharge1_2",
        "AltTargetKeyIdCosineSimSumCharge1_3",
        "AltTargetKeyIdCosineSimSumCharge1_OG",
        "AltTargetKeyIdCosineSimSumCharge2_1",
        "AltTargetKeyIdCosineSimSumCharge2_2",
        "AltTargetKeyIdCosineSimSumCharge2_3",
        "AltTargetKeyIdCosineSimSumCharge2_OG",
        "AltTargetKeyIdCosineSimSumCharge3_1",
        "AltTargetKeyIdCosineSimSumCharge3_2",
        "AltTargetKeyIdCosineSimSumCharge3_3",
        "AltTargetKeyIdCosineSimSumCharge3_OG",
        "AltTargetKeyIdCosineSimSumCharge4_1",
        "AltTargetKeyIdCosineSimSumCharge4_2",
        "AltTargetKeyIdCosineSimSumCharge4_3",
        "AltTargetKeyIdCosineSimSumCharge4_OG",
        "AltTargetKeyIdTimeDeltaCharge1_1",
        "AltTargetKeyIdTimeDeltaCharge1_2",
        "AltTargetKeyIdTimeDeltaCharge1_3",
        "AltTargetKeyIdTimeDeltaCharge2_1",
        "AltTargetKeyIdTimeDeltaCharge2_2",
        "AltTargetKeyIdTimeDeltaCharge2_3",
        "AltTargetKeyIdTimeDeltaCharge3_1",
        "AltTargetKeyIdTimeDeltaCharge3_2",
        "AltTargetKeyIdTimeDeltaCharge3_3",
        "AltTargetKeyIdTimeDeltaCharge4_1",
        "AltTargetKeyIdTimeDeltaCharge4_2",
        "AltTargetKeyIdTimeDeltaCharge4_3"
        "AminoAcidCountA",
        "AminoAcidCountC",
        "AminoAcidCountD",
        "AminoAcidCountE",
        "AminoAcidCountF",
        "AminoAcidCountG",
        "AminoAcidCountH",
        "AminoAcidCountI",
        "AminoAcidCountK",
        "AminoAcidCountL",
        "AminoAcidCountM",
        "AminoAcidCountN",
        "AminoAcidCountP",
        "AminoAcidCountQ",
        "AminoAcidCountR",
        "AminoAcidCountS",
        "AminoAcidCountT",
        "AminoAcidCountV",
        "AminoAcidCountW",
        "AminoAcidCountY",
        "AminoAcidCountB",
        "AminoAcidCountJ",
        "AminoAcidCountO",
        "AminoAcidCountU",
        "AminoAcidCountX",
        "AminoAcidCountZ",
        # 'ChargeNorm',                      # Reencoded below
        # 'ClassifierScore',                 # From classifier
        "ColumnApexIndexRatiosToAnchor1",
        "ColumnApexIndexRatiosToAnchor10",
        "ColumnApexIndexRatiosToAnchor11",
        "ColumnApexIndexRatiosToAnchor12",
        "ColumnApexIndexRatiosToAnchor2",
        "ColumnApexIndexRatiosToAnchor3",
        "ColumnApexIndexRatiosToAnchor4",
        "ColumnApexIndexRatiosToAnchor5",
        "ColumnApexIndexRatiosToAnchor6",
        "ColumnApexIndexRatiosToAnchor7",
        "ColumnApexIndexRatiosToAnchor8",
        "ColumnApexIndexRatiosToAnchor9",
        "CosineSim100MS1",
        "CosineSim100MS1Iso1",
        "CosineSim100MS1Iso2",
        "CosineSim100MS1PreMono",
        "CosineSim20MS1",
        "CosineSim45MS1",
        "CosineSimShadowsToAnchor1",
        "CosineSimShadowsToAnchor10",
        "CosineSimShadowsToAnchor11",
        "CosineSimShadowsToAnchor12",
        "CosineSimShadowsToAnchor2",
        "CosineSimShadowsToAnchor3",
        "CosineSimShadowsToAnchor4",
        "CosineSimShadowsToAnchor5",
        "CosineSimShadowsToAnchor6",
        "CosineSimShadowsToAnchor7",
        "CosineSimShadowsToAnchor8",
        "CosineSimShadowsToAnchor9",
        "CosineSimSpectrum",
        "CosineSimSpectrumCubed",
        "CosineSimSum100",
        "CosineSimSum20",
        "CosineSimSum45",
        "CosineSimSumBottom6",
        "CosineSimSumTop6",
        "CosineSimToAnchor1",
        "CosineSimToAnchor10",
        "CosineSimToAnchor11",
        "CosineSimToAnchor12",
        "CosineSimToAnchor2",
        "CosineSimToAnchor3",
        "CosineSimToAnchor4",
        "CosineSimToAnchor5",
        "CosineSimToAnchor6",
        "CosineSimToAnchor7",
        "CosineSimToAnchor8",
        "CosineSimToAnchor9",
        # 'DecoyRatio',           # From classifier
        "IRTPredicted",  # Keep this as a peptide correction factor
        # 'IntensityFoundMax1',   # All reencoded below
        # 'IntensityFoundMax10',
        # 'IntensityFoundMax11',
        # 'IntensityFoundMax12',
        # 'IntensityFoundMax2',
        # 'IntensityFoundMax3',
        # 'IntensityFoundMax4',
        # 'IntensityFoundMax5',
        # 'IntensityFoundMax6',
        # 'IntensityFoundMax7',
        # 'IntensityFoundMax8',
        # 'IntensityFoundMax9',
        # 'IsDecoy',
        "KlDivSpectrum",
        "KlDivSpectrumCubeRoot",
        "Mass",  # Keep this as a peptide correction factor
        "MzAccuracy1",
        "MzAccuracy10",
        "MzAccuracy11",
        "MzAccuracy12",
        "MzAccuracy2",
        "MzAccuracy3",
        "MzAccuracy4",
        "MzAccuracy5",
        "MzAccuracy6",
        "MzAccuracy7",
        "MzAccuracy8",
        "MzAccuracy9",
        # 'MzFoundMean1',        # All reencoded below
        # 'MzFoundMean10',
        # 'MzFoundMean11',
        # 'MzFoundMean12',
        # 'MzFoundMean2',
        # 'MzFoundMean3',
        # 'MzFoundMean4',
        # 'MzFoundMean5',
        # 'MzFoundMean6',
        # 'MzFoundMean7',
        # 'MzFoundMean8',
        # 'MzFoundMean9',
        "MzFoundStDev1",
        "MzFoundStDev10",
        "MzFoundStDev11",
        "MzFoundStDev12",
        "MzFoundStDev2",
        "MzFoundStDev3",
        "MzFoundStDev4",
        "MzFoundStDev5",
        "MzFoundStDev6",
        "MzFoundStDev7",
        "MzFoundStDev8",
        "MzFoundStDev9",
        "MzNorm",  # Keep this as a peptide correction factor
        "MzPeakLengthsNorm1",
        "MzPeakLengthsNorm10",
        "MzPeakLengthsNorm11",
        "MzPeakLengthsNorm12",
        "MzPeakLengthsNorm2",
        "MzPeakLengthsNorm3",
        "MzPeakLengthsNorm4",
        "MzPeakLengthsNorm5",
        "MzPeakLengthsNorm6",
        "MzPeakLengthsNorm7",
        "MzPeakLengthsNorm8",
        "MzPeakLengthsNorm9",
        # 'MzSearched1',         # All reencoded below
        # 'MzSearched10',
        # 'MzSearched11',
        # 'MzSearched12',
        # 'MzSearched2',
        # 'MzSearched3',
        # 'MzSearched4',
        # 'MzSearched5',
        # 'MzSearched6',
        # 'MzSearched7',
        # 'MzSearched8',
        # 'MzSearched9',
        "PeakShapeRatio1",
        "PeakShapeRatio2",
        "PeakShapeRatio3",
        "PeptideLengthNorm",  # Keep this as a peptide correction factor
        # 'PeptideStringWithMods',
        # 'ProteinGroup',
        # 'QValue',              # From classifier
        "ScanIonCount",
        # 'ScanNumber',
        # 'ScanTime',
        # 'ScanTimeDelta',       # Reencoded below
        "ScanTimePd",
        # 'ScanTimePredicted',
        "ScanTimeRange",
        "ShadowsCosineSimSum",
        "ShadowsIntensityRatio1",
        "ShadowsIntensityRatio10",
        "ShadowsIntensityRatio11",
        "ShadowsIntensityRatio12",
        "ShadowsIntensityRatio2",
        "ShadowsIntensityRatio3",
        "ShadowsIntensityRatio4",
        "ShadowsIntensityRatio5",
        "ShadowsIntensityRatio6",
        "ShadowsIntensityRatio7",
        "ShadowsIntensityRatio8",
        "ShadowsIntensityRatio9",
        # 'TargetKey',
        "TheoFragmentCount",
        # 'TheoIntensity1',          # All reencoded below
        # 'TheoIntensity10',
        # 'TheoIntensity11',
        # 'TheoIntensity12',
        # 'TheoIntensity2',
        # 'TheoIntensity3',
        # 'TheoIntensity4',
        # 'TheoIntensity5',
        # 'TheoIntensity6',
        # 'TheoIntensity7',
        # 'TheoIntensity8',
        # 'TheoIntensity9',
        "TopBottomRatio",
        "TopBottomRatioNorm",
        "TotalIntensityLog"
    ]

    _missing = set(pythia_scores) - set(columns)
    if len(_missing) == len(pythia_scores):
        raise ValueError(
            f"Could not find any of the columns {pythia_scores} in {columns}"
        )
    elif len(_missing) > 0:
        _logger.warning(
            "The following scores are missing and will be ignored: %s",
            _missing,
        )

    # Preserve ordering of scores
    pythia_scores = [c for c in pythia_scores if c in columns]

    # Now we construct additional scores from some
    # columns that we don't use directly

    addl_scores = {
        "absDeltaScanTime": _fns.abs(_fns.col("ScanTimeDelta")),
        # 1-hot encoding for charge
        **{
            f"charge{i}": _fns.when(
                _fns.col("charge") == i, _fns.lit(1.0)
            ).otherwise(0.0)
            for i in [1, 2, 3, 4]
        },
    }

    # Unconditionally use 12 scores from each array; this
    # will not work if the user chooses another setting.
    n_vec_scores = 12

    # Convert frag m/z information into individual mass deltas
    _max_mz_delta = 0.05
    for i in range(n_vec_scores):
        idx = i + 1

        _foundmz = _fns.col(f"MzFoundMean{idx}")
        _theomz = _fns.col(f"MzSearched{idx}")

        addl_scores[f"absDeltaMz_{i}"] = _fns.when(
            # Handle peaks that weren't found
            (_foundmz == 0.0) | _fns.isnull(_foundmz),
            _fns.lit(_max_mz_delta),
        ).otherwise(
            _fns.least(_fns.abs(_foundmz - _theomz), _fns.lit(_max_mz_delta))
        )

    # Convert frag intensity information into individual absolute log ratios
    _max_log_inten_ratio = 4.0
    _foundints = [
        _fns.col(f"IntensityFoundMax{i + 1}") for i in range(n_vec_scores)
    ]
    _foundint = _fns.array(*_foundints)
    _totfoundint = _fns.aggregate(_foundint, _fns.lit(0.0), lambda a, b: a + b)
    _theoints = [
        _fns.col(f"TheoIntensity{i + 1}") for i in range(n_vec_scores)
    ]
    _theoint = _fns.array(*_theoints)
    _tottheoint = _fns.aggregate(_theoint, _fns.lit(0.0), lambda a, b: a + b)
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


def pythia_scores_svm_v0(*columns) -> _Dict[str, _Column]:
    """
    Create a set of scores particularly suited to applying SVM rescoring to PythiaDIA (before v1.0) results.

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

    # Unconditionally use 12 scores from each array; this
    # will not work if the user chooses another setting.
    n_vec_scores = 12

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


_schemes = {
    "default": pythia_scores_default,
    "nn": pythia_score_classifier,
    "svm": pythia_scores_svm,
}
_plugins = None


def register_scheme(name, scheme, clobber=False):
    assert isinstance(scheme, dict) or isinstance(scheme, _Iterable)

    if name in _schemes:
        if not clobber:
            raise RuntimeError(
                f"Backend {name} is already registered and `clobber` is False"
            )

        _logger.warning(
            f"Replacing already-registered scoring scheme {name} with {scheme}"
        )

    _schemes[name] = scheme


def _get_plugins():
    """Return a dict of all installed Plugins as {name: scheme}."""

    plugins = entry_points(group="wheely_pythia.scoring.plugins")

    pluginmap = {}
    for plugin in plugins:
        pluginmap[plugin.name] = plugin

    for k, v in pluginmap.items():
        _logger.debug(f"loading {k}")
        pluginmap[k] = v.load()

    return pluginmap


def get_scheme(name):
    """Fetch a scheme with the given name."""
    global _schemes, _plugins
    try:
        return _schemes[name]
    except KeyError as e:
        if _plugins is None:
            _plugins = _get_plugins()

        if _plugins is not None and name in _plugins:
            return _plugins[name]

        all_keys = set(_schemes.keys())
        if _plugins is not None:
            all_keys = all_keys.union(_plugins.keys())

        raise KeyError(
            f"No such scheme: {name}. Only {str(all_keys)} are supported"
        ) from e
