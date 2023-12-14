"""
`wheely_pythia.scoring` -- different scoring schemes for use with PythiaDIA
"""
import struct as _struct
from typing import Dict as _Dict, List as _List

from pyspark.sql import Column as _Column, functions as _fns


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
