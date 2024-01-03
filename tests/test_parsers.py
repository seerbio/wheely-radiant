"""Tests for parsing implementations"""
import logging

import numpy as np
import pyspark.sql

import pytest

from wheely_pythia.scoring import _schemes
from wheely_pythia.parsers import *


@pytest.mark.parametrize(
    "score_cols",
    [
        None,
        *_schemes.keys(),
        *_schemes.values(),
    ],
)
def test_read_pythia_features(spark_session, pythia_features, score_cols):
    """Test that we parse DIA scoring feature (parquet) files correctly"""

    n = 256  # Expected PSM (row) count

    psms = read_pythia_features(
        pythia_features,
        spark_session,
        scoring=score_cols() if callable(score_cols) else score_cols,
    )
    assert isinstance(psms.data, pyspark.sql.DataFrame)
    assert psms.data.count() == n
    assert list(psms.spectrum_columns) == [
        "filename",
        "precursor",
        "scanNumber",
    ]
    assert all(col in psms.spectra.columns for col in psms.spectrum_columns)
    assert psms.protein_column is not None

    assert all(col in psms.data.columns for col in psms.columns)

    assert "isDecoy" in psms.data.columns, "Could not find isDecoy"
    np.testing.assert_array_equal(
        psms.data.select("isDecoy").toPandas().values,
        psms.data.select(~psms.targets).toPandas().values,
    )

    logging.debug(psms.data.dtypes)

    for col, type in psms.data.dtypes:
        if col in psms.score_columns:
            assert type in {
                "double",
                "float",
                "int",
                "bigint",
            }, f"Score column {col} had unexpected datatype!"

    target_df = psms.data.select(psms.targets).toPandas()

    assert target_df.shape == (n, 1)

    ndec = 0  # TODO: test file has no decoys!
    assert target_df[target_df.columns[0]].sum() == n - ndec
    assert (~target_df[target_df.columns[0]]).sum() == ndec


def test_read_pythia_hdf_features(spark_session, pythia_hdf_features):
    """Test that we parse legacy HDF (DDA result) files correctly"""
    psms = read_pythia_features(pythia_hdf_features, spark_session)
    assert isinstance(psms.data, pyspark.sql.DataFrame)
    assert psms.data.count() == 1000
    assert list(psms.spectrum_columns) == ["filename", "scanNumber"]
    assert all(col in psms.spectra.columns for col in psms.spectrum_columns)

    assert "isDecoy" in psms.data.columns, "Could not find isDecoy"
    np.testing.assert_array_equal(
        psms.data.select("isDecoy").toPandas().values,
        psms.data.select(~psms.targets).toPandas().values,
    )

    scores = {
        "cosine_similarity",
        "klDivergence",
        "Score",
        "hyperscore",
        "deltaScore",
        "meanErrorPPM",
        "meanAbsolueErrorPPM",  # typo in Pythia
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
    }
    assert set(psms.score_columns) == scores

    assert psms.scores.toPandas().shape == (1000, len(scores))

    target_df = psms.data.select(psms.targets).toPandas()

    assert target_df.shape == (1000, 1)
    assert target_df[target_df.columns[0]].sum() == 691
    assert (~target_df[target_df.columns[0]]).sum() == 1000 - 691
