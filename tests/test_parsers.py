"""Tests for parsing implementations"""

import logging

import numpy as np
import pyspark.sql

import pytest

from wheely.mammoth import PsmDataset
from wheely.mammoth.spectra import SpectraDataset

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
        scoring=score_cols,
    )
    assert isinstance(psms, PsmDataset)
    assert isinstance(psms, SpectraDataset)
    assert isinstance(psms.data, pyspark.sql.DataFrame)

    # Check for columns missing from .columns (dataset class check)
    assert not [
        col
        for col in [
            *psms.spectrum_columns,
            *psms.score_columns,
            psms.target_column,
            psms.peptide_column,
            psms.charge_column,
            psms.rt_column,
            psms.mz_column,
            psms.peaklist_column,
            psms.protein_column,
        ]
        if col not in psms.columns
    ]

    # Check for columns missing in the dataframe
    assert not [col for col in psms.columns if col not in psms.data.columns]

    assert not [
        col for col in psms.score_columns if col not in psms.scores.columns
    ]
    assert len(set(psms.score_columns)) == len(
        psms.score_columns
    ), "Found duplicated scores!"

    assert psms.data.count() == n
    assert not [
        col for col in psms.spectrum_columns if col not in psms.spectra.columns
    ]
    assert psms.protein_column is not None

    assert not [col for col in psms.columns if col not in psms.data.columns]

    assert any(
        c in psms.data.columns for c in ["isDecoy", "IsDecoy"]
    ), "Could not find decoy col"
    np.testing.assert_array_equal(
        psms.data.select(
            "IsDecoy" if "IsDecoy" in psms.data.columns else "isDecoy"
        )
        .toPandas()
        .values,
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

    # Check for columns missing from .columns (dataset class check)
    assert not [
        col
        for col in [
            *psms.spectrum_columns,
            *psms.score_columns,
            psms.target_column,
            psms.peptide_column,
            psms.protein_column,
        ]
        if col not in psms.columns
    ]

    # Check for columns missing from dataframe
    assert not [col for col in psms.columns if col not in psms.data.columns]

    assert psms.data.count() == 1000
    assert list(psms.spectrum_columns) == ["filename", "scanNumber"]
    assert not [
        col for col in psms.spectrum_columns if col not in psms.spectra.columns
    ]

    assert not [
        col for col in psms.score_columns if col not in psms.scores.columns
    ]
    assert len(set(psms.score_columns)) == len(
        psms.score_columns
    ), "Found duplicated scores!"

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
