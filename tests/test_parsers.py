"""Tests for parsing implementations"""

import logging

import numpy as np
import pyspark.sql

import pytest

from wheely.mammoth import PsmDataset
from wheely.mammoth.spectra import SpectraDataset

from wheely_radiant.scoring import _schemes
from wheely_radiant.parsers import *


@pytest.mark.parametrize(
    "score_cols",
    [
        None,
        *_schemes.keys(),
        *_schemes.values(),
    ],
)
@pytest.mark.parametrize(
    "read_spectra",
    [True, False],
)
def test_read_features(
    caplog, spark_session, radiant_features, score_cols, read_spectra
):
    """Test that we parse DIA scoring feature (parquet) files correctly"""
    caplog.set_level(logging.CRITICAL)  # we only want to capture logs later on

    n = 256  # Expected PSM (row) count

    psms = read_radiant_features(
        radiant_features,
        spark_session,
        scoring=score_cols,
        read_spectra=read_spectra,
    )
    assert isinstance(psms, PsmDataset)
    assert isinstance(psms, SpectraDataset) or not read_spectra
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
            *(
                [
                    psms.charge_column,
                    psms.rt_column,
                    psms.mz_column,
                    psms.peaklist_column,
                ]
                if read_spectra
                else []
            ),
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

    if "TotalIntensityLog" in psms.data.columns:
        assert "TotalIntensityLog" not in psms.score_columns

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

    if read_spectra:
        # Check for invalid peaks
        assert hasattr(psms, "peaklists")
        np.testing.assert_array_equal(
            psms.data.select(
                pyspark.sql.functions.size(
                    pyspark.sql.functions.filter(
                        psms.peaklists,
                        lambda pk: (pk.getItem(0) > 0) & (pk.getItem(1) > 0),
                    )
                )
            )
            .toPandas()
            .values,
            psms.data.select(pyspark.sql.functions.size(psms.peaklists))
            .toPandas()
            .values,
        )

    # Regardless of read_spectra, we should be able to "pass through" spectra without a join.
    # This only works for v1+ files, as array-typed ("vec") columns cause complications with
    # downstream modules that can't handle structured datatypes.
    if "discriminateScore" not in psms.data.columns:
        with caplog.at_level(logging.INFO):
            spectra_dset = read_radiant_spectra(psms)
            assert (
                "pass-thr" in caplog.text
            ), "Did not find log message confirming spectra pass-through!"
            assert (
                "Reading Radiant spectra" not in caplog.text
            ), "Found log message confirming spectra are re-read!"

            # Check for invalid peaks
            assert hasattr(spectra_dset, "peaklists")
            np.testing.assert_array_equal(
                spectra_dset.data.select(
                    pyspark.sql.functions.size(
                        pyspark.sql.functions.filter(
                            spectra_dset.peaklists,
                            lambda pk: (pk.getItem(0) > 0)
                            & (pk.getItem(1) > 0),
                        )
                    )
                )
                .toPandas()
                .values,
                spectra_dset.data.select(
                    pyspark.sql.functions.size(spectra_dset.peaklists)
                )
                .toPandas()
                .values,
            )
