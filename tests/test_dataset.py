"""
These are unit tests for the PSM Dataset Class:
"""

import re

import pandas as pd
import pyspark.sql.functions
import pytest

from wheely.mammoth import ConfidenceDataset

from wheely_pythia import read_pythia_features
from wheely_pythia.dataset import PythiaDataset, PythiaSpectraDataset

_dset_types = [PythiaDataset, PythiaSpectraDataset]


@pytest.fixture(
    params=[
        *_dset_types,
        # Test that "quoted" column names work
        *[
            lambda *args, **kwargs: typ(
                *args,
                **{
                    k: (
                        [f"`{c}`" for c in v]
                        if "columns" in k
                        else f"`{v}`" if "column" in k else v
                    )
                    for k, v in kwargs.items()
                },
            )
            for typ in _dset_types
        ],
    ]
)
def dataset_type(request):
    return request.param


@pytest.fixture
def pythia_data(pythia_features):
    dset = read_pythia_features(pythia_features)
    return (
        dset.data,
        {
            k: getattr(dset, k)
            for k in {
                "target_column",
                "spectrum_columns",
                "charge_column",
                "mz_column",
                "rt_column",
                "score_columns",
                "peptide_column",
                "protein_column",
                "protein_delim",
            }
        },
    )


_QUOTED_PATT = re.compile("^`(.*)`$")


def _drop_quotes(maybe_quoted):
    if not isinstance(maybe_quoted, str):
        return list(map(_drop_quotes, maybe_quoted))

    match = _QUOTED_PATT.match(maybe_quoted)
    return match.group(1) if match else maybe_quoted


def test_properties(pythia_data, dataset_type):
    """Check the public properties of the PsmDataset object."""
    pythia_df, cols = pythia_data

    psms = dataset_type(
        psms=pythia_df,
        **cols,
    )

    for k, v in cols.items():
        assert _drop_quotes(getattr(psms, k)) == v, f"Mismatch for {k}"

    pd.testing.assert_frame_equal(
        psms.data.select(psms.targets).toPandas(),
        pythia_df.toPandas().loc[:, ["target"]],
    )

    assert all(c is not None for c in psms.columns)
    assert set(psms.columns) == {
        psms.target_column,
        *psms.spectrum_columns,
        *psms.score_columns,
        psms.charge_column,
        psms.mz_column,
        psms.rt_column,
        psms.peptide_column,
        psms.protein_column,
        *[getattr(psms, k) for k in {"peaklist_column"} if hasattr(psms, k)],
    }


def test_mutate(pythia_df, dataset_type):
    """Check mutating a PsmDataset object."""
    psms = dataset_type(
        psms=pythia_df,
        target_column="target",
        spectrum_columns=["file", "scan"],
        score_columns=["combined p-value", "x"],
        peptide_column="sequence",
        protein_column="protein id",
        protein_delim=",",
    )

    n_rows = 5
    n_targets = 4

    mut = psms.with_data(
        psms.data.limit(n_rows).withColumn(
            "isDecoy", pyspark.sql.functions.col("target").astype("int") == 0
        ),
        target_column="isDecoy",
    )

    assert isinstance(mut, type(psms))

    assert mut.data.count() == n_rows
    # assert mut.target_column == "isDecoy"
    assert (
        mut.data.select(
            pyspark.sql.functions.sum(mut.targets.astype("int"))
        ).collect()[0][0]
        == n_rows - n_targets
    )

    # assert list(mut.spectra.columns) == ["file", "scan"]
    # assert list(mut.scores.columns) == ["combined p-value", "x"]
    # assert mut.peptide_column == "sequence"
    # assert mut.protein_column == "protein id"
    assert mut.protein_delim == ","

    if isinstance(mut, ConfidenceDataset):
        assert mut.qvalue_column == psms.qvalue_column
        assert mut.pi0 == psms.pi0
