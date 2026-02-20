"""
These are unit tests for the PSM Dataset Class:
"""

import re

import pandas as pd
import pyspark.sql.functions
import pytest

from wheely.mammoth.semantics import BasicSemantic

from wheely_radiant import read_radiant_features
from wheely_radiant.dataset import RadiantDataset, RadiantSpectraDataset

_dset_types = [RadiantDataset, RadiantSpectraDataset]


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
        # Test that non-default peaklist column name is supported
        lambda psms, **kwargs: RadiantSpectraDataset(
            psms.withColumnRenamed(
                kwargs.get("peaklist_column", "peaklist"), "__custom_peaklist"
            ).drop(kwargs.get("peaklist_column", "peaklist")),
            **dict(
                kwargs,
                peaklist_column="__custom_peaklist",
            ),
        ),
        # Semantics variants
        lambda psms, **kwargs: RadiantDataset(
            psms,
            **kwargs,
            semantics={"test_col": BasicSemantic("Test semantic")},
        ),
        lambda psms, **kwargs: RadiantSpectraDataset(
            psms,
            **kwargs,
            semantics={"test_col": BasicSemantic("Test semantic")},
        ),
    ]
)
def dataset_type(request):
    return request.param


@pytest.fixture
def radiant_data(radiant_features):
    # Must pass read_spectra=True just in case we then try to instantiate a SpectraDataset.
    # Note that the returned class here is irrelevant, we just want the DataFrame and col. names.
    dset = read_radiant_features(radiant_features, read_spectra=True)

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


def test_properties(radiant_data, dataset_type):
    """Check the public properties of the PsmDataset object."""
    radiant_df, cols = radiant_data

    psms = dataset_type(
        psms=radiant_df,
        **cols,
    )

    for k, v in cols.items():
        assert _drop_quotes(getattr(psms, k)) == v, f"Mismatch for {k}"

    pd.testing.assert_frame_equal(
        psms.data.select(psms.targets).toPandas(),
        radiant_df.toPandas().loc[:, ["target"]],
    )

    # Check that optional columns' properties are correctly handled
    for ca, a in {
        "charge_column": "charges",
        "qvalue_column": "qvalues",
        "errprob_column": "errprobs",
        "intensity_column": "intensities",
    }.items():
        if not hasattr(psms, ca):
            continue

        try:
            assert hasattr(psms, a), f"Had {ca} but no {a}!"

            assert (getattr(psms, ca) is None) == (getattr(psms, a) is None)
        except Exception as e:
            raise AssertionError(f"Error testing {ca}/{a}") from e

    # Check that semantics are properly initialized
    assert hasattr(
        psms, "semantics"
    ), "Dataset should have 'semantics' attribute"
    assert isinstance(psms.semantics, dict), "semantics should be a dict"

    # For datasets with custom semantics in fixture:
    if "test_col" in psms.semantics:
        assert psms.semantics["test_col"] is not None
        # Test get_semantics method
        assert psms.get_semantics("test_col") == psms.semantics["test_col"]

    # Check that the dataset `columns` property is correct
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

    for c in psms.columns:
        assert (c in psms.data.columns) or (
            _drop_quotes(c) in psms.data.columns
        )


def test_mutate(radiant_data, dataset_type):
    """Check mutating a PsmDataset object."""
    radiant_df, cols = radiant_data

    psms = dataset_type(
        psms=radiant_df,
        **cols,
    )

    n_rows = 5
    n_targets = 5

    mut = psms.with_data(
        psms.data.limit(n_rows).withColumn(
            "isDecoy", pyspark.sql.functions.col("target").astype("int") == 0
        ),
        target_column="isDecoy",
        semantics={"isDecoy": BasicSemantic("decoy flag")},
    )

    assert isinstance(mut, type(psms))

    if hasattr(psms, "peaklist_column"):
        assert psms.peaklist_column == mut.peaklist_column

    assert mut.data.count() == n_rows
    assert (
        mut.data.select(
            pyspark.sql.functions.sum(mut.targets.astype("int"))
        ).collect()[0][0]
        == n_rows - n_targets
    )

    # Check that semantics are preserved through with_data()
    assert hasattr(
        mut, "semantics"
    ), "Mutated dataset should have 'semantics' attribute"
    assert (
        mut.get_by_semantics(BasicSemantic("decoy flag")) == "isDecoy"
    ), "Mutated dataset did not have mutated semantic"

    for k, v in cols.items():
        if k == "target_column":
            continue
        assert _drop_quotes(getattr(mut, k)) == v, f"Mismatch for {k}"

    # Check that the dataset `columns` property is correct
    assert all(c is not None for c in mut.columns)
    assert set(mut.columns) == {
        mut.target_column,
        *mut.spectrum_columns,
        *mut.score_columns,
        mut.charge_column,
        mut.mz_column,
        mut.rt_column,
        mut.peptide_column,
        mut.protein_column,
        *[getattr(mut, k) for k in {"peaklist_column"} if hasattr(mut, k)],
    }

    for c in psms.columns:
        assert (c in psms.data.columns) or (
            _drop_quotes(c) in psms.data.columns
        )
