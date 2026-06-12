"""
`wheely_radiant.quant` -- peptide quantification backends for Radiant DIA
"""

from typing import (
    Any as _Any,
    Callable as _Callable,
    Mapping as _Mapping,
    Optional as _Optional,
    Union as _Union,
)

from pyspark.sql import Column as _Column
from wheely.mammoth import (
    ConfidenceDataset as _ConfidenceDataset,
    PsmDataset as _PsmDataset,
    PsmIntensityConfidenceDataset as _PsmIntensityConfidenceDataset,
    PsmIntensityDataset as _PsmIntensityDataset,
)
from wheely.mammoth.semantics import XIC_AREA as _XIC_AREA


def quantify_radiant(
    dset: _PsmDataset,
    sample_column: str,
    intensity_column: str = "radiant_intensity",
    normalization: _Optional[
        _Union[str, _Callable, _Mapping[str, _Any]]
    ] = None,
):
    """
    Radiant peptide quantification backend for Fulcrum.

    This backend will compute a Radiant-specific intensity column from
    existing dataset columns, then return the corresponding wheely-mammoth
    intensity dataset.
    """
    if normalization is not None:
        raise NotImplementedError(
            "Normalization is not implemented for Radiant peptide quant."
        )

    quant_data = dset.data.withColumn(
        intensity_column,
        _radiant_intensity_column(dset),
    )

    kwargs = dict(
        sample_column=sample_column,
        intensity_column=intensity_column,
        peptide_column=dset.peptide_column,
        charge_column=dset.charge_column,
        spectrum_columns=dset.spectrum_columns,
        score_columns=dset.score_columns,
        target_column=dset.target_column,
        protein_column=dset.protein_column,
        protein_delim=dset.protein_delim,
        semantics=dict(
            {
                intensity_column: _XIC_AREA,
            },
            **dset.semantics,
        ),
    )

    if isinstance(dset, _ConfidenceDataset):
        kwargs["qvalue_column"] = dset.qvalue_column
        kwargs["errprob_column"] = dset.errprob_column
        kwargs["pi0"] = dset.pi0
        return _PsmIntensityConfidenceDataset(quant_data, **kwargs)

    return _PsmIntensityDataset(quant_data, **kwargs)


def _radiant_intensity_column(dset: _PsmDataset) -> _Column:
    raise NotImplementedError(
        "Radiant peptide intensity calculation is not implemented yet."
    )
