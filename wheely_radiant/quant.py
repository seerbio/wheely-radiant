"""
`wheely_radiant.quant` -- peptide quantification backends for Radiant DIA
"""

import logging as _logging
from functools import reduce as _reduce
from typing import (
    Any as _Any,
    Callable as _Callable,
    Iterable as _Iterable,
    Mapping as _Mapping,
    Optional as _Optional,
    Sequence as _Sequence,
    Union as _Union,
)

from pyspark.sql import Column as _Column
from pyspark.sql import functions as _fns
from pyspark.sql import Window as _Window
from wheely.mammoth import (
    ConfidenceDataset as _ConfidenceDataset,
    PsmDataset as _PsmDataset,
    PsmIntensityConfidenceDataset as _PsmIntensityConfidenceDataset,
    PsmIntensityDataset as _PsmIntensityDataset,
)
from wheely.mammoth.semantics import (
    NORMALIZED_XIC_AREA as _NORMALIZED_XIC_AREA,
    XIC_AREA as _XIC_AREA,
)

_logger = _logging.getLogger(__name__)

_STANDARD_PEPTIDE_COLUMN = "PeptideStringWithMods"
_STANDARD_CHARGE_COLUMN = "Charge"
_TARGET_KEY_COLUMN = "TargetKey"
_RAW_INTENSITY_COLUMN = "TotalIntensityRaw"

_FRAGMENT_MZ_COLUMN = "__radiant_fragment_mz"
_FRAGMENT_SCORE_COLUMN = "__radiant_fragment_score"
_FRAGMENT_INTENSITY_COLUMN = "__radiant_fragment_intensity"
_TRANSITION_SCORE_COLUMN = "__radiant_transition_score"
_TRANSITION_RANK_COLUMN = "__radiant_transition_rank"

_REFINED_MZS_COLUMN = "radiant_refined_transition_mzs"
_NUM_REFINED_TRANSITIONS_COLUMN = "radiant_num_refined_transitions"
_FRAGMENTS_FOUND_COLUMN = "radiant_fragments_found"


def quantify_radiant(
    dset: _PsmDataset,
    sample_column: str,
    intensity_column: str = "radiant_intensity",
    normalization: _Optional[
        _Union[str, _Callable, _Mapping[str, _Any]]
    ] = None,
    qvalue_threshold: _Optional[float] = 0.01,
    fragment_score_threshold: float = 0.8,
    min_transitions: int = 1,
    max_transitions: int = 3,
    fallback_to_raw: bool = False,
    include_target_key: bool = True,
    num_fragments: int = 12,
):
    """
    Radiant peptide quantification backend for Fulcrum.

    This backend computes a refined precursor intensity from the fragment-level
    quantities already present in a Radiant result dataset. It is intended for
    use as a ``fulcrum.quant.peptide`` plugin.

    The calculation has two conceptual stages. First, high-quality transitions
    are learned at the precursor level from confident PSMs. Transitions are
    scored by summing fragment similarity values above ``fragment_score_threshold``
    across the confident rows, then ranked within each precursor by transition
    score, mean fragment intensity, and fragment m/z (to break ties).

    Second, the selected transitions for each precursor are joined back to
    every PSM in the input dataset. The output intensity is the sum of
    ``IntensityFoundMax{i}`` values whose corresponding ``MzSearched{i}``
    appears in the selected transition m/z set. Rows without sufficient refined
    fragments receive a ``null`` intensity, unless ``fallback_to_raw`` is
    ``True``, in which case the sum of all fragment intensities
    (``TotalIntensityRaw``) is used instead.

    Parameters
    ----------
    dset : PsmDataset
        Input Radiant PSM dataset. In normal use this should be a
        :class:`wheely.mammoth.ConfidenceDataset`, because transition learning
        defaults to filtering by q-value. The dataset is expected to use
        ``PeptideStringWithMods`` as its peptide column and ``Charge`` as its
        charge column; if the dataset annotations differ, this function logs a
        warning and uses the annotated columns.
    sample_column : str
        Name of the column identifying sample/run membership for the returned
        intensity dataset.
    intensity_column : str, optional
        Name of the refined intensity column to add. If an input column with
        this name already exists, the refined intensity replaces it in the
        returned dataframe.
    normalization : str, callable, mapping, or None, optional
        Optional normalization to apply after refined intensities are computed.
        A callable is invoked directly. A string, or a mapping with a
        ``backend`` entry, is resolved through Fulcrum's normalization registry
        using a local import only when normalization is requested. After
        normalization, the active intensity column is annotated as
        ``NORMALIZED_XIC_AREA``; otherwise it is annotated as ``XIC_AREA``.
    qvalue_threshold : float or None, optional
        Confidence threshold used when learning transitions. When not ``None``,
        ``dset`` must be a ``ConfidenceDataset`` and only rows with
        ``dset.qvalues <= qvalue_threshold`` contribute to transition
        statistics. Set ``None`` to learn transitions from all rows.
    fragment_score_threshold : float, optional
        Minimum ``CosineSimToAnchor{i}`` value required for a fragment
        observation to contribute to transition score and detection count.
    min_transitions : int, optional
        Minimum number of selected transitions required for a precursor to be
        quantifiable. Note: when ``fallback_to_raw`` is ``True``, precursors with
        fewer than ``min_transitions`` fragments meeting the fragment correlation
        threshold will still be quantified.
    max_transitions : int, optional
        Maximum number of top-ranked transitions retained per precursor.
    fallback_to_raw : bool, optional
        If ``True``, use the sum of all fragment intensities for precursors without
        sufficient transitions available after refinement. This ensures that all
        identifications receive a quantity, but may result in lower-quality
        quantification for precursors without sufficient transitions meeting the
        ``fragment_score_threshold``. Note that for precursors with sufficient refined
        fragments, some rows may have zero intensity values (if the refined fragment
        intensities are all zero) or null intensity values (if all refined fragments
        are missing from that PSM's reported peaks, which should not happen if the same
        library is used for all Radiant searches).
    include_target_key : bool, optional
        Include Radiant ``TargetKey`` in the precursor key. This is the default
        because the same peptide/charge can be observed in different isolation
        windows. Set to ``False`` to refine transitions only by peptide and charge.
    num_fragments : int, optional
        Number of Radiant fragment columns to inspect. Radiant full reports are
        expected to provide ``MzSearched{i}``, ``CosineSimToAnchor{i}``, and
        ``IntensityFoundMax{i}`` for every ``i`` from 1 through this value.
        Default: 12; this should likely not be changed.

    Returns
    -------
    PsmIntensityDataset or PsmIntensityConfidenceDataset
        A dataset backed by the input dataframe plus the refined intensity
        column and diagnostic columns. Confidence metadata is preserved when
        ``dset`` is a ``ConfidenceDataset``.
    """
    _validate_parameters(
        min_transitions=min_transitions,
        max_transitions=max_transitions,
        num_fragments=num_fragments,
    )

    precursor_keys = _get_precursor_key_columns(
        dset,
        include_target_key=include_target_key,
    )

    _logger.debug(
        "Using Radiant precursor keys %s with qvalue_threshold=%s, "
        "fragment_score_threshold=%s, min_transitions=%d, "
        "max_transitions=%d, num_fragments=%d",
        precursor_keys,
        qvalue_threshold,
        fragment_score_threshold,
        min_transitions,
        max_transitions,
        num_fragments,
    )

    _validate_required_columns(
        dset,
        sample_column=sample_column,
        precursor_keys=precursor_keys,
        qvalue_threshold=qvalue_threshold,
        num_fragments=num_fragments,
        fallback_to_raw=fallback_to_raw,
    )

    refined_precursors = _select_refined_transitions(
        _compute_transition_statistics(
            dset,
            precursor_keys=precursor_keys,
            qvalue_threshold=qvalue_threshold,
            fragment_score_threshold=fragment_score_threshold,
            num_fragments=num_fragments,
        ),
        precursor_keys=precursor_keys,
        min_transitions=min_transitions,
        max_transitions=max_transitions,
    )

    quant_data = _add_refined_intensity_column(
        dset,
        refined_precursors,
        precursor_keys=precursor_keys,
        intensity_column=intensity_column,
        num_fragments=num_fragments,
        fallback_to_raw=fallback_to_raw,
    )

    quantified = _wrap_intensity_dataset(
        dset,
        quant_data=quant_data,
        sample_column=sample_column,
        intensity_column=intensity_column,
    )

    return _apply_normalization(quantified, normalization)


def _validate_parameters(
    min_transitions: int,
    max_transitions: int,
    num_fragments: int,
) -> None:
    if num_fragments <= 0:
        raise ValueError("num_fragments must be positive")

    if min_transitions <= 0:
        raise ValueError("min_transitions must be positive")

    if max_transitions < min_transitions:
        raise ValueError(
            "max_transitions must be greater than or equal to "
            "min_transitions"
        )


def _get_precursor_key_columns(
    dset: _PsmDataset,
    include_target_key: bool,
) -> _Sequence[str]:
    peptide_column = dset.peptide_column
    charge_column = dset.charge_column

    if _unquote_column(peptide_column) != _STANDARD_PEPTIDE_COLUMN:
        _logger.warning(
            "Expected Radiant peptide column %r but dataset uses %r; using "
            "dataset-annotated peptide column",
            _STANDARD_PEPTIDE_COLUMN,
            peptide_column,
        )

    if _unquote_column(charge_column) != _STANDARD_CHARGE_COLUMN:
        _logger.warning(
            "Expected Radiant charge column %r but dataset uses %r; using "
            "dataset-annotated charge column",
            _STANDARD_CHARGE_COLUMN,
            charge_column,
        )

    precursor_keys = [peptide_column, charge_column]
    if include_target_key:
        precursor_keys.append(_TARGET_KEY_COLUMN)

    return precursor_keys


def _validate_required_columns(
    dset: _PsmDataset,
    sample_column: str,
    precursor_keys: _Sequence[str],
    qvalue_threshold: _Optional[float],
    num_fragments: int,
    fallback_to_raw: bool,
) -> None:
    required = {
        sample_column,
        dset.target_column,
        *precursor_keys,
        *(
            column_name
            for idx in range(1, num_fragments + 1)
            for column_name in _fragment_column_names(idx)
        ),
    }

    if qvalue_threshold is not None:
        if not isinstance(dset, _ConfidenceDataset):
            raise ValueError(
                "qvalue_threshold requires a ConfidenceDataset; pass "
                "qvalue_threshold=None to learn transitions without q-values"
            )
        required.add(dset.qvalue_column)

    if fallback_to_raw:
        required.add(_RAW_INTENSITY_COLUMN)

    available = set(dset.data.columns)
    missing = sorted(
        column_name
        for column_name in map(_unquote_column, required)
        if column_name not in available
    )
    if missing:
        raise ValueError(
            "Radiant peptide quantification requires missing column(s): "
            + ", ".join(missing)
        )


def _fragment_column_names(idx: int):
    return (
        f"MzSearched{idx}",
        f"CosineSimToAnchor{idx}",
        f"IntensityFoundMax{idx}",
    )


def _compute_transition_statistics(
    dset: _PsmDataset,
    precursor_keys: _Sequence[str],
    qvalue_threshold: _Optional[float],
    fragment_score_threshold: float,
    num_fragments: int,
):
    exploded = _explode_transitions(
        dset,
        precursor_keys=precursor_keys,
        qvalue_threshold=qvalue_threshold,
        num_fragments=num_fragments,
    )

    group_columns = [
        *(_column_ref(key) for key in precursor_keys),
        _FRAGMENT_MZ_COLUMN,
    ]
    transition_score = _fns.sum(
        _fns.when(
            _fns.col(_FRAGMENT_SCORE_COLUMN) > fragment_score_threshold,
            _fns.col(_FRAGMENT_SCORE_COLUMN),
        ).otherwise(_fns.lit(0.0))
    )

    return (
        exploded.groupBy(*group_columns)
        .agg(
            _fns.sum(
                (
                    _fns.col(_FRAGMENT_SCORE_COLUMN) > fragment_score_threshold
                ).cast("integer")
            ).alias("times_found"),
            transition_score.alias(_TRANSITION_SCORE_COLUMN),
            _fns.mean(_FRAGMENT_INTENSITY_COLUMN).alias("MeanIntensity"),
        )
        .filter(_fns.col("times_found") > 0)
        .withColumn(
            _TRANSITION_RANK_COLUMN,
            _fns.row_number().over(
                _Window.partitionBy(
                    *(_column_ref(key) for key in precursor_keys)
                ).orderBy(
                    _fns.col(_TRANSITION_SCORE_COLUMN).desc(),
                    _fns.col("MeanIntensity").desc(),
                    _fns.col(_FRAGMENT_MZ_COLUMN).asc(),
                )
            ),
        )
    )


def _explode_transitions(
    dset: _PsmDataset,
    precursor_keys: _Sequence[str],
    qvalue_threshold: _Optional[float],
    num_fragments: int,
):
    transition_filter = None
    if qvalue_threshold is not None:
        transition_filter = dset.qvalues <= _fns.lit(qvalue_threshold)

    data = dset.data
    if transition_filter is not None:
        data = data.filter(transition_filter)

    return (
        data.select(
            *(_column_ref(key) for key in precursor_keys),
            _fns.explode(
                _fns.array(
                    *(
                        _fns.struct(
                            _column_ref(f"MzSearched{idx}")
                            .cast("double")
                            .alias(_FRAGMENT_MZ_COLUMN),
                            _column_ref(f"CosineSimToAnchor{idx}")
                            .cast("double")
                            .alias(_FRAGMENT_SCORE_COLUMN),
                            _column_ref(f"IntensityFoundMax{idx}")
                            .cast("double")
                            .alias(_FRAGMENT_INTENSITY_COLUMN),
                        )
                        for idx in range(1, num_fragments + 1)
                    )
                )
            ).alias("__radiant_transition"),
        )
        .select(
            *(_column_ref(key) for key in precursor_keys),
            _fns.col("__radiant_transition")
            .getField(_FRAGMENT_MZ_COLUMN)
            .alias(_FRAGMENT_MZ_COLUMN),
            _fns.col("__radiant_transition")
            .getField(_FRAGMENT_SCORE_COLUMN)
            .alias(_FRAGMENT_SCORE_COLUMN),
            _fns.col("__radiant_transition")
            .getField(_FRAGMENT_INTENSITY_COLUMN)
            .alias(_FRAGMENT_INTENSITY_COLUMN),
        )
        .filter(_fns.col(_FRAGMENT_MZ_COLUMN) > 0)
    )


def _select_refined_transitions(
    transition_statistics,
    precursor_keys: _Sequence[str],
    min_transitions: int,
    max_transitions: int,
):
    return (
        transition_statistics.filter(
            _fns.col(_TRANSITION_RANK_COLUMN) <= max_transitions
        )
        .groupBy(*(_column_ref(key) for key in precursor_keys))
        .agg(_fns.collect_list(_FRAGMENT_MZ_COLUMN).alias(_REFINED_MZS_COLUMN))
        .withColumn(
            _NUM_REFINED_TRANSITIONS_COLUMN,
            _fns.size(_fns.col(_REFINED_MZS_COLUMN)),
        )
        .filter(_fns.col(_NUM_REFINED_TRANSITIONS_COLUMN) >= min_transitions)
    )


def _add_refined_intensity_column(
    dset: _PsmDataset,
    refined_precursors,
    precursor_keys: _Sequence[str],
    intensity_column: str,
    num_fragments: int,
    fallback_to_raw: bool,
):
    joined = dset.data.alias("dataset").join(
        refined_precursors.alias("refined"),
        on=_join_condition("dataset", "refined", precursor_keys),
        how="left",
    )

    selected_mzs = _qualified_column_ref("refined", _REFINED_MZS_COLUMN)
    fragment_matches = [
        _fns.coalesce(
            _fns.array_contains(
                selected_mzs,
                _column_ref(f"MzSearched{idx}", qualifier="dataset").cast(
                    "double"
                ),
            ).cast("integer"),
            _fns.lit(0),
        )
        for idx in range(1, num_fragments + 1)
    ]
    fragments_found = _sum_array(fragment_matches).cast("integer")

    intensity_terms = [
        _fns.when(
            _fns.array_contains(
                selected_mzs,
                _column_ref(f"MzSearched{idx}", qualifier="dataset").cast(
                    "double"
                ),
            ),
            _fns.coalesce(
                _column_ref(
                    f"IntensityFoundMax{idx}", qualifier="dataset"
                ).cast("double"),
                _fns.lit(0.0),
            ),
        ).otherwise(_fns.lit(0.0))
        for idx in range(1, num_fragments + 1)
    ]
    refined_intensity = _fns.when(
        fragments_found > 0,
        _sum_array(intensity_terms),
    ).otherwise(
        # No fragments match, but we have refined transitions -- give null
        # even when fallback_to_raw is True, because we need to only use
        # the refined transitions -- we must ensure that quantification is
        # consistent across all PSMs for the precursor.
        _fns.when(
            _fns.array_size(selected_mzs) > 0,
            _fns.lit(None),
        )
        .otherwise(
            _fns.col(_RAW_INTENSITY_COLUMN)
            if fallback_to_raw
            else _fns.lit(None)
        )
        .cast("double")
    )

    output_column_names = {
        _unquote_column(intensity_column),
        _FRAGMENTS_FOUND_COLUMN,
        _NUM_REFINED_TRANSITIONS_COLUMN,
        _REFINED_MZS_COLUMN,
    }
    output_columns = [
        refined_intensity.alias(_unquote_column(intensity_column)),
        fragments_found.alias(_FRAGMENTS_FOUND_COLUMN),
        _qualified_column_ref(
            "refined", _NUM_REFINED_TRANSITIONS_COLUMN
        ).alias(_NUM_REFINED_TRANSITIONS_COLUMN),
        selected_mzs.alias(_REFINED_MZS_COLUMN),
    ]

    return joined.select(
        *(
            _qualified_column_ref("dataset", column_name).alias(column_name)
            for column_name in dset.data.columns
            if column_name not in output_column_names
        ),
        *output_columns,
    )


def _wrap_intensity_dataset(
    dset: _PsmDataset,
    quant_data,
    sample_column: str,
    intensity_column: str,
):
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
            dset.semantics,
            **{
                intensity_column: _XIC_AREA,
            },
        ),
    )

    if isinstance(dset, _ConfidenceDataset):
        kwargs["qvalue_column"] = dset.qvalue_column
        kwargs["errprob_column"] = dset.errprob_column
        kwargs["pi0"] = dset.pi0
        return _PsmIntensityConfidenceDataset(quant_data, **kwargs)

    return _PsmIntensityDataset(quant_data, **kwargs)


def _apply_normalization(dset, normalization):
    if normalization is None:
        return dset

    if isinstance(normalization, _Mapping):
        normalization = dict(normalization)
        if "backend" not in normalization:
            raise ValueError(
                "normalization mapping must include a 'backend' key"
            )
        norm_backend = normalization.pop("backend")
    else:
        norm_backend = normalization
        normalization = dict()

    if not callable(norm_backend):
        from fulcrum.quant.normalization import (
            get_backend as _get_normalization_backend,
        )

        norm_backend = _get_normalization_backend(norm_backend)

    normalized = norm_backend(dset, **normalization)
    return normalized.with_data(
        normalized.data,
        semantics={
            normalized.intensity_column: _NORMALIZED_XIC_AREA,
        },
    )


def _join_condition(
    left_alias: str,
    right_alias: str,
    precursor_keys: _Sequence[str],
) -> _Column:
    return _reduce(
        lambda left, right: left & right,
        (
            _column_ref(key, qualifier=left_alias)
            == _column_ref(key, qualifier=right_alias)
            for key in precursor_keys
        ),
    )


def _sum_array(columns: _Iterable[_Column]) -> _Column:
    return _fns.aggregate(
        _fns.array(*columns),
        _fns.lit(0.0),
        lambda acc, value: acc + value,
    )


def _column_ref(column_name: str, qualifier: str = None) -> _Column:
    if qualifier is not None:
        return _qualified_column_ref(qualifier, column_name)

    return _fns.col(_quote_column(column_name))


def _qualified_column_ref(qualifier: str, column_name: str) -> _Column:
    return _fns.col(f"{qualifier}.{_quote_column(column_name)}")


def _quote_column(column_name: str) -> str:
    if column_name.startswith("`") and column_name.endswith("`"):
        return column_name

    return f"`{column_name}`"


def _unquote_column(column_name: str) -> str:
    if column_name.startswith("`") and column_name.endswith("`"):
        return column_name[1:-1]

    return column_name
