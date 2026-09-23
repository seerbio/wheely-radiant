"""Conservative competition for coeluting, isobaric DIA assignments."""

import numpy as _np
import pandas as _pd

_ION_INDICES = tuple(range(1, 13))
_ANNOTATIONS = (
    "isobaric_group_size",
    "isobaric_unique_fragments",
    "isobaric_unique_intensity",
    "isobaric_keep",
)


def resolve_isobaric_features(
    frame: _pd.DataFrame,
    precursor_ppm: float = 5.0,
    fragment_ppm: float = 20.0,
    min_shared_fragments: int = 4,
    min_trace_cosine: float = 0.5,
    max_apex_width_fraction: float = 0.5,
    max_rows: int = 500_000,
) -> _pd.DataFrame:
    """Annotate a single run with label-blind fragment competition.

    Link equal-charge candidates when their theoretical neutral masses and
    observed peak apices agree and at least ``min_shared_fragments`` distinct
    supported fragment traces are shared. In each connected component, retain
    the candidate with the greatest unshared, cosine-squared-weighted fragment
    intensity. A component with no unshared supported fragments is unresolved
    and is rejected. Singletons are unchanged.

    Unshared means absent from the other candidates' extracted fragment lists,
    not absent from all possible theoretical fragments or library peptides.
    This is a conservative single-assignment policy, not PTM localization or
    proof of sequence identity. Truly coeluting isobaric peptides may be lost.
    Protein accessions, entrapment annotations and q-values are not used.
    Decoys receive priority only in otherwise equal evidence/score ties.

    All rows are returned, including rejected assignments, so callers can
    retain a complete audit before filtering ``isobaric_keep``. Input scores
    and quantities are not changed. Confidence must be re-estimated downstream.
    """
    positive_parameters = {
        "precursor_ppm": precursor_ppm,
        "fragment_ppm": fragment_ppm,
        "max_apex_width_fraction": max_apex_width_fraction,
    }
    for name, value in positive_parameters.items():
        if not _np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if precursor_ppm >= 1_000_000 or fragment_ppm >= 1_000_000:
        raise ValueError("Mass tolerances must be less than one million ppm")
    if not 0 <= min_trace_cosine <= 1:
        raise ValueError("min_trace_cosine must be between zero and one")
    if not isinstance(
        min_shared_fragments, int
    ) or not 1 <= min_shared_fragments <= len(_ION_INDICES):
        raise ValueError(
            "min_shared_fragments must be an integer from 1 to 12"
        )
    if not isinstance(max_rows, int) or max_rows < 1:
        raise ValueError("max_rows must be a positive integer")
    if len(frame) > max_rows:
        raise ValueError(f"Run exceeds the {max_rows} candidate memory guard")
    required = [
        "Mass",
        "Charge",
        "ScanTime",
        "ScanTimeStart",
        "ScanTimeEnd",
        "ClassifierScore",
        "IsDecoy",
        "PeptideStringWithMods",
        *[f"MzSearched{index}" for index in _ION_INDICES],
        *[f"IntensityFoundMax{index}" for index in _ION_INDICES],
        *[f"CosineSimToAnchor{index}" for index in _ION_INDICES],
    ]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(
            f"Fragment competition requires full Radiant features: {missing}"
        )
    if set(_ANNOTATIONS) & set(frame.columns):
        raise ValueError("Fragment competition has already been applied")
    if "filename" in frame and frame["filename"].nunique(dropna=False) > 1:
        raise ValueError(
            "Fragment competition must be applied separately per run"
        )
    result = frame.reset_index(drop=True).copy()
    count = len(result)
    mass = result["Mass"].to_numpy(dtype=float)
    charge = result["Charge"].to_numpy(dtype=float)
    apex = result["ScanTime"].to_numpy(dtype=float)
    width = (result["ScanTimeEnd"] - result["ScanTimeStart"]).to_numpy(
        dtype=float
    )
    score = result["ClassifierScore"].to_numpy(dtype=float)
    decoy = result["IsDecoy"].to_numpy()
    if (
        not _np.isfinite(
            _np.column_stack([mass, charge, apex, width, score])
        ).all()
        or (mass <= 0).any()
        or (charge <= 0).any()
        or (charge != _np.floor(charge)).any()
        or (width <= 0).any()
        or not _np.isin(decoy, [0, 1]).all()
    ):
        raise ValueError(
            "Invalid precursor mass, charge, peak, score or decoy flag"
        )
    searched = result[
        [f"MzSearched{index}" for index in _ION_INDICES]
    ].to_numpy(dtype=float)
    intensity = result[
        [f"IntensityFoundMax{index}" for index in _ION_INDICES]
    ].to_numpy(dtype=float)
    cosine = result[
        [f"CosineSimToAnchor{index}" for index in _ION_INDICES]
    ].to_numpy(dtype=float)
    supported = (
        _np.isfinite(searched)
        & _np.isfinite(intensity)
        & _np.isfinite(cosine)
        & (searched > 0)
        & (intensity > 0)
        & (cosine >= min_trace_cosine)
    )
    weighted = _np.where(supported, intensity * _np.clip(cosine, 0, 1) ** 2, 0)
    for row in range(count):
        ordered = _np.argsort(-weighted[row], kind="stable")
        retained = []
        for ion in ordered:
            if not supported[row, ion]:
                continue
            if any(
                abs(searched[row, ion] - searched[row, previous])
                <= fragment_ppm
                * 1e-6
                * max(searched[row, ion], searched[row, previous])
                for previous in retained
            ):
                supported[row, ion] = False
                weighted[row, ion] = 0
            else:
                retained.append(ion)

    parents = _np.arange(count)

    def find_root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for charge_value in _np.unique(charge):
        subset = _np.flatnonzero(charge == charge_value)
        ordered = subset[_np.argsort(mass[subset], kind="stable")]
        ordered_mass = mass[ordered]
        for position, first in enumerate(ordered):
            stop = _np.searchsorted(
                ordered_mass,
                mass[first] / (1 - precursor_ppm * 1e-6),
                side="right",
            )
            for second in ordered[position + 1 : stop]:
                if abs(apex[first] - apex[second]) > (
                    max_apex_width_fraction * min(width[first], width[second])
                ):
                    continue
                matched = _np.abs(
                    searched[first, :, None] - searched[second, None, :]
                ) <= fragment_ppm * 1e-6 * _np.maximum(
                    searched[first, :, None], searched[second, None, :]
                )
                matched &= (
                    supported[first, :, None] & supported[second, None, :]
                )
                shared = min(
                    matched.any(axis=0).sum(), matched.any(axis=1).sum()
                )
                if shared < min_shared_fragments:
                    continue
                first_root, second_root = find_root(first), find_root(second)
                if first_root != second_root:
                    parents[second_root] = first_root

    components = {}
    for index in range(count):
        components.setdefault(find_root(index), []).append(index)
    group_sizes = _np.ones(count, dtype="int32")
    unique_counts = _np.full(count, -1, dtype="int32")
    unique_intensities = _np.full(count, _np.nan)
    keep = _np.ones(count, dtype=bool)
    peptides = result["PeptideStringWithMods"].astype(str).to_numpy()
    for members in components.values():
        if len(members) == 1:
            continue
        indices = _np.asarray(members)
        group_sizes[indices] = len(indices)
        keep[indices] = False
        for index in indices:
            competitors = searched[indices[indices != index]].ravel()
            competitors = competitors[
                _np.isfinite(competitors) & (competitors > 0)
            ]
            shared = (
                _np.abs(searched[index, :, None] - competitors[None, :])
                <= fragment_ppm
                * 1e-6
                * _np.maximum(searched[index, :, None], competitors[None, :])
            ).any(axis=1)
            unique = supported[index] & ~shared
            unique_counts[index] = unique.sum()
            unique_intensities[index] = weighted[index, unique].sum()
        winner = min(
            indices,
            key=lambda index: (
                -unique_intensities[index],
                score[index],
                -int(decoy[index]),
                peptides[index],
                apex[index],
                mass[index],
            ),
        )
        if unique_intensities[winner] > 0:
            keep[winner] = True
    result["isobaric_group_size"] = group_sizes
    result["isobaric_unique_fragments"] = unique_counts
    result["isobaric_unique_intensity"] = unique_intensities
    result["isobaric_keep"] = keep
    return result


def compete_isobaric_features(dset, audit_location=None, **kwargs):
    """Resolve each run independently and return the retained PSM dataset.

    When ``audit_location`` is provided, retain all annotated rows in a new
    Parquet dataset and reload that materialization before filtering. Existing
    audits are never overwritten. Large experiments are partitioned by run;
    ``max_rows`` limits each Python worker's in-memory candidate table.
    """
    from pyspark.sql import functions, types

    if "filename" not in dset.data.columns:
        raise ValueError("Per-run fragment competition requires filename")
    schema = types.StructType(
        [
            *dset.data.schema.fields,
            types.StructField(
                "isobaric_group_size", types.IntegerType(), False
            ),
            types.StructField(
                "isobaric_unique_fragments", types.IntegerType(), False
            ),
            types.StructField(
                "isobaric_unique_intensity", types.DoubleType(), True
            ),
            types.StructField("isobaric_keep", types.BooleanType(), False),
        ]
    )

    def resolve_run(frame):
        return resolve_isobaric_features(frame, **kwargs)

    annotated = dset.data.groupBy("filename").applyInPandas(
        resolve_run, schema=schema
    )
    if audit_location is not None:
        annotated.write.mode("errorifexists").parquet(str(audit_location))
        annotated = dset.data.sparkSession.read.parquet(str(audit_location))
    return dset.with_data(annotated.filter(functions.col("isobaric_keep")))
