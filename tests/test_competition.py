import numpy as np
import pandas as pd
import pytest

from wheely_radiant.competition import resolve_isobaric_features


def candidate(peptide, unique_mz, unique_intensity, score=0.1, decoy=0):
    row = {
        "PeptideStringWithMods": peptide,
        "Mass": 1000.0,
        "Charge": 2,
        "ScanTime": 10.0,
        "ScanTimeStart": 9.95,
        "ScanTimeEnd": 10.05,
        "ClassifierScore": score,
        "IsDecoy": decoy,
        "filename": "run1",
    }
    for index in range(1, 13):
        row[f"MzSearched{index}"] = 200.0 + index * 100 if index <= 4 else -1.0
        row[f"IntensityFoundMax{index}"] = 100.0 if index <= 4 else 0.0
        row[f"CosineSimToAnchor{index}"] = 1.0 if index <= 5 else 0.0
    row["MzSearched5"] = unique_mz
    row["IntensityFoundMax5"] = unique_intensity
    return row


@pytest.fixture
def competing():
    return pd.DataFrame(
        [
            candidate("PEPTIDEA", 700.0, 0.0, score=0.001),
            candidate("PEPTIDEB", 800.0, 80.0, score=0.05),
        ]
    )


def retained(frame, **kwargs):
    result = resolve_isobaric_features(frame, **kwargs)
    return set(result.loc[result.isobaric_keep, "PeptideStringWithMods"])


def test_discriminating_evidence_beats_native_score(competing):
    original = competing.copy(deep=True)
    result = resolve_isobaric_features(competing)
    assert retained(competing) == {"PEPTIDEB"}
    assert result.isobaric_group_size.tolist() == [2, 2]
    assert result.isobaric_unique_fragments.tolist() == [0, 1]
    assert result.isobaric_unique_intensity.tolist() == [0.0, 80.0]
    pd.testing.assert_frame_equal(competing, original)


def test_protein_annotations_do_not_affect_competition(competing):
    competing["ProteinGroup"] = ["human", "entrapment"]
    first = retained(competing)
    competing["ProteinGroup"] = ["entrapment", "human"]
    competing["IsDecoy"] = [1, 0]
    assert retained(competing) == first


def test_no_distinguishing_evidence_rejects_component(competing):
    competing["IntensityFoundMax5"] = 0.0
    assert retained(competing) == set()


def test_decoy_wins_equal_evidence_and_score():
    frame = pd.DataFrame(
        [
            candidate("TARGET", 700.0, 50.0),
            candidate("DECOY", 800.0, 50.0, decoy=1),
        ]
    )
    assert retained(frame) == {"DECOY"}


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_order_invariance(competing, seed):
    assert retained(competing.sample(frac=1, random_state=seed)) == {
        "PEPTIDEB"
    }


@pytest.mark.parametrize(
    "column,value", [("Mass", 1000.1), ("Charge", 3), ("ScanTime", 10.2)]
)
def test_distinct_features_do_not_compete(competing, column, value):
    competing.loc[1, column] = value
    assert retained(competing) == {"PEPTIDEA", "PEPTIDEB"}


def test_duplicate_fragment_traces_do_not_inflate_support(competing):
    for index in range(2, 5):
        competing[f"MzSearched{index}"] = competing["MzSearched1"]
    assert retained(competing) == {"PEPTIDEA", "PEPTIDEB"}


def test_runs_cannot_be_mixed(competing):
    competing.loc[1, "filename"] = "run2"
    with pytest.raises(ValueError, match="separately per run"):
        resolve_isobaric_features(competing)


def test_missing_full_features_fail_explicitly(competing):
    with pytest.raises(ValueError, match="full Radiant features"):
        resolve_isobaric_features(competing.drop(columns="MzSearched12"))


@pytest.mark.parametrize(
    "column", ["Mass", "Charge", "ScanTime", "ClassifierScore"]
)
def test_nonfinite_precursor_fields_rejected(competing, column):
    competing.loc[0, column] = np.nan
    with pytest.raises(ValueError, match="Invalid precursor"):
        resolve_isobaric_features(competing)


def test_empty_input(competing):
    result = resolve_isobaric_features(competing.iloc[:0])
    assert result.empty
    assert "isobaric_keep" in result


def test_repeated_competition_is_rejected(competing):
    result = resolve_isobaric_features(competing)
    with pytest.raises(ValueError, match="already been applied"):
        resolve_isobaric_features(result)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"precursor_ppm": 0},
        {"fragment_ppm": np.inf},
        {"min_shared_fragments": 2.5},
        {"min_trace_cosine": 1.1},
        {"max_apex_width_fraction": -1},
        {"max_rows": 0},
    ],
)
def test_invalid_parameters_rejected(competing, kwargs):
    with pytest.raises(ValueError):
        resolve_isobaric_features(competing, **kwargs)


def test_per_run_memory_guard(competing):
    with pytest.raises(ValueError, match="memory guard"):
        resolve_isobaric_features(competing, max_rows=1)


def test_spark_partitioning_and_audit(competing, spark_session, tmp_path):
    from pyspark.sql.utils import AnalysisException
    from wheely.mammoth import PsmDataset
    from wheely_radiant.competition import compete_isobaric_features

    another_run = competing.assign(filename="run2")
    frame = pd.concat([competing, another_run], ignore_index=True)
    frame["target"] = frame.IsDecoy.eq(0)
    dataset = PsmDataset(
        spark_session.createDataFrame(frame),
        target_column="target",
        score_columns=["ClassifierScore"],
        spectrum_columns=["filename", "Mass", "ScanTime"],
        peptide_column="PeptideStringWithMods",
        charge_column="Charge",
    )
    location = tmp_path / "competition-audit"
    result = compete_isobaric_features(dataset, audit_location=location)
    kept = result.data.toPandas()
    assert set(kept.filename) == {"run1", "run2"}
    assert kept.PeptideStringWithMods.tolist() == ["PEPTIDEB", "PEPTIDEB"]
    assert result.score_columns == dataset.score_columns
    assert spark_session.read.parquet(str(location)).count() == 4
    with pytest.raises(AnalysisException):
        compete_isobaric_features(dataset, audit_location=location)
