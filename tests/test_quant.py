"""
Tests for Radiant peptide quantification backends.
"""

import math

import pytest
from pyspark.sql import functions as fns
from wheely.mammoth import ConfidenceDataset
from wheely.mammoth.semantics import NORMALIZED_XIC_AREA, XIC_AREA

from wheely_radiant.quant import quantify_radiant


def _fragment_columns(
    mzs=(0.0,) * 12,
    scores=(0.0,) * 12,
    intensities=(0.0,) * 12,
):
    mzs = [*mzs, *([0.0] * 12)][:12]
    scores = [*scores, *([0.0] * 12)][:12]
    intensities = [*intensities, *([0.0] * 12)][:12]

    cols = {}
    for idx, (mz, score, intensity) in enumerate(
        zip(mzs, scores, intensities),
        start=1,
    ):
        cols[f"MzSearched{idx}"] = mz
        cols[f"CosineSimToAnchor{idx}"] = score
        cols[f"IntensityFoundMax{idx}"] = intensity

    return cols


@pytest.fixture
def radiant_quant_dataset(spark_session):
    rows = [
        dict(
            sample="s1",
            PeptideStringWithMods="PEPTIDE",
            Charge=2,
            TargetKey="window_a",
            target=True,
            qvalue=0.001,
            errprob=0.01,
            score=10.0,
            ProteinGroup="P1",
            **_fragment_columns(
                mzs=(100.0, 200.0, 300.0),
                scores=(0.90, 0.70, 0.10),
                intensities=(10.0, 20.0, 30.0),
            ),
        ),
        dict(
            sample="s2",
            PeptideStringWithMods="PEPTIDE",
            Charge=2,
            TargetKey="window_a",
            target=False,
            qvalue=0.001,
            errprob=0.01,
            score=11.0,
            ProteinGroup="P1",
            **_fragment_columns(
                mzs=(100.0, 200.0, 300.0),
                scores=(0.70, 0.95, 0.10),
                intensities=(11.0, 21.0, 31.0),
            ),
        ),
        dict(
            sample="s3",
            PeptideStringWithMods="PEPTIDE",
            Charge=2,
            TargetKey="window_b",
            target=True,
            qvalue=0.50,
            errprob=0.50,
            score=1.0,
            ProteinGroup="P1",
            **_fragment_columns(
                mzs=(100.0, 200.0, 300.0),
                scores=(0.95, 0.95, 0.95),
                intensities=(999.0, 999.0, 999.0),
            ),
        ),
    ]

    return ConfidenceDataset(
        spark_session.createDataFrame(rows),
        target_column="target",
        score_columns=["score"],
        spectrum_columns=["sample"],
        peptide_column="PeptideStringWithMods",
        charge_column="Charge",
        protein_column="ProteinGroup",
        protein_delim=";",
        qvalue_column="qvalue",
        errprob_column="errprob",
    )


def test_quantify_radiant_uses_target_key_by_default(radiant_quant_dataset):
    quantified = quantify_radiant(
        radiant_quant_dataset,
        sample_column="sample",
    )

    rows = {
        row["sample"]: row.asDict()
        for row in quantified.data.select(
            "sample",
            "radiant_intensity",
            "radiant_fragments_found",
            "radiant_num_refined_transitions",
        ).collect()
    }

    assert rows["s1"]["radiant_intensity"] == pytest.approx(30.0)
    assert rows["s1"]["radiant_fragments_found"] == 2
    assert rows["s1"]["radiant_num_refined_transitions"] == 2
    assert rows["s2"]["radiant_intensity"] == pytest.approx(32.0)
    assert rows["s2"]["radiant_fragments_found"] == 2
    assert rows["s3"]["radiant_intensity"] is None
    assert rows["s3"]["radiant_fragments_found"] == 0
    assert rows["s3"]["radiant_num_refined_transitions"] is None

    assert quantified.intensity_column == "radiant_intensity"
    assert quantified.get_by_semantics(XIC_AREA) == "radiant_intensity"


def test_quantify_radiant_can_ignore_target_key(radiant_quant_dataset):
    quantified = quantify_radiant(
        radiant_quant_dataset,
        sample_column="sample",
        include_target_key=False,
    )

    rows = {
        row["sample"]: row["radiant_intensity"]
        for row in quantified.data.select("sample", "radiant_intensity")
        .orderBy("sample")
        .collect()
    }

    assert rows == {
        "s1": pytest.approx(30.0),
        "s2": pytest.approx(32.0),
        "s3": pytest.approx(1998.0),
    }


def test_quantify_radiant_can_fallback_to_raw(spark_session):
    rows = [
        dict(
            sample="raw",
            PeptideStringWithMods="RAWONLY",
            Charge=2,
            TargetKey="window_a",
            target=True,
            qvalue=0.50,
            errprob=0.50,
            score=1.0,
            ProteinGroup="P1",
            TotalIntensityRaw=123.0,
            **_fragment_columns(
                mzs=(100.0,),
                scores=(0.90,),
                intensities=(999.0,),
            ),
        ),
        dict(
            sample="match",
            PeptideStringWithMods="REFINED",
            Charge=2,
            TargetKey="window_a",
            target=True,
            qvalue=0.001,
            errprob=0.01,
            score=10.0,
            ProteinGroup="P1",
            TotalIntensityRaw=111.0,
            **_fragment_columns(
                mzs=(100.0,),
                scores=(0.90,),
                intensities=(10.0,),
            ),
        ),
        dict(
            sample="no_match",
            PeptideStringWithMods="REFINED",
            Charge=2,
            TargetKey="window_a",
            target=True,
            qvalue=0.50,
            errprob=0.50,
            score=1.0,
            ProteinGroup="P1",
            TotalIntensityRaw=456.0,
            **_fragment_columns(
                mzs=(200.0,),
                scores=(0.10,),
                intensities=(456.0,),
            ),
        ),
    ]
    dset = ConfidenceDataset(
        spark_session.createDataFrame(rows),
        target_column="target",
        score_columns=["score"],
        spectrum_columns=["sample"],
        peptide_column="PeptideStringWithMods",
        charge_column="Charge",
        protein_column="ProteinGroup",
        protein_delim=";",
        qvalue_column="qvalue",
        errprob_column="errprob",
    )

    quantified = quantify_radiant(
        dset,
        sample_column="sample",
        fallback_to_raw=True,
        num_fragments=1,
    )

    rows = {
        row["sample"]: row.asDict()
        for row in quantified.data.select(
            "sample",
            "radiant_intensity",
            "radiant_fragments_found",
            "radiant_num_refined_transitions",
        ).collect()
    }

    assert rows["raw"]["radiant_intensity"] == pytest.approx(123.0)
    assert rows["raw"]["radiant_fragments_found"] == 0
    assert rows["raw"]["radiant_num_refined_transitions"] is None
    assert rows["match"]["radiant_intensity"] == pytest.approx(10.0)
    assert rows["match"]["radiant_fragments_found"] == 1
    assert rows["match"]["radiant_num_refined_transitions"] == 1
    assert rows["no_match"]["radiant_intensity"] is None
    assert rows["no_match"]["radiant_fragments_found"] == 0
    assert rows["no_match"]["radiant_num_refined_transitions"] == 1


def test_quantify_radiant_warns_for_nonstandard_precursor_columns(
    spark_session,
    caplog,
):
    row = dict(
        sample="s1",
        peptide="PEPTIDE",
        z=2,
        TargetKey="window_a",
        target=True,
        qvalue=0.001,
        errprob=0.01,
        score=10.0,
        ProteinGroup="P1",
        **_fragment_columns(
            mzs=(100.0,),
            scores=(0.90,),
            intensities=(10.0,),
        ),
    )
    dset = ConfidenceDataset(
        spark_session.createDataFrame([row]),
        target_column="target",
        score_columns=["score"],
        spectrum_columns=["sample"],
        peptide_column="peptide",
        charge_column="z",
        protein_column="ProteinGroup",
        protein_delim=";",
        qvalue_column="qvalue",
        errprob_column="errprob",
    )

    quantified = quantify_radiant(dset, sample_column="sample")

    assert quantified.data.select("radiant_intensity").collect()[0][0] == 10.0
    assert "dataset-annotated peptide column" in caplog.text
    assert "dataset-annotated charge column" in caplog.text


def test_quantify_radiant_applies_callable_normalization(
    radiant_quant_dataset,
):
    def normalize(dset):
        return dset.with_data(
            dset.data.withColumn(
                "normalized_radiant_intensity",
                dset.intensities / fns.lit(10.0),
            ),
            intensity_column="normalized_radiant_intensity",
        )

    quantified = quantify_radiant(
        radiant_quant_dataset,
        sample_column="sample",
        normalization=normalize,
    )

    rows = {
        row["sample"]: row["normalized_radiant_intensity"]
        for row in quantified.data.select(
            "sample",
            "normalized_radiant_intensity",
        ).collect()
    }

    assert rows["s1"] == pytest.approx(3.0)
    assert rows["s2"] == pytest.approx(3.2)
    assert rows["s3"] is None or math.isnan(rows["s3"])
    assert quantified.intensity_column == "normalized_radiant_intensity"
    assert (
        quantified.get_by_semantics(NORMALIZED_XIC_AREA)
        == "normalized_radiant_intensity"
    )
