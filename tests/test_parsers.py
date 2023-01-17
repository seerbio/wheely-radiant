"""Tests for parsing implementations"""
import pyspark.sql

from wheely_pythia.parsers import read_pythia_features


def test_read_pythia_features(spark_session, real_pythia_features):
    """Test that we parse crux files correctly"""
    psms = read_pythia_features(real_pythia_features, spark_session)
    assert isinstance(psms.data, pyspark.sql.DataFrame)
    assert psms.data.count() == 1770
    assert list(psms.spectrum_columns) == ["id"]
    assert all(col in psms.spectra.columns for col in psms.spectrum_columns)

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
        "pred_rt_diff",
        "rel_pred_rt",
    }
    assert set(psms.score_columns) == scores

    assert psms.scores.toPandas().shape == (1770, len(scores))

    target_df = psms.data.select(psms.targets).toPandas()

    assert target_df.shape == (1770, 1)
    assert target_df[target_df.columns[0]].sum() == 901
    assert (~target_df[target_df.columns[0]]).sum() == 1770 - 901
