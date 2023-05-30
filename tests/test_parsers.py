"""Tests for parsing implementations"""
import pyspark.sql

from wheely_pythia.parsers import read_pythia_features


def test_read_pythia_features(spark_session, pythia_features):
    """Test that we parse crux files correctly"""
    psms = read_pythia_features(pythia_features, spark_session)
    assert isinstance(psms.data, pyspark.sql.DataFrame)
    assert psms.data.count() == 1000
    assert list(psms.spectrum_columns) == ["filename", "scanNumber"]
    assert all(col in psms.spectra.columns for col in psms.spectrum_columns)

    scores = {
        "charge",
        "cosineSim",
        "discScore",
        "discScoreMax",
        "discScoreMean",
        "discScoreMedian",
        "discScoreMin",
        "discScoreStDev",
        "fractionFound",
        "frameCandidateCount",
        "frameError",
        "frameFStat",
        "frameRankDiscScore",
        "frameRankScore",
        "ionsFound",
        "isotopeFoundCount",
        "klDiv",
        "missedCleavages",
        "monoIsoOffset",
        "ms1CosineSim",
        "mz",
        "mzFound",
        "peptideSize",
        "ppmDiffMs1",
        "score",
        "scoreMax",
        "scoreMean",
        "scoreMedian",
        "scoreStDev",
    }
    assert set(psms.score_columns) == scores

    assert psms.scores.toPandas().shape == (1000, len(scores))

    target_df = psms.data.select(psms.targets).toPandas()

    assert target_df.shape == (1000, 1)
    assert target_df[target_df.columns[0]].sum() == 481
    assert (~target_df[target_df.columns[0]]).sum() == 1000 - 481


def test_read_pythia_hdf_features(spark_session, pythia_hdf_features):
    """Test that we parse crux files correctly"""
    psms = read_pythia_features(pythia_hdf_features, spark_session)
    assert isinstance(psms.data, pyspark.sql.DataFrame)
    assert psms.data.count() == 1000
    assert list(psms.spectrum_columns) == ["filename", "scanNumber"]
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
    }
    assert set(psms.score_columns) == scores

    assert psms.scores.toPandas().shape == (1000, len(scores))

    target_df = psms.data.select(psms.targets).toPandas()

    assert target_df.shape == (1000, 1)
    assert target_df[target_df.columns[0]].sum() == 691
    assert (~target_df[target_df.columns[0]]).sum() == 1000 - 691
