"""Fixtures that are used in multiple tests"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np
import pytest
from pyspark.sql import SparkSession


def quiet_py4j():
    """Suppress spark logging for the test context."""
    logger = logging.getLogger("py4j")
    logger.setLevel(logging.WARN)


@pytest.fixture(scope="session")
def spark_session(request):
    """Fixture for creating a spark context."""

    spark = (
        SparkSession.builder.master("local[2]")
        # .config('spark.jars.packages', 'com.databricks:spark-avro_2.11:3.0.1')
        .appName("pytest-pyspark-local-testing")
        # .enableHiveSupport()
        .getOrCreate()
    )
    request.addfinalizer(lambda: spark.stop())

    quiet_py4j()
    return spark


@pytest.fixture(
    params=[
        "data/1.mzML.subset.v1.prqFF.pythiaDIA",
        "data/EXP22092_2022ms0742X32_A.raw.mzML_trunc.prqFF.pythiaDIA",
    ]
)
def pythia_features(request):
    """Return the path of a Parquet (DIA) PSM file from Pythia"""
    return Path(request.param)
