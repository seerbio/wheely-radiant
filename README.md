**wheely-radiant**: Reader for Radiant DIA results, compatible with
[`wheely-mammoth`](https://github.com/seerbio/wheely-mammoth).

## Installation  

This library requires Python 3.8+ and can be installed with pip:  

```shell
pip install wheely-radiant
```

## Basic Usage

This package provides a plugin for the [Fulcrum Pipeline](https://github.com/seerbio/fulcrum/)'s
`read_existing` search backend.

After installing `wheely-radiant` you may use the `radiant` engine to load
existing Radiant DIA results:

```toml
[search]
backend = "read_existing"
engine = "radiant"
location = ["uri_one", "uri_two", ...]
```

(or similar in JSON or `dict` format)

## Direct Usage

To load raw PSM scores from a Radiant Parquet (`.radiantDIA`) file, use the function
`read_radiant_features()`.

```pycon
>>> from wheely_radiant import read_radiant_features
>>> ds = read_radiant_features("data/1.mzML.subset.radiantDIA")
>>> type(ds)
<class 'wheely.mammoth.dataset.PsmDataset'>
>>> ds.scores.select(ds.score_columns[1]).describe().toPandas()
  summary            cosineSim
0   count                 1000
1    mean   0.3947062000000001
2  stddev  0.15665395271892263
3     min               0.0315
4     max               0.9926
```

To read multiple files, pass a tuple, list, array, or series of file paths.
Currently only full paths are supported; you can not pass wildcard ("glob")
paths to the function.
