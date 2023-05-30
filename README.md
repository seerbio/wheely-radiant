**wheely-pythia**: Reader for Pythia results, compatible with
[`wheely-mammoth`](https://github.com/seerbio/wheely-mammoth).

## Installation  

This library requires Python 3.8+ and can be installed with pip:  

```shell
pip install wheely-pythia
```

To read the (to-be-deprecated) HDF-based `.scored` format you must
install additional optional dependencies:

```shell
pip install 'wheely-pythia[hdf]'
```

## Basic Usage

This package provides a plugin for [Scry](https://github.com/seerbio/scry/)'s
`read_existing` search backend.

After installing `wheely-pythia` you may use the `pythia` engine to load
existing Pythia results:

```toml
[search]
backend = "read_existing"
engine = "pythia"
location = ["uri_one", "uri_two", ...]
```

(or similar in JSON or `dict` format)

## Direct Usage

To load raw PSM scores from a Pythia Parquet (`.prq.pythiaDIA`) file, use the function
`read_pythia_features()` (also supports to to-be-deprecated HDF5-based `.scored` format).

```pycon
>>> from wheely_pythia import read_pythia_features
>>> ds = read_pythia_features("data/1.mzML.subset.prq.pythiaDIA")
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
