**wheely-pythia**: Reader for Pythia results, compatible with
[`wheely-mammoth`](https://github.com/seerbio/wheely-mammoth).

## Installation  

This library requires Python 3.8+ and can be installed with pip:  

```shell
pip install git+ssh://git@github.com/seerbio/wheely-pythia@main
```

## Basic Usage

To load raw PSM scores from a Pythia `.scored` file, use the function
`read_pythia_features()`:

```pycon
>>> from wheely_pythia import read_pythia_features
>>> ds = read_pythia_features("data/test.scored")
>>> type(ds)
<class 'wheely.mammoth.dataset.PsmDataset'>
>>> ds.scores.select(ds.score_columns[0]).describe().toPandas()
  summary    cosine_similarity
0   count                 1000
1    mean   0.4806335000000002
2  stddev  0.32854787653388495
3     min                  0.0
4     max               0.9963
```

To read multiple files, pass a tuple, list, array, or series of file paths.
Currently only full paths are supported; you can not pass wildcard ("glob")
paths to the function.
