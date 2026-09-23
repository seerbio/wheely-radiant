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

## Isobaric Fragment Competition (Opt-In)

Full native reports can contain several coeluting, equal-mass peptide
assignments supported mostly by the same fragment traces. The single-cell
branch adds a conservative, per-run single-assignment option:

```python
from wheely_radiant.competition import compete_isobaric_features

resolved = compete_isobaric_features(
    ds,
    audit_location="new-competition-audit.parquet",
    precursor_ppm=5.0,
    fragment_ppm=20.0,
    min_shared_fragments=4,
)
```

Competition requires the same precursor charge, matching mass, overlapping
peak apices, and at least four distinct supported shared fragment traces.
The largest unshared cosine-squared-weighted fragment intensity determines the
retained assignment. All-zero unshared evidence leaves the component unresolved
and rejects it. Exact evidence/score ties favor ordinary decoys. Protein
accessions, organism/entrapment labels and q-values are never selection inputs.

The optional audit preserves all rows and four `isobaric_*` annotations; existing
audit files are never overwritten. A singleton has group size 1, unique fragment
count -1 and undefined unique intensity because no competitor was evaluated.
Work is partitioned by run, with a default 500,000-row per-worker memory guard.

Re-estimate confidence after competition. This option does not establish 1%
empirical FDR, resolve all sequence ambiguity, or localize PTMs. "Unshared"
refers only to other candidates' extracted fragment lists, not every possible
theoretical ion. Real coeluting isobaric peptides can be lost. Defaults remain
unchanged; evaluate entrapment, sensitivity and quantification for your assay.
