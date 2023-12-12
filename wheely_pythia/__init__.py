"""`wheely-pythia`: Pythia results reader"""

# Initialize the wheely-pythia package.
try:
    from importlib.metadata import version, PackageNotFoundError

    try:
        __version__ = version("wheely-pythia")
    except PackageNotFoundError:
        pass

except ImportError:
    from pkg_resources import get_distribution, DistributionNotFound

    try:
        __version__ = get_distribution("wheely-pythia").version
    except DistributionNotFound:
        pass

# Here is where we can export public functions and classes.
from .parsers import read_pythia_features

from .parsers import (
    pythia_scores_default,
    pythia_score_classifier,
    pythia_scores_svm,
)
