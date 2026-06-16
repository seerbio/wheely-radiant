"""`wheely-radiant`: Radiant DIA results reader"""

# Initialize the wheely-radiant package.
try:
    from importlib.metadata import version, PackageNotFoundError

    try:
        __version__ = version("wheely-radiant")
    except PackageNotFoundError:
        pass

except ImportError:
    from pkg_resources import get_distribution, DistributionNotFound

    try:
        __version__ = get_distribution("wheely-radiant").version
    except DistributionNotFound:
        pass

# Here is where we can export public functions and classes.

from .parsers import read_radiant_features
from .quant import quantify_radiant

from .scoring import (
    radiant_scores_default,
    radiant_score_classifier,
    radiant_scores_svm,
)
