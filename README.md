# Template for Python tool or library Projects

## Using this template

1. Create a new repository for your project by following
   [GitHub's documentation](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template).
2. Update `setup.cfg` and `README.md` (this file) to define the project's
   name and dependencies. Remove references to `python-project-template`.
   Document the installation and usage of your project in `README.md`.
3. Create relevant package/module structure in your project repo. Typically
   all sources should be contained in a folder with the same name as the project
   (or whatever the root package is named in the Python import namespace). This
   folder should contain an `__init__.py` which imports/re-exports the key
   symbols in the library.

   This file MUST also set up the necessary version information (this is
   required by `setuptools_scm` which is used in the build and release process
   set up by this project template):

    ```python
    """
    project-name: <help string>

    Exports:

    - TODO
    """

    # Initialize the package.
    try:
        from importlib.metadata import version, PackageNotFoundError

        try:
            __version__ = version("package-name")  # TODO: update me!
        except PackageNotFoundError:
            pass

    except ImportError:
        from pkg_resources import get_distribution, DistributionNotFound

        try:
            __version__ = get_distribution("package-name").version  # TODO: update me!
        except DistributionNotFound:
            pass

    # Here is where we can export public functions and classes.
    from .package import Symbol  # import relative to this package to avoid namespace collisions
    ```
4. Follow the instructions in `CONTRIBUTING.md` to contribute to the project.

## Installation  

This library requires Python 3.8+ and can be installed with pip:  

```shell
pip install package-name  # TODO
```

## Basic Usage  

TODO: describe using this project.
At a minimum, include a brief demo of using the library from the Python REPL:

```pycon
>>> from my.favorite.library import function
>>> res = function(*args)  # TODO
>>> print(res)
Some amazing output goes here: Hello World.
```
