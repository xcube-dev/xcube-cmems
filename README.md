[![Build Status](https://github.com/dcs4cop/xcube-cmems/actions/workflows/unitest-workflow.yml/badge.svg)](https://github.com/dcs4cop/xcube-cmems/actions/workflows/unitest-workflow.yml)
# xcube-cmems

A [xcube plugin](https://xcube.readthedocs.io/en/latest/plugins.html) that that allows generating 
data cubes from the CMEMS API.

## Setup

### Configuring access to the CMEMS API

In order to access the CMEMS API via the `xcube-cmems` plugin, you need to create a 
[cmems account](https://resources.marine.copernicus.eu/registration-form) 
first, if you not do already have one.

### Installing the xcube-cmems plugin
This section describes three alternative methods you can use to install the
xcube-cmems plugin.

conda can sometimes be inconveniently slow when resolving an environment.
If this causes problems, consider using
[mamba](https://github.com/mamba-org/mamba) as a much faster drop-in
alternative to conda.

#### Installation into a new environment with conda

xcube-cmems and all necessary dependencies (including xcube itself) are available
on [conda-forge](https://conda-forge.org/), and can be installed using the
[conda package manager](https://docs.conda.io/projects/conda/en/latest/).
The conda package manager itself can be obtained in the [miniconda
distribution](https://docs.conda.io/en/latest/miniconda.html). 
Once conda is installed, xcube-cmems can be installed like this:

```
$ conda create --name xcube-cmems-environment --channel conda-forge xcube-cmems
$ conda activate xcube-cmems-environment
```
The name of the environment may be freely chosen.

#### Installation into an existing environment with conda

This method assumes that you have an existing conda environment, and you want
to install xcube-cmems into it.

xcube-cmems can also be installed into an existing conda environment.
With the existing environment activated, execute this command:

```
$ conda install --channel conda-forge xcube-cmems
```
Once again, xcube and any other necessary dependencies will be installed
automatically if they are not already installed.

#### Installation into an existing environment from the repository

If you want to install xcube-cmems directly from the git repository (for example
if order to use an unreleased version or to modify the code), you can
do so as follows:

```
$ conda create --name xcube-cmems-environment --channel conda-forge --only-deps xcube-cmems
$ conda activate xcube-cmems-environment
$ git clone https://github.com/dcs4cop/xcube-cmems.git
$ python -m pip install --no-deps --editable xcube-cmems/
```
This installs all the dependencies of xcube-cmems into a fresh conda environment,
then installs xcube-cmems into this environment from the repository.

# Testing

You can run the unit tests for xcube-cmems by executing

```
$ pytest
```

in the `xcube-cmems` repository. Note that, in order to successfully run the
tests using the current repository version of xcube-cmems, you may also need to
install the repository version of xcube rather than its latest conda-forge
release.

To create a test coverage report, you can use

```
pytest --cov xcube_cmems --cov-report html
```

This will write a coverage report to `htmlcov/index.html`.

## Use

Jupyter notebooks demonstrating the use of the  `xcube-cmems` plugin can be found
in the `examples/notebooks/` subdirectory of the repository.

## Releasing

To release `xcube-cmems`, please follow the steps outlined in the 
[xcube Developer Guide](https://github.com/dcs4cop/xcube/blob/master/docs/source/devguide.md#release-process).