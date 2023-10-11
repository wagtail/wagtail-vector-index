# Wagtail Vector Index

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/wagtail-vector-index.svg)](https://badge.fury.io/py/wagtail-vector-index)
[![ai CI](https://github.com/tomusher/wagtail-vector-index/actions/workflows/test.yml/badge.svg)](https://github.com/tomusher/wagtail-vector-index/actions/workflows/test.yml)

Wagtail Vector Index provides a way to turn Django models, Wagtail pages, and anything else in to embeddings which are stored in a one of multiple vector database backends.

This provides the backbone for features including:

* Natural language search
* Similarity search
* Content recommendations

## Links

- [Documentation](https://github.com/tomusher/wagtail-vector-index/blob/main/README.md)
- [Changelog](https://github.com/tomusher/wagtail-vector-index/blob/main/CHANGELOG.md)
- [Contributing](https://github.com/tomusher/wagtail-vector-index/blob/main/CHANGELOG.md)
- [Discussions](https://github.com/tomusher/wagtail-vector-index/discussions)
- [Security](https://github.com/tomusher/wagtail-vector-index/security)

## Supported Versions

* Wagtail 4.0, 4.1, 4.2, 5.0, 5.1

## Contributing

### Install

To make changes to this project, first clone this repository:

```sh
git clone https://github.com/tomusher/wagtail-vector-index.git
cd wagtail-vector-index
```

With your preferred virtualenv activated, install testing dependencies:

#### Using pip

```sh
python -m pip install --upgrade pip>=21.3
python -m pip install -e .[testing] -U
```

#### Using flit

```sh
python -m pip install flit
flit install
```

### pre-commit

Note that this project uses [pre-commit](https://github.com/pre-commit/pre-commit).
It is included in the project testing requirements. To set up locally:

```shell
# go to the project directory
$ cd wagtail-vector-index
# initialize pre-commit
$ pre-commit install

# Optional, run all checks once for this, then the checks will run only on the changed files
$ git ls-files --others --cached --exclude-standard | xargs pre-commit run --files
```

### How to run tests

Now you can run tests as shown below:

```sh
tox
```

or, you can run them for a specific environment `tox -e python3.8-django3.2-wagtail2.15` or specific test
`tox -e python3.9-django3.2-wagtail2.15-sqlite wagtail-vector-index.tests.test_file.TestClass.test_method`

To run the test app interactively, use `tox -e interactive`, visit `http://127.0.0.1:8020/admin/` and log in with `admin`/`changeme`.
