# Wagtail Vector Index

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/wagtail-vector-index.svg)](https://badge.fury.io/py/wagtail-vector-index)
[![ai CI](https://github.com/wagtail/wagtail-vector-index/actions/workflows/test.yml/badge.svg)](https://github.com/wagtail/wagtail-vector-index/actions/workflows/test.yml)

Wagtail Vector Index provides a way to turn Django models, Wagtail pages, and anything else in to embeddings which are stored in a one of multiple vector database backends.

This provides the backbone for features including:

* Natural language search
* Similarity search
* Content recommendations

## Links

- [Documentation](https://github.com/wagtail/wagtail-vector-index/blob/main/README.md)
- [Changelog](https://github.com/wagtail/wagtail-vector-index/blob/main/CHANGELOG.md)
- [Contributing](https://github.com/wagtail/wagtail-vector-index/blob/main/CHANGELOG.md)
- [Discussions](https://github.com/wagtail/wagtail-vector-index/discussions)
- [Security](https://github.com/wagtail/wagtail-vector-index/security)

## Supported Versions

* Wagtail 5.2
* Django 4.2
* Python 3.11, 3.12

## Contributing

### Install

To make changes to this project, first clone this repository:

```sh
git clone https://github.com/wagtail/wagtail-vector-index.git
cd wagtail-vector-index
```

With your preferred virtualenv activated, install testing dependencies:

#### Using pip

```sh
python -m pip install --upgrade pip>=21.3
python -m pip install -e .'[testing,llm,numpy,pgvector,qdrant,weaviate]' -U
```

#### Using flit

```sh
python -m pip install flit
python -m flit install -s
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

or, you can run them for a specific environment `tox -e py3.11-django4.2-wagtail5.2` or specific test
`tox -e py3.11-django4.2-wagtail5.2 -- tests.test_file.TestClass.test_method`

Sometimes tox contains cached dependencies, so if you want to run tests with the latest dependencies, you can use `tox -r` or run `rm -rf .tox` to delete the whole tox environment.

To run the test app interactively, use `tox -e interactive`, visit `http://127.0.0.1:8020/admin/` and log in with `admin`/`changeme`.
