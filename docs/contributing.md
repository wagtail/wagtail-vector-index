# Contributing to Wagtail Vector Index

## Getting the Code

To make changes to this project, first clone this repository:

```shell
git clone https://github.com/wagtail/wagtail-vector-index.git
cd wagtail-vector-index
```

## Setting up your development environment

The easiest way to bootstrap your development environment is using `tox`. We recommending installing this in an isolated environment using `pipx`:

```shell
python -m pip install pipx-in-pipx --user
pipx install tox
```

Alternative installation methods can be found in the [tox documentation](https://tox.wiki/en/latest/installation.html).

Once installed, create a development virtual environment with:

```shell
tox devenv -e interactive
. ./venv/bin/activate
```

### Using devcontainers

Alernatively, a [devcontainer](https://containers.dev/) configuration is available in this repository with `tox` configured.

Using this devcontainer in VSCode will automatically enable the virtual environment, in other devcontainer environments you will need to activate it with `. ./venv/bin/activate`.

## Working with the test application

A Wagtail example for testing/development is bundled in this repo (find it at `tests/testapp`).

You can interact with this application inside your virtual environment using the `testmanage.py` script as you would a normal Django/Wagtail app.

For example, to bring up a development server on port `8000`, run:

```shell
python testmanage.py runserver 0:8000
```

If you have bootstrapped your environment with `tox`, there will be a default admin user (username: `admin`, password: `changeme`) available.

### Developing with the pgvector backend

By default, the development environment runs using the `NumpyBackend` but if you need to test with large datasets or work on the `PgvectorBackend` itself, you'll need a local PostgreSQL instance with `pgvector` installed.

This project comes bundled with a `docker-compose.yml` file which can bring up a suitable database instance for you.

To use it:

1. Install [Docker](https://docs.docker.com/engine/install/)
2. Run `docker-compose up -d`

You can then configure the test application to use this database by setting the following environment variables.

```
WAGTAIL_VECTOR_INDEX_DEFAULT_BACKEND=pgvector
DATABASE_URL=postgres://postgres:postgres@localhost:5432/postgres
```

If you're using the devcontainer, this database service is automatically started for you.

## Working with your own application

If you already have an application you'd like to use when developing Wagtail Vector Index, you can install the package directly from source alongside it using `flit`:

```
# Install flit
python -m pip install flit
# Change directory to where you have cloned the wagtail-ai repo
cd wagtail-vector-index
# Install the package using 'symlink' mode so you can change the code without having to reinstall it
python -m flit install -s
```

## pre-commit

This project uses [pre-commit](https://github.com/pre-commit/pre-commit) to help keep to coding standards by automatically checking your commits.

If you are using the devcontainer, this is automatically configured. In other environments run:

```shell
# go to the project directory
cd wagtail-vector-index
# initialize pre-commit
pre-commit install

# Optional, run all checks once for this, then the checks will run only on the changed files
git ls-files --others --cached --exclude-standard | xargs pre-commit run --files
```

## Running tests

You can run tests using `tox`:

```shell
tox
```

or, you can run them for a specific environment `tox -e python3.11-django4.2-wagtail5.2` or specific test
`tox -e python3.11-django4.2-wagtail5.2-sqlite wagtail-ai.tests.test_file.TestClass.test_method`

## Building the documentation

Documentation for this package is built using `mkdocs`. These are automatically built by ReadTheDocs when pushed to Github, but you can build them yourself locally by:

```
# Installing the package with docs depdendencies
pip install -e .[docs] -U
# Build the docs
mkdocs build
```
