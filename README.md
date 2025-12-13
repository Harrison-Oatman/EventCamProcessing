# EventCamProcessing
EventCamProcessing is a Python package designed to facilitate the post-processing of data collected by event cameras, specifically for applications in particle tracking.


## Features

### Filter Functions
- A suite of filter functions designed to effiently process event camera data.

### Particle Detection and Tracking
- Algorithms to detect and track particles within the event data.


## Guidance for Development

Ensure you have uv installed on your system: [(installation guide)](https://docs.astral.sh/uv/getting-started/installation/).


Navigate to the cloned project directory, and run the following command to install the package with required dependencies:

```bash
uv sync
```

This will create a virtual environment in the `.venv` directory and install all necessary dependencies.

By default, this will install the dependencies for development. If you want to install only the core dependencies without development tools, use the `--no-dev` flag:

```bash
uv sync --no-dev
```

## Nox

Nox is used to automate testing and linting tasks. To run the default Nox sessions, use the following command:

```bash
uv run --group nox nox
```

This will run all tests by default. You can also specify individual sessions to run. For example, to run only the docs session, use:

```bash
uv run --group nox nox -s docs
```

## Precommit Hooks

Before a pull request is merged, pre-commit hooks will automatically run to ensure code quality. To manually run the pre-commit hooks, use the following command:

```bash
uvx pre-commit run --all-files
```


## Additional guidance
### Using PyCharm with UV

Ensure you have a recent version of pycharm that supports uv projects. Open the project directory in pycharm, and set the python interpreter to the one created by uv. Add new interpreter > Add Local Interpreter

- Select existing
- Choose UV
- Add your uv path (should be autodetected)
- Python executable at EventCamProcessing/.venv/(bin/Scripts)/python.exe
