# EventCamProcessing
MAE 506 Project



## Getting started

Ensure you have uv installed on your system: [(installation guide)](https://docs.astral.sh/uv/getting-started/installation/).


Navigate to the cloned project directory, and run the following command to install the package with required dependencies:

```bash
uv sync
```

When you make and save changes to src, the changes will be reflected in the installed package without needing to reinstall it.

### Set up pycharm

Ensure you have a recent version of pycharm that supports uv projects. Open the project directory in pycharm, and set the python interpreter to the one created by uv. Add new interpreter > Add Local Interpreter

- Select existing
- Choose UV
- Add your uv path (should be autodetected)
- Python executable at EventCamProcessing/.venv/(bin/Scripts)/python.exe

## Writing a script

Scripts should be placed in the `scripts` directory. You can run a script using the following command:

```bash
uv run <script_name>
```

Scripts should treat eventcamprocessing as an installed package, and import it accordingly. For example:

```python
from eventcamprocessing import main
```
