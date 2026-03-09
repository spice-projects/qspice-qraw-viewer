# QSPICE QRAW Waveform Viewer

A lightweight PySide6/QML desktop application to parse and view QSPICE `.qraw` binary waveform files. It uses zero-copy numpy views and `QLineSeries.replaceNp` for efficient rendering of large datasets.

## Features

- Parse `.qraw` simulation output using a memory-mapped numpy buffer
- Lazy `Variable` views with magnitude/phase support for complex data
- Multiple decimation algorithms for interactive plotting
- Built-in FFT tooling with a configurable window registry
- QML + Python bridge for responsive charts and dialogs

## Requirements

- Python 3.12+
- PySide6 6.7+
- numpy

Install runtime dependencies with:

```bash
pip install -r requirements.txt
```

## Install from GitHub

Install directly from the repository via pip (replace `<GITHUB_URL>` with the repo HTTPS URL):

```bash
pip install git+https://github.com/spice-projects/qspice-qraw-viewer.git
```

If you plan to develop or run the UI locally, create a virtualenv and install editable plus dev dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Quick usage

Run the viewer on a `.qraw` file:

```bash
python -m viewer path/to/file.qraw
```

Parse without launching the UI (headless):

```bash
python -m viewer path/to/file.qraw --headless
```

Enable debug logging:

```bash
python -m viewer path/to/file.qraw --log-level=DEBUG
```

## Developer notes

- Compile QML `.ui` files (VS Code task: "Compile UI files"):

```bash
for f in viewer/*.ui; do .venv/bin/pyside6-uic "$f" -o "viewer/ui_$(basename ${f%.ui}).py"; done
```

- Run the test suite:

```bash
python -m unittest discover -p '*_test.py'
```

## Project layout

- `viewer/` — application package (parsers, UI, dialogs, charting, FFT)
- `tests/` — unit tests and fixtures
- `requirements.txt` — runtime dependencies

## JupyterLab integration

- **Open in JupyterLab:** the application can launch an embedded JupyterLab instance (Tools → Open in JupyterLab...) that opens a starter notebook for the currently loaded `.qraw` file.
- **Starter notebooks:** created under the user data directory at `~/.qraw-viewer/notebooks` (one notebook per `.qraw` file). Notebooks are written only once and not overwritten to preserve user edits.
- **Theme & settings:** a JupyterLab user-settings directory is created beside the notebooks folder to force the "JupyterLab Dark" theme for a consistent dark UI.
- **Helpers available in the starter notebook:** after the load cell the following objects/helpers are available for quick analysis and plotting:
	- `qraw` — the loaded `QRawFile` instance
	- `qraw.expression_manager.expressions` — list of available expressions (signals)
	- `qraw.expression_manager.evaluate('<expr>')` — evaluate a derived expression
	- `qraw.abscissa` — abscissa (time/frequency/parameter) data and metadata
	- `qraw.steps` — number of simulation steps

- **Notes:** the app launches `jupyter lab` as a subprocess and extracts the access token from its stdout; if `jupyter` is not found the app logs an error and the integration is disabled. To inspect starter notebooks directly:

```bash
ls -la ~/.qraw-viewer/notebooks
```

## License

See the `LICENSE` file for license terms.

