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

Install latest version directly from GitHub repository:

```bash
pip install git+https://github.com/spice-projects/qspice-qraw-viewer.git
```

After installation you can execute the application:

```bash
qspice-qraw-viewer
```

Note: ensure the directory containing pip-installed console scripts is on your `PATH` so the `qspice-qraw-viewer` command is discoverable. To locate the scripts directory programmatically use:

```bash
python -c "import sysconfig; print(sysconfig.get_path('scripts'))"
```

Examples:

- Unix-like systems (common user installs): `$HOME/.local/bin` — add with:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

- Windows (example locations): `%APPDATA%\Python\Python312\Scripts` or the `Scripts` folder under your Python installation (e.g. `C:\Program Files\Python312\Scripts`). To make a scripts directory discoverable in the current PowerShell session:

```powershell
$env:PATH = "$env:PATH;C:\Path\To\Scripts"
```

Adjust the concrete path according to the output of the `python -c ...` command above.

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

