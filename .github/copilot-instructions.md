# GitHub Copilot Instructions

## Project Overview

QSPICE QRAW Waveform Viewer — a PySide6/QML desktop app that parses binary `.qraw` simulation output files from QSPICE and renders waveform data using `QLineSeries.replaceNp` (zero-copy numpy → Qt). Requires Python ≥ 3.12 and PySide6 ≥ 6.7.

## Build and Test

```bash
# Install dependencies
pip install numpy "PySide6>=6.7"

# Run the app
python -m viewer <file.qraw>
python -m viewer <file.qraw> --log-level=DEBUG
python -m viewer <file.qraw> --headless   # parse without UI

# Run tests
python -m unittest discover -p '*_test.py'

# Compile .ui files (VS Code task: "Compile UI files")
for f in viewer/*.ui; do .venv/bin/pyside6-uic "$f" -o "viewer/ui_$(basename ${f%.ui}).py"; done
```

## Architecture

| Module | Role |
|---|---|
| `viewer/qraw_file.py` | Binary parser → `QRawFile`; module-level singletons `TRANSIENT`, `AC`, `DC` (`ChartTypeSpec`) |
| `viewer/variable.py` | `Variable` — numpy view into mmap buffer; lazy `.magnitude`/`.phase` for complex data |
| `viewer/chart.py` | `Chart` — wraps a QML `QQuickItem`; calls `decimate_xy` then `QLineSeries.replaceNp` |
| `viewer/main_window.py` | `QMainWindow` subclass; hosts `QQuickView`; owns all `Chart` objects; manages zoom state |
| `viewer/expression_{node,parser,evaluator}.py` | Stateless recursive-descent parser + AST walker → `(np.ndarray, unit_str)` |
| `viewer/decimation_algorithm.py` | Six downsampling algorithms; M4 is the default |
| `viewer/fft.py` | Pure numpy FFT; pluggable window registry |
| `viewer/fft_dialog.py` | FFT params `QDialog`; result opens a second `MainWindow` with a synthetic `QRawFile` |
| `viewer/add_plot_dialog.py` | Variable-selection `QDialog`; sole user of `setContextProperty` |

**Data flow:** `.qraw` → `QRawFile.load()` (mmap + `np.frombuffer`) → `Variable[]` → `MainWindow._populate_charts` → `Chart.render()` → `decimate_xy` → `QLineSeries.replaceNp`.

## Code Style

Follow [STYLE_GUIDE.md](../STYLE_GUIDE.md) exactly. Key rules:

- **Indentation:** 4 spaces
- **Quotes:** double quotes everywhere
- **Imports:** three sections (stdlib → third-party → local), blank line between each; `import` before `from ... import` within each section; one import per line; alphabetical
- **Naming:** `snake_case` for functions and variables, `PascalCase` for classes
- **Comments:** placed on the line *above* the code they describe; start with a lowercase letter; no trailing period; every non-trivial statement gets its own comment
- **Blank lines:** two blank lines between top-level definitions; no blank lines *inside* function bodies (comments are the only visual separators)
- **Function calls:** always on a single line — never multiline call syntax
- **Type hints:** use `X | Y` union syntax (Python 3.12+), not `Optional[X]`; annotate all parameters and return types
- **Properties:** use private `_field` storage with a `@property` for public read-only access (see `QRawFile`, `Variable`)
- **Logging:** `logger = logging.getLogger(__name__)` at module level; `%`-style lazy formatting (never f-strings in log calls)
- **Error returns:** `load()`-style constructors return `None` on failure; expression parser/evaluator raise `ValueError`

## QML–Python Bridge

Two patterns — do not mix them within a single dialog:

**Pattern A — context property** (only `AddPlotDialog`): call `rootContext().setContextProperty(key, value)` before `setSource`. Data is available during QML binding.

**Pattern B — property injection after ready** (`MainWindow`, `FftDialog`): connect `statusChanged` → `_on_qml_ready`; in the slot, call `root.setProperty(key, value)` and connect QML signals.

QML signals are declared with explicit types (`signal zoom(int idx, real ratio)`). Python slots use matching `@Slot(int, float)` decorators. Python calls QML functions by name on the root object (`self._root.addChart()`).

## Testing

- Framework: `unittest.TestCase` only — no pytest, no setUp/tearDown, no class fixtures
- Every test is fully self-contained; all test data defined inside the test method
- Required comment markers: `# arrange`, `# act`, `# assert` — no blank lines between sections
- Qt mocking: mock `PySide6` submodules via `sys.modules` before importing viewer modules; `Slot` must be a pass-through decorator; `QMainWindow` must be a real inheritable class
- Bypass `__init__` for heavy Qt classes: `win = MainWindow.__new__(MainWindow)`, then set only the attributes the method under test needs
- Fixture files live in `tests/PyQSPICE/`; reference them via `FIXTURES_DIR = Path(__file__).parent / "PyQSPICE"`
- Array assertions: `np.testing.assert_array_equal` / `np.testing.assert_array_almost_equal`
- Test files named `<module>_test.py`; test class named `Test<ClassUnderTest>`

## Project Conventions

- **Stepped simulations:** period detected with `np.argmax(np.isclose(...))` in one pass; `Variable.step_values(n)` returns a zero-copy slice
- **Zoom state:** stored as `(from_index: int, y_top_ratio: float, to_index: int, y_bottom_ratio: float)`; horizontal zoom is global (owned by `MainWindow`), vertical zoom is per-`Chart`
- **FFT result:** a synthetic `QRawFile` is constructed and passed to a new `MainWindow` — no separate rendering path
- **Decimation config:** `_DECIMATION_ALGORITHM` constant in `viewer/chart.py` is the single change point
- **Series color cycling:** monotonic `seriesCounter` that never resets; 8-color palette defined in Python and mirrored in QML
- **mmap lifetime:** `QRawFile._mmap` must stay alive as long as any `Variable` is referenced (all `.values` arrays are zero-copy views)
- **Status bar throttle:** 30 fps cap via `_MIN_STATUS_INTERVAL = 1.0 / 30`
