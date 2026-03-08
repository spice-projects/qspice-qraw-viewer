import json
import uuid
from pathlib import Path

# application data directory under the user home directory
_APP_DIR = Path.home() / ".qraw-viewer"

# absolute path to the project root (two levels up from this file: viewer/ → project root)
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# directory where starter notebooks are written
NOTEBOOKS_DIR = _APP_DIR / "notebooks"


def _cell_id() -> str:
    # generate a short random UUID-based ID matching the nbformat cell id convention
    return uuid.uuid4().hex[:8]


def _make_notebook(qraw_path: Path) -> dict:
    # absolute resolved path string to embed in the load cell
    path_str = str(qraw_path.resolve())
    # simulation name used in the markdown title
    stem = qraw_path.stem
    # assemble the full notebook structure as a plain dict
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "cells": [
            {
                "cell_type": "markdown",
                "id": _cell_id(),
                "metadata": {},
                "source": [
                    f"# QSPICE Analysis — {stem}\n",
                    "\n",
                    f"Simulation file: `{path_str}`\n",
                    "\n",
                    "## Available helpers after loading\n",
                    "- `qraw.expression_manager.expressions` — all signals in the file\n",
                    "- `qraw.expression_manager.evaluate('V(out)')` — evaluate a derived expression\n",
                    "- `qraw.abscissa` — x-axis data (time / frequency / swept parameter)\n",
                    "- `qraw.steps` — number of simulation steps",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": _cell_id(),
                "metadata": {},
                "outputs": [],
                "source": [
                    "import sys\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "# ensure the live project source takes priority over any installed copy\n",
                    f"_project_root = {str(_PROJECT_ROOT)!r}\n",
                    "if _project_root not in sys.path:\n",
                    "    sys.path.insert(0, _project_root)\n",
                    "from viewer.qraw_file import QRawFile\n",
                    "\n",
                    f"qraw = QRawFile.load({path_str!r})\n",
                    "print(f'File   : {qraw.filename.name}')\n",
                    "print(f'Type   : {qraw.plotname}')\n",
                    "print(f'Steps  : {qraw.steps}')\n",
                    "print(f'X-axis : {qraw.abscissa.name} ({qraw.abscissa.unit}),"
                    " {len(qraw.abscissa.data)} points')",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": _cell_id(),
                "metadata": {},
                "outputs": [],
                "source": [
                    "# list all available signals\n",
                    "for expr in qraw.expression_manager.expressions:\n",
                    "    print(f'{expr.name:40s}  [{expr.unit}]')",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": _cell_id(),
                "metadata": {},
                "outputs": [],
                "source": [
                    "# example: plot a signal\n",
                    "# replace 'V(out)' with any name from the list above\n",
                    "# expr = qraw.expression_manager.evaluate('V(out)')\n",
                    "# plt.figure()\n",
                    "# plt.plot(qraw.abscissa.data, np.real(expr.data))\n",
                    "# plt.xlabel(f'{qraw.abscissa.name} ({qraw.abscissa.unit})')\n",
                    "# plt.ylabel(f'{expr.name} ({expr.unit})')\n",
                    "# plt.grid(True)\n",
                    "# plt.tight_layout()\n",
                    "# plt.show()",
                ],
            },
        ],
    }


def ensure_starter_notebook(qraw_path: Path) -> Path:
    # create the notebooks directory if it does not already exist
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    # notebook path is named after the qraw file stem
    notebook_path = NOTEBOOKS_DIR / f"{qraw_path.stem}.ipynb"
    # only write the template when the file is absent — never overwrite the user's work
    if not notebook_path.exists():
        notebook_data = _make_notebook(qraw_path)
        notebook_path.write_text(json.dumps(notebook_data, indent=1), encoding="utf-8")
    # exit
    return notebook_path
