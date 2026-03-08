import json
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from viewer.notebook_template import _make_notebook, ensure_starter_notebook


class TestMakeNotebook(TestCase):

    def test_nbformat_version_is_4(self):
        # arrange
        qraw_path = Path("/sim/buck.qraw")
        # act
        nb = _make_notebook(qraw_path)
        # assert
        self.assertEqual(nb["nbformat"], 4)

    def test_has_four_cells(self):
        # arrange
        qraw_path = Path("/sim/buck.qraw")
        # act
        nb = _make_notebook(qraw_path)
        # assert
        self.assertEqual(len(nb["cells"]), 4)

    def test_first_cell_is_markdown(self):
        # arrange
        qraw_path = Path("/sim/buck.qraw")
        # act
        nb = _make_notebook(qraw_path)
        # assert
        self.assertEqual(nb["cells"][0]["cell_type"], "markdown")

    def test_remaining_cells_are_code(self):
        # arrange
        qraw_path = Path("/sim/buck.qraw")
        # act
        nb = _make_notebook(qraw_path)
        # assert
        for cell in nb["cells"][1:]:
            self.assertEqual(cell["cell_type"], "code")

    def test_markdown_cell_contains_stem(self):
        # arrange
        qraw_path = Path("/sim/my_simulation.qraw")
        # act
        nb = _make_notebook(qraw_path)
        # assert — stem name appears in the markdown header for identification
        source = "".join(nb["cells"][0]["source"])
        self.assertIn("my_simulation", source)

    def test_load_cell_embeds_qraw_path(self):
        # arrange
        qraw_path = Path("/sim/test.qraw")
        # act
        nb = _make_notebook(qraw_path)
        # assert — resolved path string must appear in the load cell source
        source = "".join(nb["cells"][1]["source"])
        self.assertIn(str(qraw_path.resolve()), source)

    def test_load_cell_contains_sys_path_injection(self):
        # arrange
        qraw_path = Path("/sim/test.qraw")
        # act
        nb = _make_notebook(qraw_path)
        # assert — project root must be injected so live source takes priority
        source = "".join(nb["cells"][1]["source"])
        self.assertIn("sys.path.insert", source)

    def test_load_cell_imports_qraw_file(self):
        # arrange
        qraw_path = Path("/sim/test.qraw")
        # act
        nb = _make_notebook(qraw_path)
        # assert
        source = "".join(nb["cells"][1]["source"])
        self.assertIn("from viewer.qraw_file import QRawFile", source)

    def test_each_cell_has_unique_id(self):
        # arrange
        qraw_path = Path("/sim/test.qraw")
        # act
        nb = _make_notebook(qraw_path)
        # assert — cell ids must all be distinct
        ids = [cell["id"] for cell in nb["cells"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_kernelspec_is_python3(self):
        # arrange
        qraw_path = Path("/sim/test.qraw")
        # act
        nb = _make_notebook(qraw_path)
        # assert
        self.assertEqual(nb["metadata"]["kernelspec"]["name"], "python3")


class TestEnsureStarterNotebook(TestCase):

    def test_creates_notebook_when_absent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # arrange
            notebooks_dir = Path(tmpdir) / "notebooks"
            qraw_path = Path("/sim/test.qraw")
            # act
            with patch("viewer.notebook_template.NOTEBOOKS_DIR", notebooks_dir):
                result = ensure_starter_notebook(qraw_path)
            # assert
            self.assertTrue(result.exists())

    def test_returned_path_uses_qraw_stem(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # arrange
            notebooks_dir = Path(tmpdir) / "notebooks"
            qraw_path = Path("/sim/my_circuit.qraw")
            # act
            with patch("viewer.notebook_template.NOTEBOOKS_DIR", notebooks_dir):
                result = ensure_starter_notebook(qraw_path)
            # assert
            self.assertEqual(result.name, "my_circuit.ipynb")

    def test_returned_path_is_inside_notebooks_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # arrange
            notebooks_dir = Path(tmpdir) / "notebooks"
            qraw_path = Path("/sim/test.qraw")
            # act
            with patch("viewer.notebook_template.NOTEBOOKS_DIR", notebooks_dir):
                result = ensure_starter_notebook(qraw_path)
            # assert
            self.assertEqual(result.parent, notebooks_dir)

    def test_written_file_is_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # arrange
            notebooks_dir = Path(tmpdir) / "notebooks"
            qraw_path = Path("/sim/test.qraw")
            # act
            with patch("viewer.notebook_template.NOTEBOOKS_DIR", notebooks_dir):
                result = ensure_starter_notebook(qraw_path)
            # assert — file must be parseable as JSON with no exception
            content = json.loads(result.read_text(encoding="utf-8"))
            self.assertIn("cells", content)

    def test_creates_notebooks_directory_if_absent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # arrange
            notebooks_dir = Path(tmpdir) / "deeply" / "nested" / "notebooks"
            qraw_path = Path("/sim/test.qraw")
            # act
            with patch("viewer.notebook_template.NOTEBOOKS_DIR", notebooks_dir):
                ensure_starter_notebook(qraw_path)
            # assert
            self.assertTrue(notebooks_dir.is_dir())

    def test_does_not_overwrite_existing_notebook(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # arrange — pre-write a sentinel file at the expected path
            notebooks_dir = Path(tmpdir) / "notebooks"
            notebooks_dir.mkdir(parents=True)
            existing = notebooks_dir / "test.ipynb"
            sentinel = "user work must not be deleted"
            existing.write_text(sentinel, encoding="utf-8")
            qraw_path = Path("/sim/test.qraw")
            # act
            with patch("viewer.notebook_template.NOTEBOOKS_DIR", notebooks_dir):
                ensure_starter_notebook(qraw_path)
            # assert — sentinel content is preserved
            self.assertEqual(existing.read_text(encoding="utf-8"), sentinel)

    def test_returns_existing_path_without_modification(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # arrange
            notebooks_dir = Path(tmpdir) / "notebooks"
            notebooks_dir.mkdir(parents=True)
            existing = notebooks_dir / "test.ipynb"
            existing.write_text("{}", encoding="utf-8")
            qraw_path = Path("/sim/test.qraw")
            # act
            with patch("viewer.notebook_template.NOTEBOOKS_DIR", notebooks_dir):
                result = ensure_starter_notebook(qraw_path)
            # assert
            self.assertEqual(result, existing)
