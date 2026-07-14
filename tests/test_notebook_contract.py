from pathlib import Path

import nbformat


def test_committed_demo_notebook_is_executed_and_reader_facing() -> None:
    path = Path("notebooks/01_synthetic_workflow.ipynb")
    notebook = nbformat.read(path, as_version=4)
    markdown = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
    for heading in ("## Goal", "## Setup", "## Steps", "## Checks", "## Next Steps"):
        assert heading in markdown

    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code_cells
    assert all(cell.execution_count is not None for cell in code_cells)
    assert not any(output.output_type == "error" for cell in code_cells for output in cell.outputs)
    assert "/Users/" not in str(notebook)


def test_notebook_builder_uses_nbformat() -> None:
    source = Path("scripts/build_demo_notebook.py").read_text(encoding="utf-8")
    assert "import nbformat" in source
    assert "nbf.write" in source
