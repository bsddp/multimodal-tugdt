from pathlib import Path

from multimodal_tugdt.cli import main


def test_cli_generates_then_validates_demo(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "example.yaml").write_text(
        """
project:
  root: ".."
paths:
  manifest: data/synthetic/participants.csv
study:
  allowed_conditions: [single_task, dual_task]
""",
        encoding="utf-8",
    )

    assert main(["generate-synthetic", "--output", "data/synthetic"]) == 0
    assert main(["validate-manifest", "--config", "configs/example.yaml"]) == 0


def test_cli_returns_nonzero_for_missing_config(tmp_path: Path) -> None:
    result = main(["validate-manifest", "--config", str(tmp_path / "missing.yaml")])
    assert result == 2

