from pathlib import Path

import pytest

from multimodal_tugdt.config import ConfigurationError, load_config


def test_load_config_resolves_paths_from_declared_root(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "study.yaml"
    config_path.write_text(
        """
project:
  root: ".."
paths:
  manifest: data/participants.csv
  output_dir: results
study:
  allowed_conditions: [single_task, dual_task]
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.project_root == tmp_path.resolve()
    assert config.manifest_path == (tmp_path / "data/participants.csv").resolve()
    assert config.output_dir == (tmp_path / "results").resolve()
    assert config.allowed_conditions == ("single_task", "dual_task")


def test_load_config_requires_manifest_path(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("project: {}\npaths: {}\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="paths.manifest"):
        load_config(config_path)


def test_pose_estimation_requires_explicit_model_path(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_video.yaml"
    config_path.write_text(
        """
paths:
  manifest: participants.csv
video:
  enable_pose_estimation: true
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="pose_model_path"):
        load_config(config_path)


def test_modeling_rejects_nonparticipant_grouping(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_modeling.yaml"
    config_path.write_text(
        """
paths:
  manifest: participants.csv
modeling:
  group_column: trial_id
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="participant_id"):
        load_config(config_path)


def test_classification_requires_explicit_positive_label(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_classification.yaml"
    config_path.write_text(
        """
paths:
  manifest: participants.csv
modeling:
  task_type: classification
  models: [logistic_regression]
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="positive_label"):
        load_config(config_path)
