from pathlib import Path

import matplotlib.pyplot as plt
import yaml


def test_citation_metadata_matches_project_version() -> None:
    citation = yaml.safe_load(Path("CITATION.cff").read_text(encoding="utf-8"))
    project = Path("pyproject.toml").read_text(encoding="utf-8")

    assert citation["title"] == "Multimodal TUG-DT Analysis Pipeline"
    assert citation["version"] == "0.8.0"
    assert citation["license"] == "MIT"
    assert 'version = "0.8.0"' in project


def test_public_example_report_has_no_individual_identifiers_or_paths() -> None:
    report = Path("docs/example_outputs/synthetic_research_summary.md").read_text(encoding="utf-8")

    assert "| Participants | 1 |" in report
    assert "| Trials | 2 |" in report
    assert "Baseline modeling is disabled" in report
    assert "P001" not in report
    assert "/Users/" not in report


def test_readme_hero_is_committed_high_resolution_synthetic_output() -> None:
    image_path = Path("docs/assets/readme_pipeline_demo.png")
    image = plt.imread(image_path)
    readme = Path("README.md").read_text(encoding="utf-8")

    assert image.shape[1] >= 2000
    assert image.shape[0] >= 1000
    assert "docs/assets/readme_pipeline_demo.png" in readme
    assert "Synthetic" in readme
