from pathlib import Path

from multimodal_tugdt.synthetic import generate_synthetic_dataset


def test_generate_synthetic_dataset_creates_complete_fixture(tmp_path: Path) -> None:
    dataset = generate_synthetic_dataset(
        tmp_path / "data/synthetic",
        participants=2,
        project_root=tmp_path,
    )

    assert dataset.trial_count == 4
    assert dataset.manifest.is_file()
    assert dataset.clinical.is_file()
    assert len(list(dataset.root.glob("P*/S01/T*/imu.csv"))) == 4
    assert len(list(dataset.root.glob("P*/S01/T*/audio.wav"))) == 4
    assert "not physiologically or clinically valid" in (dataset.root / "README.md").read_text()


def test_generate_synthetic_dataset_is_deterministic(tmp_path: Path) -> None:
    first = generate_synthetic_dataset(tmp_path / "first", seed=7)
    second = generate_synthetic_dataset(tmp_path / "second", seed=7)

    first_imu = next(first.root.glob("P*/S01/T01/imu.csv")).read_bytes()
    second_imu = next(second.root.glob("P*/S01/T01/imu.csv")).read_bytes()
    assert first_imu == second_imu
