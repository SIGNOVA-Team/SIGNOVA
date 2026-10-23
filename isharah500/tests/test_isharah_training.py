"""Focused tests for Phase 3A training infrastructure."""

import random
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

SOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_DIRECTORY))

from isharah_ctc import BiGRUCTC  # noqa: E402
from isharah_dataset import PreprocessingConfig, build_training_vocabulary  # noqa: E402
from isharah_training import (  # noqa: E402
    TrainingConfig,
    build_si_datasets,
    build_si_train_dev_datasets,
    capture_random_states,
    ctc_minimum_input_length,
    load_checkpoint,
    locate_ml_archive,
    resolve_data_root,
    restore_random_states,
    run_si_preflight,
    save_checkpoint,
    select_ctc_subset,
    set_reproducible_seed,
    validate_ctc_lengths,
)


def write_metadata_and_keypoints(tmp_path):
    project_root = tmp_path / "project"
    metadata_dir = project_root / "isharah500" / "metadata"
    data_root = project_root
    keypoint_dir = data_root / "data" / "processed"
    metadata_dir.mkdir(parents=True)
    keypoint_dir.mkdir(parents=True)
    rows = {
        "train": [("s1", "alpha beta")],
        "dev": [("s2", "alpha")],
        "test": [("s3", "beta")],
    }
    for split, split_rows in rows.items():
        lines = ["sample_id,gloss,text,keypoints_path"]
        for sample_id, gloss in split_rows:
            relative_path = f"data/processed/{sample_id}.npy"
            lines.append(f"{sample_id},{gloss},text,{relative_path}")
            np.save(keypoint_dir / f"{sample_id}.npy", np.ones((4, 2, 63), dtype=np.float32))
        (metadata_dir / f"isharah500_si_{split}.csv").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
    return project_root, data_root


def test_real_data_ingestion_rejects_missing_files(tmp_path):
    project_root, data_root = write_metadata_and_keypoints(tmp_path)
    (data_root / "data" / "processed" / "s2.npy").unlink()

    train_set, dev_set, _, _ = build_si_datasets(project_root, data_root)
    assert train_set[0]["keypoints"].shape == (4, 2, 63)
    with pytest.raises(FileNotFoundError):
        _ = dev_set[0]


def test_train_dev_setup_does_not_require_si_test_metadata(tmp_path):
    """Ordinary train/dev setup must not open or inspect the SI test split."""
    project_root, data_root = write_metadata_and_keypoints(tmp_path)
    test_metadata = project_root / "isharah500" / "metadata" / "isharah500_si_test.csv"
    test_metadata.unlink()

    train_set, dev_set, vocabulary = build_si_train_dev_datasets(project_root, data_root)

    assert len(train_set) == 1
    assert len(dev_set) == 1
    assert "alpha" in vocabulary.token2id


def test_archive_locator_does_not_extract(tmp_path):
    archive = tmp_path / "isharah500_ml_data.zip"
    archive.write_bytes(b"archive placeholder")

    assert locate_ml_archive(search_roots=[tmp_path]) == archive.resolve()
    with pytest.raises(FileNotFoundError, match="not extracted"):
        resolve_data_root(search_roots=[tmp_path])


def test_checkpoint_round_trip(tmp_path):
    set_reproducible_seed(3)
    project_root, _ = write_metadata_and_keypoints(tmp_path)
    vocabulary = build_training_vocabulary(
        project_root / "isharah500" / "metadata" / "isharah500_si_train.csv"
    )
    model = BiGRUCTC(len(vocabulary))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = TrainingConfig(project_root=str(tmp_path))
    preprocessing = PreprocessingConfig(missing_file_policy="error")
    checkpoint_path = save_checkpoint(
        tmp_path / "checkpoint.pt",
        model,
        optimizer,
        vocabulary,
        preprocessing,
        config,
        epoch=2,
        train_metrics={"ctc_loss": 1.0},
        validation_metrics={"ctc_loss": 1.2},
    )

    loaded_model, loaded_vocab, loaded_preprocessing, payload = load_checkpoint(
        checkpoint_path,
    )
    assert loaded_model.vocabulary_size == len(vocabulary)
    assert loaded_vocab.token2id == vocabulary.token2id
    assert loaded_preprocessing == preprocessing
    assert payload["epoch"] == 2
    assert "optimizer_state_dict" in payload
    assert "best_validation_loss" in payload
    assert "epochs_without_improvement" in payload
    assert "random_states" in payload
    assert torch.is_tensor(payload["random_states"]["numpy"]["keys"])
    assert payload["preprocessing_config"] == preprocessing.to_dict()

    # Anchor expected values to the serialized state, not to the state after
    # load_checkpoint's model construction consumed CPU RNG values.
    restore_random_states(payload["random_states"])
    expected = (random.random(), np.random.random(), torch.rand(3))

    # Loading again deliberately consumes RNG during model initialization.
    load_checkpoint(checkpoint_path)
    restore_random_states(payload["random_states"])
    actual = (random.random(), np.random.random(), torch.rand(3))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])

    if torch.cuda.is_available():
        restore_random_states(payload["random_states"])
        expected_cuda = torch.rand(3, device="cuda")
        torch.rand(3, device="cuda")
        restore_random_states(payload["random_states"])
        actual_cuda = torch.rand(3, device="cuda")
        assert torch.equal(actual_cuda, expected_cuda)


def test_random_states_round_trip():
    set_reproducible_seed(17)
    states = capture_random_states()
    expected = (random.random(), np.random.random(), torch.rand(3))
    expected_cuda = None
    if torch.cuda.is_available():
        expected_cuda = torch.rand(3, device="cuda")

    restore_random_states(states)
    actual = (random.random(), np.random.random(), torch.rand(3))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
    if expected_cuda is not None:
        actual_cuda = torch.rand(3, device="cuda")
        assert torch.equal(actual_cuda, expected_cuda)


def test_training_config_supports_resume_path():
    config = TrainingConfig(
        project_root="project",
        epochs=10,
        resume_from="checkpoint.pt",
    )

    assert config.to_dict()["resume_from"] == "checkpoint.pt"


def test_target_length_weighting_config_is_optional_and_serializable():
    standard = TrainingConfig(project_root="project")
    assert standard.to_dict()["ctc_target_length_weighting"] == "none"

    weighted = TrainingConfig(
        project_root="project",
        output_dir="isharah500/checkpoints/experiments/target_length_weighted_ctc_15ep",
        ctc_target_length_weighting="linear",
        ctc_target_length_weight_cap=2.0,
    )
    assert weighted.to_dict()["ctc_target_length_weighting"] == "linear"
    assert weighted.to_dict()["ctc_target_length_weight_cap"] == 2.0

    with pytest.raises(ValueError, match="ctc_target_length_weighting"):
        TrainingConfig(project_root="project", ctc_target_length_weighting="bad")
    with pytest.raises(ValueError, match="weight_cap"):
        TrainingConfig(project_root="project", ctc_target_length_weight_cap=0.0)


def test_ctc_length_validation_accounts_for_repeated_targets(tmp_path):
    """Repeated adjacent targets require an extra CTC input frame."""
    project_root, data_root = write_metadata_and_keypoints(tmp_path)
    train_csv = project_root / "isharah500" / "metadata" / "isharah500_si_train.csv"
    train_csv.write_text(
        train_csv.read_text(encoding="utf-8").replace("alpha beta", "alpha alpha"),
        encoding="utf-8",
    )
    np.save(
        data_root / "data" / "processed" / "s1.npy",
        np.ones((2, 2, 63), dtype=np.float32),
    )

    _, _, _, vocabulary = build_si_datasets(project_root, data_root)
    assert ctc_minimum_input_length(vocabulary.encode("alpha alpha")) == 3
    train_set, _, _, _ = build_si_datasets(project_root, data_root)
    report = validate_ctc_lengths(train_set)

    assert report["total_samples"] == 1
    assert report["valid_samples"] == 0
    assert report["rejected_samples"][0]["sample_id"] == "s1"
    assert "below CTC minimum" in report["rejected_samples"][0]["reason"]
    with pytest.raises(ValueError, match="CTC validation rejected"):
        select_ctc_subset(train_set, report, allow_rejected=False)
    rejected_subset = select_ctc_subset(train_set, report, allow_rejected=True)
    assert len(rejected_subset) == 0


def test_training_config_exposes_full_run_controls():
    """Full-run controls are serializable and validation-safe."""
    config = TrainingConfig(
        project_root="project",
        epochs=20,
        batch_size=4,
        learning_rate=5e-4,
        early_stopping_patience=3,
        checkpoint_every=2,
        reject_invalid_ctc_samples=True,
    )
    values = config.to_dict()
    assert values["epochs"] == 20
    assert values["early_stopping_patience"] == 3
    assert values["checkpoint_every"] == 2
    assert values["reject_invalid_ctc_samples"] is True


def test_training_config_exposes_coordinate_normalization():
    config = TrainingConfig(
        project_root="project",
        coordinate_normalization="hand_centric",
    )
    assert config.to_dict()["coordinate_normalization"] == "hand_centric"


def test_si_preflight_does_not_require_test_metadata(tmp_path):
    """Read-only model-selection preflight validates train/dev only."""
    project_root, data_root = write_metadata_and_keypoints(tmp_path)
    (project_root / "isharah500" / "metadata" / "isharah500_si_test.csv").unlink()

    report = run_si_preflight(project_root, data_root)

    assert report["validated_splits"] == ["train", "dev"]
    assert report["total_samples"] == 2
    assert report["valid_samples"] == 2
    assert report["invalid_samples"] == 0
