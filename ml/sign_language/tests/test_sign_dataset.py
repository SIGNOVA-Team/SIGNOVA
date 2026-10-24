"""
Unit tests for Dataset and DataLoader modules in SIGNOVA.
"""

from pathlib import Path

from ml.sign_language.src.data_loader import create_dataloader
from ml.sign_language.src.dataset import SignLanguageDataset
from ml.sign_language.src.preprocessing import SignVocabulary


def test_sign_language_dataset_arabsign():
    root = Path(__file__).resolve().parents[3]
    metadata_path = root / "data/metadata/canonical/arabsign_canonical_metadata.csv"
    manifest_path = root / "data/metadata/splits/arabsign_split_manifest_signer_independent.csv"

    if not metadata_path.exists() or not manifest_path.exists():
        return

    vocab = SignVocabulary(language="ArSL")
    vocab.build_from_texts(["مرحبا", "شكرا"])

    dataset = SignLanguageDataset(
        language="ArSL",
        canonical_csv=metadata_path,
        split_manifest_csv=manifest_path,
        split="test",
        vocabulary=vocab,
        max_samples=20,
    )

    assert len(dataset) > 0
    sample = dataset[0]
    assert "sample_id" in sample
    assert "keypoints" in sample
    assert "token_ids" in sample
    assert "text" in sample
    # Check shape: (T, 126)
    assert sample["keypoints"].ndim == 2
    assert sample["keypoints"].shape[1] == 126


def test_dataloader_collate():
    root = Path(__file__).resolve().parents[3]
    metadata_path = root / "data/metadata/canonical/arabsign_canonical_metadata.csv"
    manifest_path = root / "data/metadata/splits/arabsign_split_manifest_signer_independent.csv"

    if not metadata_path.exists():
        return

    vocab = SignVocabulary(language="ArSL")
    vocab.build_from_texts(["مرحبا", "شكرا"])

    dataset = SignLanguageDataset(
        language="ArSL",
        canonical_csv=metadata_path,
        split_manifest_csv=manifest_path,
        split="test",
        vocabulary=vocab,
        max_samples=8,
    )

    loader = create_dataloader(
        dataset=dataset,
        batch_size=4,
        shuffle=False,
    )

    batch = next(iter(loader))
    assert "keypoints" in batch
    assert "lengths" in batch
    assert "token_ids" in batch
    assert "token_lengths" in batch
    assert batch["keypoints"].shape[0] == 4
    assert batch["keypoints"].shape[2] == 126
