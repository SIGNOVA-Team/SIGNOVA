"""Reproducible SI training pipeline for the Isharah-500 BiGRU + CTC baseline."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset

from isharah_ctc import BiGRUCTC, BiGRUCTCConfig, greedy_ctc_decode
from isharah_dataset import (
    GlossVocabulary,
    IsharahDataset,
    PreprocessingConfig,
    build_training_vocabulary,
    collate_isharah_ctc_batch,
)
from isharah_features import (
    FEATURE_REPRESENTATION_GLOBAL_POSITION_V1,
    FEATURE_REPRESENTATION_RAW_63,
)
from isharah_metrics import calculate_token_metrics


@dataclass(frozen=True)
class TrainingConfig:
    """Serializable training settings for SI smoke tests and full runs."""

    project_root: str
    data_root: str | None = None
    archive_path: str | None = None
    output_dir: str = "isharah500/checkpoints"
    protocol: str = "SI"
    batch_size: int = 8
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 5.0
    seed: int = 42
    num_workers: int = 0
    smoke_samples: int = 4
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0
    checkpoint_every: int = 1
    reject_invalid_ctc_samples: bool = False
    coordinate_normalization: str = "none"
    feature_representation: str = FEATURE_REPRESENTATION_RAW_63
    global_coordinate_scale: float = 1.0
    ctc_target_length_weighting: str = "none"
    ctc_target_length_weight_cap: float = 2.0
    resume_from: str | None = None
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.protocol.upper() != "SI":
            raise ValueError("Phase 3A currently supports the SI protocol only")
        if self.batch_size <= 0 or self.epochs <= 0 or self.num_workers < 0:
            raise ValueError("batch_size, epochs, and num_workers are invalid")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.gradient_clip <= 0:
            raise ValueError("Optimizer settings must be positive")
        if self.early_stopping_patience <= 0 or self.early_stopping_min_delta < 0:
            raise ValueError("Early stopping settings are invalid")
        if self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive")
        if self.ctc_target_length_weighting not in {"none", "linear", "sqrt"}:
            raise ValueError(
                "ctc_target_length_weighting must be one of: none, linear, sqrt"
            )
        if self.ctc_target_length_weight_cap <= 0:
            raise ValueError("ctc_target_length_weight_cap must be positive")
        if self.feature_representation not in {
            FEATURE_REPRESENTATION_RAW_63,
            FEATURE_REPRESENTATION_GLOBAL_POSITION_V1,
        }:
            raise ValueError(
                "Unsupported feature_representation: "
                f"{self.feature_representation}"
            )
        if self.global_coordinate_scale <= 0:
            raise ValueError("global_coordinate_scale must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_cross_platform_path(value: str | Path) -> Path:
    """Normalize Windows paths under WSL and WSL paths under Windows."""
    raw = os.path.expandvars(str(value).strip().strip('"'))
    windows_drive = re.match(r"^([A-Za-z]):[\\/](.*)$", raw)
    if os.name != "nt" and windows_drive:
        drive, remainder = windows_drive.groups()
        return Path("/mnt", drive.lower(), *re.split(r"[\\/]", remainder))
    if os.name == "nt" and raw.lower().startswith("/mnt/"):
        parts = raw.replace("\\", "/").split("/")
        if len(parts) >= 4 and len(parts[2]) == 1:
            return Path(f"{parts[2].upper()}:\\", *parts[3:])
    return Path(raw).expanduser()


def locate_ml_archive(
    archive_path: str | Path | None = None,
    search_roots: list[str | Path] | None = None,
) -> Path | None:
    """Locate the existing archive without copying or extracting it."""
    if archive_path is not None:
        candidate = _normalize_cross_platform_path(archive_path)
        if candidate.is_file() and candidate.name == "isharah500_ml_data.zip":
            return candidate.resolve()
        raise FileNotFoundError(f"Isharah archive not found: {candidate}")

    roots = search_roots or [Path.cwd(), Path(__file__).resolve().parents[2]]
    checked: set[Path] = set()
    for root_value in roots:
        root = _normalize_cross_platform_path(root_value).resolve()
        if root in checked or not root.exists():
            continue
        checked.add(root)
        direct = root / "isharah500_ml_data.zip"
        if direct.is_file():
            return direct
        for candidate in root.glob("**/isharah500_ml_data.zip"):
            if candidate.is_file():
                return candidate.resolve()
    return None


def resolve_data_root(
    data_root: str | Path | None = None,
    archive_path: str | Path | None = None,
    *,
    search_roots: list[str | Path] | None = None,
) -> Path:
    """Resolve an extracted data directory and refuse archive-only training."""
    if data_root is not None:
        root = _normalize_cross_platform_path(data_root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Isharah data root is not a directory: {root}")
        return root

    archive = locate_ml_archive(archive_path, search_roots)
    if archive is not None:
        raise FileNotFoundError(
            f"Found archive at {archive}, but it is not extracted. "
            "Provide an extracted --data-root; the pipeline will not extract automatically."
        )
    raise FileNotFoundError(
        "No extracted Isharah data root or isharah500_ml_data.zip was found. "
        "Provide --data-root or --archive-path."
    )


def set_reproducible_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch without starting any training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def capture_random_states() -> dict[str, Any]:
    """Capture process RNG state for reproducible checkpoint continuation."""
    numpy_algorithm, numpy_keys, numpy_position, has_gauss, cached_gaussian = (
        np.random.get_state()
    )
    states: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "algorithm": str(numpy_algorithm),
            "keys": torch.as_tensor(numpy_keys, dtype=torch.uint32).clone(),
            "position": int(numpy_position),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached_gaussian),
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["torch_cuda"] = torch.cuda.get_rng_state_all()
    return states


def restore_random_states(states: dict[str, Any]) -> None:
    """Restore RNG state saved by ``capture_random_states``."""
    if not isinstance(states, dict):
        raise TypeError("random_states must be a dictionary")
    random.setstate(states["python"])
    numpy_state = states["numpy"]
    if isinstance(numpy_state, tuple):
        # Compatibility with checkpoints written before the safe serialized form.
        np.random.set_state(numpy_state)
    else:
        numpy_keys = numpy_state["keys"]
        if torch.is_tensor(numpy_keys):
            numpy_keys = numpy_keys.detach().cpu().numpy().astype(np.uint32, copy=False)
        np.random.set_state(
            (
                str(numpy_state["algorithm"]),
                numpy_keys,
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
    def as_cpu_byte_tensor(value: Any, name: str) -> torch.Tensor:
        if torch.is_tensor(value):
            if value.dtype != torch.uint8:
                raise TypeError(f"{name} must have dtype torch.uint8")
            result = value.detach().to(device="cpu").contiguous()
        elif isinstance(value, (bytes, bytearray)):
            result = torch.tensor(list(value), dtype=torch.uint8)
        elif isinstance(value, (list, tuple)):
            if not all(isinstance(item, int) and 0 <= item <= 255 for item in value):
                raise TypeError(f"{name} contains non-byte values")
            result = torch.tensor(value, dtype=torch.uint8)
        else:
            raise TypeError(f"{name} must be a torch.ByteTensor or byte sequence")
        if result.ndim != 1:
            raise TypeError(f"{name} must be one-dimensional")
        return result

    torch.set_rng_state(as_cpu_byte_tensor(states["torch"], "torch RNG state"))
    if torch.cuda.is_available():
        if "torch_cuda" not in states:
            raise ValueError("random_states is missing CUDA RNG state")
        cuda_states = [
            as_cpu_byte_tensor(state, f"CUDA RNG state {index}")
            for index, state in enumerate(states["torch_cuda"])
        ]
        torch.cuda.set_rng_state_all(cuda_states)


def select_device(device_name: str) -> torch.device:
    """Select a requested device while preventing accidental CUDA fallback."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def build_si_datasets(
    project_root: str | Path,
    data_root: str | Path,
    preprocessing_config: PreprocessingConfig | None = None,
) -> tuple[IsharahDataset, IsharahDataset, IsharahDataset, GlossVocabulary]:
    """Build strict SI datasets using a vocabulary from SI train only."""
    train_set, dev_set, vocabulary = build_si_train_dev_datasets(
        project_root,
        data_root,
        preprocessing_config,
    )
    project_root = _normalize_cross_platform_path(project_root).resolve()
    data_root = _normalize_cross_platform_path(data_root).resolve()
    metadata_dir = project_root / "isharah500" / "metadata"
    test_csv = metadata_dir / "isharah500_si_test.csv"
    if not test_csv.is_file():
        raise FileNotFoundError(f"SI metadata file not found: {test_csv}")
    test_set = IsharahDataset(
        test_csv,
        root_dir=data_root,
        vocabulary=vocabulary,
        preprocessing_config=train_set.preprocessing_config,
        oov_policy="error",
        strict=True,
    )
    return train_set, dev_set, test_set, vocabulary


def build_si_train_dev_datasets(
    project_root: str | Path,
    data_root: str | Path,
    preprocessing_config: PreprocessingConfig | None = None,
) -> tuple[IsharahDataset, IsharahDataset, GlossVocabulary]:
    """Build only SI train/dev datasets for read-only model-selection preflight."""
    project_root = _normalize_cross_platform_path(project_root).resolve()
    data_root = _normalize_cross_platform_path(data_root).resolve()
    metadata_dir = project_root / "isharah500" / "metadata"
    train_csv = metadata_dir / "isharah500_si_train.csv"
    dev_csv = metadata_dir / "isharah500_si_dev.csv"
    for metadata_path in (train_csv, dev_csv):
        if not metadata_path.is_file():
            raise FileNotFoundError(f"SI metadata file not found: {metadata_path}")

    preprocessing_config = preprocessing_config or PreprocessingConfig(
        nan_strategy="interpolate",
        flatten_hands=False,
        missing_file_policy="error",
    )
    if preprocessing_config.missing_file_policy != "error":
        raise ValueError("Training and evaluation require missing_file_policy='error'")
    vocabulary = build_training_vocabulary(train_csv)
    dataset_args = {
        "root_dir": data_root,
        "vocabulary": vocabulary,
        "preprocessing_config": preprocessing_config,
        "oov_policy": "error",
        "strict": True,
    }
    return (
        IsharahDataset(train_csv, **dataset_args),
        IsharahDataset(dev_csv, **dataset_args),
        vocabulary,
    )


def validate_real_dataset(dataset: IsharahDataset) -> None:
    """Preflight every sample and reject missing or invalid real keypoint files."""
    if dataset.missing_file_policy != "error":
        raise ValueError("Dataset is not configured for strict real-data loading")
    for index in range(len(dataset)):
        dataset[index]


def ctc_minimum_input_length(target_ids: list[int]) -> int:
    """Return the shortest CTC input length for a target sequence."""
    repeated_adjacent_tokens = sum(
        left == right
        for left, right in zip(target_ids, target_ids[1:])
    )
    return len(target_ids) + repeated_adjacent_tokens


def validate_ctc_lengths(dataset: IsharahDataset) -> dict[str, Any]:
    """Validate real dataset samples against CTC input/target length rules."""
    if dataset.missing_file_policy != "error":
        raise ValueError("CTC validation requires strict real-data loading")

    valid_indices: list[int] = []
    rejected_samples: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        sample_id = dataset.samples[index].get("sample_id", f"index_{index}")
        try:
            item = dataset[index]
            target_ids = item["token_ids"]
            if hasattr(target_ids, "detach"):
                target_ids = target_ids.detach().cpu().tolist()
            else:
                target_ids = list(target_ids)
            target_length = int(item["token_length"])
            input_length = int(item["length"])
            minimum_input_length = ctc_minimum_input_length(target_ids)
            if target_length != len(target_ids):
                raise ValueError(
                    f"target_length={target_length} disagrees with "
                    f"target_ids={len(target_ids)}"
                )
            if target_length == 0:
                raise ValueError("empty CTC target")
            if input_length < minimum_input_length:
                raise ValueError(
                    f"input_length={input_length} is below CTC minimum "
                    f"{minimum_input_length} for target_length={target_length}"
                )
        except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
            reason = str(exc)
            reason_lower = reason.lower()
            if "oov" in reason_lower:
                reason_type = "oov_token"
            elif "keypoint file" in reason_lower or "keypoints" in reason_lower:
                reason_type = "keypoint_file_or_value"
            elif "ctc" in reason_lower or "target" in reason_lower:
                reason_type = "ctc_length_or_target"
            else:
                reason_type = "metadata_or_loading"
            rejected_samples.append(
                {
                    "index": index,
                    "sample_id": sample_id,
                    "reason_type": reason_type,
                    "reason": reason,
                    "keypoints_path": dataset.samples[index].get("keypoints_path"),
                }
            )
            continue
        valid_indices.append(index)

    return {
        "total_samples": len(dataset),
        "valid_samples": len(valid_indices),
        "rejected_samples": rejected_samples,
        "valid_indices": valid_indices,
    }


def run_si_preflight(
    project_root: str | Path,
    data_root: str | Path,
) -> dict[str, Any]:
    """Run read-only CTC validation over SI train and dev only."""
    train_set, dev_set, vocabulary = build_si_train_dev_datasets(
        project_root,
        data_root,
        PreprocessingConfig(missing_file_policy="error"),
    )
    split_reports = {
        "train": validate_ctc_lengths(train_set),
        "dev": validate_ctc_lengths(dev_set),
    }
    public_reports: dict[str, Any] = {}
    total_samples = 0
    total_valid = 0
    total_invalid = 0
    for split_name, report in split_reports.items():
        invalid_samples = report["rejected_samples"]
        total_samples += report["total_samples"]
        total_valid += report["valid_samples"]
        total_invalid += len(invalid_samples)
        public_reports[split_name] = {
            "total_samples": report["total_samples"],
            "valid_samples": report["valid_samples"],
            "invalid_samples": len(invalid_samples),
            "rejected_samples": invalid_samples,
        }
    return {
        "protocol": "SI",
        "validated_splits": ["train", "dev"],
        "vocabulary_size": len(vocabulary),
        "total_samples": total_samples,
        "valid_samples": total_valid,
        "invalid_samples": total_invalid,
        "splits": public_reports,
    }


def select_ctc_subset(
    dataset: IsharahDataset,
    validation_report: dict[str, Any],
    *,
    allow_rejected: bool,
) -> Subset:
    """Select valid samples, requiring explicit permission to reject any."""
    rejected = validation_report["rejected_samples"]
    if rejected and not allow_rejected:
        examples = ", ".join(
            f"{item['sample_id']}: {item['reason']}" for item in rejected[:3]
        )
        raise ValueError(
            f"CTC validation rejected {len(rejected)} samples. "
            f"Use reject_invalid_ctc_samples=True only after review. Examples: {examples}"
        )
    return Subset(dataset, validation_report["valid_indices"])


def _write_json(path: str | Path, payload: Any) -> Path:
    output_path = _normalize_cross_platform_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def make_loader(
    dataset: IsharahDataset,
    batch_size: int,
    num_workers: int,
    *,
    shuffle: bool,
) -> DataLoader:
    """Create a CTC loader using the Phase 1 collator."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_isharah_ctc_batch,
    )


def _batch_metrics(
    predictions: list[str],
    target_ids: torch.Tensor,
    target_lengths: torch.Tensor,
    vocabulary: GlossVocabulary,
) -> tuple[int, int, int]:
    offset = 0
    references: list[list[str]] = []
    hypotheses: list[list[str]] = []
    for prediction, length in zip(predictions, target_lengths.tolist(), strict=True):
        target = vocabulary.decode(
            target_ids[offset : offset + length].tolist(),
        )
        offset += length
        references.append(target.split() if target else [])
        hypotheses.append(prediction.split() if prediction else [])
    metrics = calculate_token_metrics(references, hypotheses)
    return (
        metrics["edit_errors"],
        metrics["reference_tokens"],
        metrics["exact_matches"],
    )


def run_epoch(
    model: BiGRUCTC,
    loader: DataLoader,
    vocabulary: GlossVocabulary,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip: float = 5.0,
    ctc_target_length_weighting: str = "none",
    ctc_target_length_weight_cap: float = 2.0,
) -> dict[str, float]:
    """Run one train or validation epoch over real batches."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    batches = 0
    samples = 0
    errors = 0
    reference_tokens = 0
    exact_matches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            keypoints = batch["keypoints"].to(device)
            masks = batch["masks"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            target_ids = batch["target_ids"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(keypoints, input_lengths, masks)
            loss = model.ctc_loss(
                logits,
                input_lengths,
                target_ids,
                target_lengths,
                vocabulary.token2id[vocabulary.BLANK_TOKEN],
                target_length_weighting=ctc_target_length_weighting,
                target_length_weight_cap=ctc_target_length_weight_cap,
            )
            if training:
                loss.backward()
                clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()

            predictions = greedy_ctc_decode(logits.detach(), input_lengths, vocabulary)
            batch_errors, batch_tokens, batch_exact = _batch_metrics(
                predictions,
                target_ids.detach().cpu(),
                target_lengths.detach().cpu(),
                vocabulary,
            )
            total_loss += float(loss.detach().cpu())
            errors += batch_errors
            reference_tokens += batch_tokens
            exact_matches += batch_exact
            batches += 1
            samples += len(predictions)

    if batches == 0:
        raise ValueError("The data loader produced no batches")
    return {
        "ctc_loss": total_loss / batches,
        "gloss_error_rate": errors / max(reference_tokens, 1),
        "gloss_exact_match": exact_matches / max(samples, 1),
    }


def train_si(config: TrainingConfig) -> dict[str, Any]:
    """Train the SI baseline with CTC validation and early stopping."""
    set_reproducible_seed(config.seed)
    data_root = resolve_data_root(
        config.data_root,
        config.archive_path,
        search_roots=[config.project_root],
    )
    device = select_device(config.device)
    resume_payload: dict[str, Any] | None = None
    if config.resume_from is not None:
        model, vocabulary, preprocessing_config, resume_payload = load_checkpoint(
            config.resume_from,
            device=device,
        )
        saved_training = resume_payload.get("training_config", {})
        protected_settings = (
            "protocol",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "gradient_clip",
            "num_workers",
            "seed",
            "early_stopping_patience",
            "early_stopping_min_delta",
            "checkpoint_every",
            "reject_invalid_ctc_samples",
            "coordinate_normalization",
            "feature_representation",
            "global_coordinate_scale",
            "ctc_target_length_weighting",
            "ctc_target_length_weight_cap",
        )
        mismatches = {
            name: (saved_training.get(name), getattr(config, name))
            for name in protected_settings
            if name in saved_training and saved_training[name] != getattr(config, name)
        }
        if mismatches:
            raise ValueError(
                "Resume settings differ from checkpoint: "
                f"{mismatches}. Use the checkpoint training settings."
            )
    else:
        preprocessing_config = PreprocessingConfig(
            missing_file_policy="error",
            coordinate_normalization=config.coordinate_normalization,
            feature_representation=config.feature_representation,
            global_coordinate_scale=config.global_coordinate_scale,
        )
        model = None
        vocabulary = None

    train_set, dev_set, test_set, built_vocabulary = build_si_datasets(
        config.project_root,
        data_root,
        preprocessing_config,
    )
    if vocabulary is None:
        vocabulary = built_vocabulary
    elif vocabulary.token2id != built_vocabulary.token2id:
        raise ValueError("Checkpoint vocabulary differs from SI training vocabulary")
    ctc_reports = {
        "train": validate_ctc_lengths(train_set),
        "dev": validate_ctc_lengths(dev_set),
        "test": validate_ctc_lengths(test_set),
    }
    report_path = _write_json(
        Path(config.project_root)
        / config.output_dir
        / "isharah500_si_ctc_validation.json",
        ctc_reports,
    )
    train_set = select_ctc_subset(
        train_set,
        ctc_reports["train"],
        allow_rejected=config.reject_invalid_ctc_samples,
    )
    dev_set = select_ctc_subset(
        dev_set,
        ctc_reports["dev"],
        allow_rejected=config.reject_invalid_ctc_samples,
    )
    test_set = select_ctc_subset(
        test_set,
        ctc_reports["test"],
        allow_rejected=config.reject_invalid_ctc_samples,
    )
    train_loader = make_loader(
        train_set,
        config.batch_size,
        config.num_workers,
        shuffle=True,
    )
    dev_loader = make_loader(
        dev_set,
        config.batch_size,
        config.num_workers,
        shuffle=False,
    )
    if model is None:
        model = BiGRUCTC(
            len(vocabulary),
            config=BiGRUCTCConfig(
                features_per_hand=preprocessing_config.output_features_per_hand,
            ),
        ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    start_epoch = 0
    history: list[dict[str, Any]] = []
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        start_epoch = int(resume_payload["epoch"])
        if config.epochs <= start_epoch:
            raise ValueError(
                f"Target epochs ({config.epochs}) must exceed checkpoint epoch "
                f"({start_epoch})"
            )
        best_validation_loss = float(
            resume_payload.get(
                "best_validation_loss",
                resume_payload.get("validation_metrics", {}).get("ctc_loss", float("inf")),
            )
        )
        epochs_without_improvement = int(
            resume_payload.get("epochs_without_improvement", 0)
        )
        if "random_states" not in resume_payload:
            raise ValueError(
                "Checkpoint has no random_states; refusing a non-reproducible resume"
            )
        restore_random_states(resume_payload["random_states"])
    output_dir = Path(config.project_root) / config.output_dir
    for epoch in range(start_epoch, config.epochs):
        train_metrics = run_epoch(
            model,
            train_loader,
            vocabulary,
            device,
            optimizer=optimizer,
            gradient_clip=config.gradient_clip,
            ctc_target_length_weighting=config.ctc_target_length_weighting,
            ctc_target_length_weight_cap=config.ctc_target_length_weight_cap,
        )
        validation_metrics = run_epoch(
            model,
            dev_loader,
            vocabulary,
            device,
            ctc_target_length_weighting=config.ctc_target_length_weighting,
            ctc_target_length_weight_cap=config.ctc_target_length_weight_cap,
        )
        epoch_number = epoch + 1
        validation_loss = validation_metrics["ctc_loss"]
        improved = validation_loss < (
            best_validation_loss - config.early_stopping_min_delta
        )
        best_checkpoint = None
        if improved:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            best_checkpoint = save_checkpoint(
                output_dir / "isharah500_si_best.pt",
                model,
                optimizer,
                vocabulary,
                preprocessing_config,
                config,
                epoch=epoch_number,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                data_validation=ctc_reports,
                best_validation_loss=best_validation_loss,
                epochs_without_improvement=epochs_without_improvement,
            )
        else:
            epochs_without_improvement += 1

        checkpoint = None
        if epoch_number % config.checkpoint_every == 0:
            checkpoint = save_checkpoint(
                output_dir / f"isharah500_si_epoch_{epoch_number:03d}.pt",
                model,
                optimizer,
                vocabulary,
                preprocessing_config,
                config,
                epoch=epoch_number,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                data_validation=ctc_reports,
                best_validation_loss=best_validation_loss,
                epochs_without_improvement=epochs_without_improvement,
            )

        history.append(
            {
                "epoch": epoch_number,
                "train": train_metrics,
                "validation": validation_metrics,
                "checkpoint": str(checkpoint) if checkpoint else None,
                "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
                "early_stopping": {
                    "best_validation_ctc_loss": best_validation_loss,
                    "epochs_without_improvement": epochs_without_improvement,
                },
            }
        )
        _write_json(output_dir / "isharah500_si_training_history.json", history)
        _write_json(
            output_dir / "isharah500_si_validation_metrics.json",
            [entry["validation"] for entry in history],
        )
        if epochs_without_improvement >= config.early_stopping_patience:
            break
    return {
        "history": history,
        "device": str(device),
        "ctc_validation_report": str(report_path),
    }


def save_checkpoint(
    path: str | Path,
    model: BiGRUCTC,
    optimizer: torch.optim.Optimizer,
    vocabulary: GlossVocabulary,
    preprocessing_config: PreprocessingConfig,
    training_config: TrainingConfig,
    *,
    epoch: int,
    train_metrics: dict[str, float],
    validation_metrics: dict[str, float],
    data_validation: dict[str, Any] | None = None,
    best_validation_loss: float | None = None,
    epochs_without_improvement: int = 0,
) -> Path:
    """Save a complete, inference-reusable training checkpoint."""
    output_path = _normalize_cross_platform_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model.config.to_dict(),
        "vocabulary": {
            "add_ctc_blank": vocabulary.add_ctc_blank,
            "token2id": vocabulary.token2id,
            "id2token": vocabulary.id2token,
        },
        "preprocessing_config": preprocessing_config.to_dict(),
        "training_config": training_config.to_dict(),
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "data_validation": data_validation,
        "best_validation_loss": (
            float(best_validation_loss)
            if best_validation_loss is not None
            else float(validation_metrics.get("ctc_loss", float("inf")))
        ),
        "epochs_without_improvement": epochs_without_improvement,
        "random_states": capture_random_states(),
    }
    torch.save(payload, output_path)
    return output_path


def load_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[BiGRUCTC, GlossVocabulary, PreprocessingConfig, dict[str, Any]]:
    """Load model and preprocessing metadata from a Phase 3A checkpoint."""
    checkpoint_path = _normalize_cross_platform_path(path)
    try:
        payload = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except pickle.UnpicklingError as exc:
        # Only trusted local legacy checkpoints containing the old NumPy RNG
        # tuple need this compatibility path. New checkpoints remain safe-loadable.
        if "numpy" not in str(exc).lower():
            raise
        payload = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    model_config = BiGRUCTCConfig.from_dict(payload["model_config"])
    vocabulary = GlossVocabulary(
        add_ctc_blank=payload["vocabulary"]["add_ctc_blank"],
    )
    vocabulary.token2id = {
        str(token): int(index)
        for token, index in payload["vocabulary"]["token2id"].items()
    }
    vocabulary.id2token = {
        int(index): str(token)
        for index, token in payload["vocabulary"]["id2token"].items()
    }
    model = BiGRUCTC(len(vocabulary), config=model_config)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    preprocessing_config = PreprocessingConfig.from_dict(
        payload["preprocessing_config"]
    )
    return model, vocabulary, preprocessing_config, payload


def run_smoke_training(config: TrainingConfig) -> dict[str, Any]:
    """Run one real train batch and one real validation batch only."""
    set_reproducible_seed(config.seed)
    data_root = resolve_data_root(
        config.data_root,
        config.archive_path,
        search_roots=[config.project_root],
    )
    preprocessing_config = PreprocessingConfig(
        missing_file_policy="error",
        coordinate_normalization=config.coordinate_normalization,
        feature_representation=config.feature_representation,
        global_coordinate_scale=config.global_coordinate_scale,
    )
    train_set, dev_set, _, vocabulary = build_si_datasets(
        config.project_root,
        data_root,
        preprocessing_config,
    )
    if config.smoke_samples <= 0:
        raise ValueError("smoke_samples must be positive")
    train_subset = Subset(train_set, range(min(config.smoke_samples, len(train_set))))
    dev_subset = Subset(dev_set, range(min(config.smoke_samples, len(dev_set))))
    train_loader = make_loader(train_subset, config.batch_size, 0, shuffle=False)
    dev_loader = make_loader(dev_subset, config.batch_size, 0, shuffle=False)
    device = select_device(config.device)
    model = BiGRUCTC(
        len(vocabulary),
        config=BiGRUCTCConfig(
            features_per_hand=preprocessing_config.output_features_per_hand,
        ),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_metrics = run_epoch(
        model,
        train_loader,
        vocabulary,
        device,
        optimizer=optimizer,
        gradient_clip=config.gradient_clip,
        ctc_target_length_weighting=config.ctc_target_length_weighting,
        ctc_target_length_weight_cap=config.ctc_target_length_weight_cap,
    )
    validation_metrics = run_epoch(
        model,
        dev_loader,
        vocabulary,
        device,
        ctc_target_length_weighting=config.ctc_target_length_weighting,
        ctc_target_length_weight_cap=config.ctc_target_length_weight_cap,
    )
    checkpoint = save_checkpoint(
        Path(config.project_root) / config.output_dir / "isharah500_si_smoke.pt",
        model,
        optimizer,
        vocabulary,
        preprocessing_config,
        config,
        epoch=0,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
    )
    return {
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "checkpoint": str(checkpoint),
        "device": str(device),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--archive-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("isharah500/checkpoints"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument(
        "--reject-invalid-ctc-samples",
        action="store_true",
        help="Explicitly exclude and document CTC-infeasible samples.",
    )
    parser.add_argument("--smoke-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--coordinate-normalization",
        choices=("none", "hand_centric"),
        default="none",
        help="Optional hand-centric wrist/palm normalization; baseline default is none.",
    )
    parser.add_argument(
        "--feature-representation",
        choices=(FEATURE_REPRESENTATION_RAW_63, FEATURE_REPRESENTATION_GLOBAL_POSITION_V1),
        default=FEATURE_REPRESENTATION_RAW_63,
        help="Optional versioned feature construction; raw_63 is the baseline.",
    )
    parser.add_argument(
        "--global-coordinate-scale",
        type=float,
        default=1.0,
        help="Scale applied to appended global-position channels.",
    )
    parser.add_argument(
        "--ctc-target-length-weighting",
        choices=("none", "linear", "sqrt"),
        default="none",
        help="Optional capped target-length weighting for per-sample CTC loss.",
    )
    parser.add_argument(
        "--ctc-target-length-weight-cap",
        type=float,
        default=2.0,
        help="Cap applied before mean-normalizing target-length weights.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Resume from a complete checkpoint without resetting optimizer state.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Read-only SI train/dev CTC validation; never trains or writes files.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    if arguments.preflight:
        if arguments.smoke or arguments.train:
            raise SystemExit("--preflight cannot be combined with --smoke or --train")
        if arguments.data_root is None:
            raise SystemExit("--preflight requires --data-root")
        report = run_si_preflight(
            arguments.project_root,
            arguments.data_root,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if report["invalid_samples"]:
            raise SystemExit(1)
        return
    config = TrainingConfig(
        project_root=str(arguments.project_root),
        data_root=str(arguments.data_root) if arguments.data_root else None,
        archive_path=str(arguments.archive_path) if arguments.archive_path else None,
        output_dir=str(arguments.output_dir),
        batch_size=arguments.batch_size,
        epochs=arguments.epochs,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        gradient_clip=arguments.gradient_clip,
        num_workers=arguments.num_workers,
        early_stopping_patience=arguments.early_stopping_patience,
        early_stopping_min_delta=arguments.early_stopping_min_delta,
        checkpoint_every=arguments.checkpoint_every,
        reject_invalid_ctc_samples=arguments.reject_invalid_ctc_samples,
        smoke_samples=arguments.smoke_samples,
        seed=arguments.seed,
        coordinate_normalization=arguments.coordinate_normalization,
        feature_representation=arguments.feature_representation,
        global_coordinate_scale=arguments.global_coordinate_scale,
        ctc_target_length_weighting=arguments.ctc_target_length_weighting,
        ctc_target_length_weight_cap=arguments.ctc_target_length_weight_cap,
        resume_from=str(arguments.resume_from) if arguments.resume_from else None,
        device=arguments.device,
    )
    if arguments.smoke:
        result = run_smoke_training(config)
    elif arguments.train:
        result = train_si(config)
    else:
        raise SystemExit("Choose exactly one execution mode: --smoke or --train")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
