"""Train or verify KArSL-502 centroid and linear baselines on frozen PASS shards."""

import argparse
import hashlib
import importlib.util
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from torch import nn

SOURCE_SHA256 = "f6e0f60a817de984fc356b7bbd7edcaff72ffbc63e046fe09175564cc6633b5e"
REFERENCE_SHA256 = "a2af758254de890621b1296c9167a4bacae98e2ca4eaec20e1ae8ab2b9eed7e5"
EXPECTED_CENTROID = (0.19973073572503366, 0.176593137986175, 0.35408252158073966)
EXPECTED_LINEAR = (0.2782133523402233, 0.2675255841641251, 0.47806288112774215)


def load_frozen_source(path):
    path = Path(path)
    data = path.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError("Frozen loader changed.")
    spec = importlib.util.spec_from_file_location("frozen_baseline_loader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Centroid(nn.Module):
    def __init__(self, centers):
        super().__init__()
        self.register_buffer("centers", centers)
        self.register_buffer("norms", centers.square().sum(dim=1))

    def forward(self, x):
        flat = x.flatten(1)
        return 2 * flat @ self.centers.T - self.norms


class Standardized(nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        flat = (x.flatten(1) - self.mean) / self.std
        return self.model(flat.reshape(-1, 32, 150))


def train_statistics(x):
    flat = x.reshape(-1, 4800)
    return torch.from_numpy(flat.mean(axis=0)), torch.from_numpy(
        flat.std(axis=0) + np.float32(1e-6)
    )


def centroid_model(train_x, train_y, mean, std, device):
    flat = torch.from_numpy(train_x.reshape(-1, 4800)).to(device)
    labels = torch.from_numpy(train_y).to(device)
    centers = torch.zeros((502, 4800), device=device)
    centers.index_add_(0, labels, flat)
    counts = torch.bincount(labels, minlength=502)
    if bool(torch.any(counts == 0)):
        raise ValueError("Each of 502 classes needs train support.")
    centers = (centers / counts[:, None] - mean.to(device)) / std.to(device)
    return Standardized(Centroid(centers), mean, std).to(device)


def centroid_metrics(model, loader, device):
    truth, scores = [], []
    model.eval()
    with torch.inference_mode():
        for x, y in loader:
            result = model(x.to(device, non_blocking=True))
            if not bool(torch.isfinite(result).all()):
                raise ValueError("Nonfinite centroid scores.")
            truth.append(y.numpy())
            scores.append(result.cpu().numpy())
    y, scores = np.concatenate(truth), np.concatenate(scores)
    pred = scores.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(
            f1_score(y, pred, average="macro", labels=np.arange(502), zero_division=0)
        ),
        "top5_accuracy": float(top_k_accuracy_score(y, scores, k=5, labels=np.arange(502))),
    }


def linear_model(mean, std, device):
    return Standardized(nn.Sequential(nn.Flatten(), nn.Linear(4800, 502)), mean, std).to(device)


def validate_inputs(train_x, train_y, val_x, val_y):
    if (
        train_x.ndim != 3
        or val_x.ndim != 3
        or train_x.shape[1:] != (32, 150)
        or val_x.shape[1:] != (32, 150)
        or not len(train_y)
        or not len(val_y)
        or len(train_x) != len(train_y)
        or len(val_x) != len(val_y)
    ):
        raise ValueError("Wrong or unaligned frozen feature shape.")
    if not np.isfinite(train_x).all() or not np.isfinite(val_x).all():
        raise ValueError("Nonfinite features.")
    if (
        not np.issubdtype(train_y.dtype, np.integer)
        or not np.issubdtype(val_y.dtype, np.integer)
        or np.any((train_y < 0) | (train_y >= 502))
        or np.any((val_y < 0) | (val_y >= 502))
    ):
        raise ValueError("Invalid encoded class IDs.")


def run_baselines(
    frozen, train_x, train_y, val_x, val_y, output, device, epochs=30, historical_checkpoint=None
):
    validate_inputs(train_x, train_y, val_x, val_y)
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    mean, std = train_statistics(train_x)
    val_loader = frozen.make_loader(val_x, val_y, 256, False, device)
    centroid = centroid_model(train_x, train_y, mean, std, device)
    centroid_result = centroid_metrics(centroid, val_loader, device)
    result = {
        "source_fingerprint": frozen.FINGERPRINT,
        "train_samples": len(train_y),
        "validation_samples": len(val_y),
        "centroid_validation": centroid_result,
        "test_evaluated": False,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
        },
    }
    if historical_checkpoint is not None:
        path = Path(historical_checkpoint)
        if hashlib.sha256(path.read_bytes()).hexdigest() != REFERENCE_SHA256:
            raise ValueError("Historical checkpoint SHA-256 mismatch.")
        old = torch.load(path, map_location="cpu", weights_only=True)
        if (
            old["source_fingerprint"] != frozen.FINGERPRINT
            or not torch.equal(old["mean"], mean)
            or not torch.equal(old["std"], std)
        ):
            raise ValueError("Historical train-only statistics changed.")
        model = linear_model(mean, std, device)
        model.model[1].load_state_dict(old["state_dict"], strict=True)
        result["linear_validation"] = frozen.evaluate(model, val_loader, device)
        for metrics, expected in (
            (centroid_result, EXPECTED_CENTROID),
            (result["linear_validation"], EXPECTED_LINEAR),
        ):
            if any(
                abs(metrics[k] - value) > 1e-10
                for k, value in zip(
                    ("accuracy", "macro_f1", "top5_accuracy"), expected, strict=True
                )
            ):
                raise ValueError("Historical validation scores did not reproduce.")
        result["status"] = "HISTORICAL_VALIDATION_REPLAY_PASSED"
        result["original_checkpoint_sha256"] = REFERENCE_SHA256
    else:
        result["experiment_id"] = "baseline_502_reference_v2"
        model = linear_model(mean, std, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
        loader = frozen.make_loader(train_x, train_y, 128, True, device)
        best = -1.0
        history = []
    output = Path(output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite an experiment.")
    output.mkdir(parents=True)
    if historical_checkpoint is None:
        config = {
            "experiment_id": result["experiment_id"],
            "seed": 42,
            "source_fingerprint": frozen.FINGERPRINT,
            "preprocessing_version": frozen.PREPROCESSING,
            "class_ids": frozen.CLASSES,
            "normalization": "train_mean_train_std_plus_1e-6",
            "optimizer": "AdamW",
            "lr": 0.001,
            "weight_decay": 0.001,
            "batch_size": 128,
            "epochs": epochs,
            "selection": "validation_macro_f1",
            "test_accessed": False,
        }
        (output / "experiment.json").write_text(json.dumps(config, indent=2) + "\n")
        torch.save(
            {
                "centers": centroid.model.centers.cpu(),
                "mean": mean,
                "std": std,
                "validation_metrics": centroid_result,
                "config": config,
            },
            output / "nearest_centroid.pt",
        )
        started = time.perf_counter()
        for epoch in range(1, epochs + 1):
            model.train()
            for x, y in loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(x), y)
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Nonfinite linear loss.")
                loss.backward()
                optimizer.step()
            metrics = frozen.evaluate(model, val_loader, device)
            history.append({"epoch": epoch, **metrics})
            frozen.save_csv(output / "linear_history.csv", history)
            print(
                f"Epoch {epoch:02d}: val_acc={metrics['accuracy']:.4f}, "
                f"val_macro_f1={metrics['macro_f1']:.4f}",
                flush=True,
            )
            if metrics["macro_f1"] > best + 1e-6:
                best = metrics["macro_f1"]
                result["linear_validation"] = metrics
                result["linear_best_epoch"] = epoch
                torch.save(
                    {
                        "state_dict": frozen.cpu_state(model.model[1]),
                        "mean": mean,
                        "std": std,
                        "config": config,
                        "validation_metrics": metrics,
                        "test_evaluated": False,
                    },
                    output / "linear_best.pt",
                )
        result["training_seconds"] = time.perf_counter() - started
        ckpt = torch.load(output / "linear_best.pt", map_location="cpu", weights_only=True)
        model.model[1].load_state_dict(ckpt["state_dict"], strict=True)
        frozen.check_metrics(
            frozen.evaluate(model, val_loader, device), result["linear_validation"]
        )
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ("features-root", "cache", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--verify-historical-checkpoint", type=Path)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; use --device cpu.")
    torch.set_num_threads(2)
    frozen = load_frozen_source(
        Path(__file__).resolve().parents[1] / "scripts/signova_compare_cnn_502.py"
    )
    frozen.seed_all(42)
    report = json.loads((args.features_root / "preprocessing_report.json").read_text())
    manifest = json.loads((args.features_root / "shards.json").read_text())
    if (
        report["fingerprint"] != manifest["fingerprint"]
        or report["fingerprint"] != frozen.FINGERPRINT
        or report["expected_samples"] != 75515
        or len(manifest["shards"]) != 149
        or report["raw_cross_split_duplicate_groups"] != 0
    ):
        raise ValueError("Frozen source inventory mismatch.")
    train_x, train_y, train_paths, train_raw, train_feat, train_info = frozen.load_pass_split(
        args.features_root, args.cache, manifest, "train"
    )
    val_x, val_y, val_paths, val_raw, val_feat, val_info = frozen.load_pass_split(
        args.features_root, args.cache, manifest, "validation"
    )
    if (
        len(train_y) != 25032
        or len(val_y) != 25254
        or train_info["signers"] != [3]
        or val_info["signers"] != [2]
        or set(train_paths) & set(val_paths)
        or train_raw & val_raw
        or train_feat & val_feat
    ):
        raise ValueError("Frozen PASS split or overlap mismatch.")
    result = run_baselines(
        frozen,
        train_x,
        train_y,
        val_x,
        val_y,
        args.output,
        torch.device(args.device),
        args.epochs,
        args.verify_historical_checkpoint,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
