from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
from pathlib import Path
from struct import unpack

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-pj1")

import matplotlib.pyplot as plt
import numpy as np

import mynn as nn


ROOT = Path(__file__).resolve().parent


def load_mnist(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_images_path = root / "dataset/MNIST/train-images-idx3-ubyte.gz"
    train_labels_path = root / "dataset/MNIST/train-labels-idx1-ubyte.gz"
    test_images_path = root / "dataset/MNIST/t10k-images-idx3-ubyte.gz"
    test_labels_path = root / "dataset/MNIST/t10k-labels-idx1-ubyte.gz"

    with gzip.open(train_images_path, "rb") as f:
        _, num, rows, cols = unpack(">4I", f.read(16))
        train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
    with gzip.open(train_labels_path, "rb") as f:
        _, num = unpack(">2I", f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)
    with gzip.open(test_images_path, "rb") as f:
        _, num, rows, cols = unpack(">4I", f.read(16))
        test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
    with gzip.open(test_labels_path, "rb") as f:
        _, num = unpack(">2I", f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

    return train_imgs.astype(np.float32) / 255.0, train_labs, test_imgs.astype(np.float32) / 255.0, test_labs


def split_train_valid(images: np.ndarray, labels: np.ndarray, valid_size: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(images.shape[0])
    valid_idx = idx[:valid_size]
    train_idx = idx[valid_size:]
    return images[train_idx], labels[train_idx], images[valid_idx], labels[valid_idx]


def batch_iter(X: np.ndarray, y: np.ndarray, batch_size: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    for start in range(0, X.shape[0], batch_size):
        batch_idx = idx[start : start + batch_size]
        if batch_idx.size:
            yield X[batch_idx], y[batch_idx]


def evaluate(model, X: np.ndarray, y: np.ndarray, batch_size: int = 512) -> tuple[float, float]:
    loss_fn = nn.op.MultiCrossEntropyLoss(model=None, max_classes=10)
    total_loss = 0.0
    total_correct = 0
    total = 0
    for start in range(0, X.shape[0], batch_size):
        xb = X[start : start + batch_size]
        yb = y[start : start + batch_size]
        logits = model(xb)
        loss = loss_fn(logits, yb)
        total_loss += loss * xb.shape[0]
        total_correct += (np.argmax(logits, axis=1) == yb).sum()
        total += xb.shape[0]
    return total_loss / total, total_correct / total


def predict(model, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    preds = []
    for start in range(0, X.shape[0], batch_size):
        logits = model(X[start : start + batch_size])
        preds.append(np.argmax(logits, axis=1))
    return np.concatenate(preds)


def train_model(name: str, model, optimizer, scheduler, data: dict, args) -> dict:
    output_dir = Path(args.output_dir) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)
    history = []
    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        train_losses = []
        train_correct = 0
        seen = 0
        for xb, yb in batch_iter(data["train_X"], data["train_y"], args.batch_size, args.seed + epoch):
            logits = model(xb)
            loss = loss_fn(logits, yb)
            train_losses.append(loss)
            train_correct += (np.argmax(logits, axis=1) == yb).sum()
            seen += xb.shape[0]
            loss_fn.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        train_loss = float(np.mean(train_losses))
        train_acc = float(train_correct / seen)
        val_loss, val_acc = evaluate(model, data["valid_X"], data["valid_y"], args.eval_batch_size)
        row = {
            "epoch": epoch,
            "lr": float(optimizer.init_lr),
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
        }
        history.append(row)
        print(f"{name} epoch={epoch} train_acc={train_acc:.4f} val_acc={val_acc:.4f} val_loss={val_loss:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_model(output_dir / "best_model.pickle")

    with (output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    plot_history(history, output_dir / "learning_curve.png", title=name)
    model.save_model(output_dir / "last_model.pickle")
    return {"best_val_acc": best_val_acc, "history": history, "output_dir": str(output_dir)}


def plot_history(history: list[dict], path: Path, title: str) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train_loss")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val_loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_acc"] for row in history], label="train_acc")
    axes[1].plot(epochs, [row["val_acc"] for row in history], label="val_acc")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_class: int = 10) -> np.ndarray:
    cm = np.zeros((n_class, n_class), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def save_confusion_matrix(cm: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_misclassified(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, path: Path, max_images: int = 16) -> None:
    wrong = np.where(y_true != y_pred)[0][:max_images]
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for ax, idx in zip(axes.flat, wrong):
        ax.imshow(X[idx].reshape(28, 28), cmap="gray")
        ax.set_title(f"t={y_true[idx]}, p={y_pred[idx]}", fontsize=8)
        ax.axis("off")
    for ax in axes.flat[len(wrong) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_kernels(model, path: Path) -> None:
    first_conv = next(layer for layer in model.layers if isinstance(layer, nn.op.conv2D))
    kernels = first_conv.W[:, 0]
    fig, axes = plt.subplots(2, 4, figsize=(6, 3))
    for ax, kernel in zip(axes.flat, kernels):
        ax.imshow(kernel, cmap="coolwarm")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def load_cnn(path: Path):
    model = nn.models.Model_CNN()
    model.load_model(path)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--valid-size", type=int, default=10000)
    parser.add_argument("--train-limit", type=int, default=20000)
    parser.add_argument("--test-limit", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=309)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.epochs = 1
        args.train_limit = 512
        args.valid_size = 256
        args.test_limit = 512

    np.random.seed(args.seed)
    X, y, test_X, test_y = load_mnist(ROOT)
    train_X, train_y, valid_X, valid_y = split_train_valid(X, y, args.valid_size, args.seed)
    if args.train_limit:
        train_X, train_y = train_X[: args.train_limit], train_y[: args.train_limit]
    if args.test_limit:
        test_X, test_y = test_X[: args.test_limit], test_y[: args.test_limit]
    data = {"train_X": train_X, "train_y": train_y, "valid_X": valid_X, "valid_y": valid_y}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = []
    mlp = nn.models.Model_MLP([784, 128, 10], "ReLU", [1e-4, 1e-4])
    experiments.append(("mlp_sgd", mlp, nn.optimizer.SGD(0.08, mlp), None))

    cnn = nn.models.Model_CNN(weight_decay=False)
    experiments.append(("cnn_sgd", cnn, nn.optimizer.SGD(0.03, cnn), None))

    cnn_m = nn.models.Model_CNN(weight_decay=False)
    opt_m = nn.optimizer.MomentGD(0.02, cnn_m, mu=0.9)
    scheduler_m = nn.lr_scheduler.MultiStepLR(opt_m, milestones=[max(1, (train_X.shape[0] // args.batch_size) * 3)], gamma=0.5)
    experiments.append(("cnn_momentum", cnn_m, opt_m, scheduler_m))

    summary = {}
    for name, model, optimizer, scheduler in experiments:
        result = train_model(name, model, optimizer, scheduler, data, args)
        best_path = Path(result["output_dir"]) / "best_model.pickle"
        if name.startswith("cnn"):
            model.load_model(best_path)
        else:
            model.load_model(best_path)
        test_loss, test_acc = evaluate(model, test_X, test_y, args.eval_batch_size)
        summary[name] = {
            "best_val_acc": float(result["best_val_acc"]),
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "best_model": str(best_path),
        }

    best_cnn_name = max([k for k in summary if k.startswith("cnn")], key=lambda k: summary[k]["test_acc"])
    best_cnn = load_cnn(Path(summary[best_cnn_name]["best_model"]))
    pred = predict(best_cnn, test_X, args.eval_batch_size)
    cm = confusion_matrix(test_y, pred)
    save_confusion_matrix(cm, output_dir / "confusion_matrix.png")
    save_misclassified(test_X, test_y, pred, output_dir / "misclassified_examples.png")
    save_kernels(best_cnn, output_dir / "conv1_kernels.png")
    summary["best_cnn_for_analysis"] = best_cnn_name

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
