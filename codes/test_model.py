from __future__ import annotations

import argparse

import mynn as nn
from pathlib import Path

from run_experiments import ROOT, evaluate, load_mnist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["mlp", "cnn"], default="cnn")
    parser.add_argument("--model-path", default="outputs_final/cnn_momentum/best_model.pickle")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    _, _, test_X, test_y = load_mnist(ROOT)
    if args.model_type == "mlp":
        model = nn.models.Model_MLP()
    else:
        model = nn.models.Model_CNN()
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = ROOT.parent / model_path
    model.load_model(model_path)
    test_loss, test_acc = evaluate(model, test_X, test_y, args.batch_size)
    print(f"test_loss={test_loss:.6f}")
    print(f"test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
