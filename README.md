# Neural Network and Deep Learning Project 1

This project implements MNIST classification with NumPy-based neural network components.

## Implemented Parts

1. MLP baseline
   - Linear forward and backward propagation.
   - Softmax cross-entropy loss.
   - MNIST training and validation curves.

2. CNN model
   - NumPy `conv2D` forward and backward propagation.
   - A simple CNN model for MNIST.
   - MLP vs CNN comparison under the same data split.

3. Additional directions
   - Optimization: Momentum SGD compared with vanilla SGD.
   - Error analysis and visualization: confusion matrix, misclassified examples, and convolution kernels.

## Main Files

```text
codes/mynn/op.py              Linear, conv2D, ReLU, Flatten, softmax loss
codes/mynn/models.py          Model_MLP and Model_CNN
codes/mynn/optimizer.py       SGD and Momentum SGD
codes/mynn/lr_scheduler.py    StepLR, MultiStepLR, ExponentialLR
codes/run_experiments.py      Training, evaluation, and visualization script
```

## Run Experiments

```bash
cd codes
PYTHONPATH=. python run_experiments.py \
  --output-dir ../outputs_final \
  --epochs 5 \
  --batch-size 128 \
  --train-limit 20000 \
  --valid-size 10000 \
  --test-limit 10000
```

Quick smoke test:

```bash
cd codes
PYTHONPATH=. python run_experiments.py --smoke --output-dir ../outputs_smoke
```

## Current Results

The saved experiment under `outputs_final/` uses 20000 training images, 10000 validation images, and 10000 test images.

| Model | Best Val Acc | Test Acc |
| --- | ---: | ---: |
| MLP + SGD | 0.9213 | 0.9254 |
| CNN + SGD | 0.9243 | 0.9292 |
| CNN + Momentum | 0.9673 | 0.9699 |

Final selected model weights are stored under `artifacts/final_weights/`.
