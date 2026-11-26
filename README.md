# CS6886W Assignment 2 — MobileNetV2 Compression

This repository contains the notebook and saved artifacts for IIT Madras CS6886W Systems Engineering for Deep Learning Assignment 2. The project trains MobileNetV2 on CIFAR-10 and evaluates eight weight/activation quantization configurations.

## Contents

- `Assignment2.ipynb` — training, evaluation, compression implementation, sweep, accounting, and W&B visualization cells.
- `results/compression_sweep_results.csv` — metrics for all eight configurations.
- `results/compression_accounting.csv` — model storage accounting.
- `results/metrics_summary.json` — experiment settings and selected result summary.
- `results/baseline_training_curves.png` — baseline training curves.
- `results/compression_parallel_coordinates.png` — parallel coordinates plot.
- `results/results_report.txt` — exported results summary.
- `checkpoints/mobilenetv2_cifar10_baseline_state.pth` — baseline state dict.

## Environment

The recorded experiment environment used Python in Google Colab with CUDA, PyTorch `2.14.1+cu130`, and torchvision `0.29.1+cu130`. The notebook installs W&B and uses NumPy, pandas, Matplotlib, Plotly, and wandb-workspaces for logging and visualization.

Install dependencies (use a CUDA-compatible PyTorch build for your runtime):

```bash
pip install -r requirements.txt
```

## Reproduce

1. Open `Assignment2.ipynb` in Google Colab or a compatible Jupyter environment.
2. Select a CUDA GPU runtime for training.
3. Run the notebook cells in order. CIFAR-10 is downloaded by torchvision.
4. When prompted, authenticate with W&B if you want to log runs to your account. Set `USE_WANDB = False` to run without W&B logging where applicable.
5. The notebook exports the sweep CSV, accounting CSV, checkpoint, JSON summary, and plots to `assignment2_outputs_improved/`.

The notebook uses seed `42`, baseline training for `20` epochs, batch size `128`, image size `96`, and `3` recovery epochs per compression configuration at learning rate `0.001`. See the notebook and `results/metrics_summary.json` for the recorded configuration.

## Recorded results

- Baseline test accuracy: **89.97%**
- Selected configuration: **W4A8** (4-bit weights, 8-bit activations)
- W4A8 test accuracy: **89.18%**
- Approximate resident packed model size: **1.444584 MB**
- Reported resident weight compression ratio: **6.288×**
- Largest measured ReLU6 activation FP32 size: **113.246208 MB**
- A8 packed activation estimate: **28.311552 MB** (4×)

## Important implementation limitation

This notebook uses fake-quantized FP32 inference and separately accounts for packed weight byte buffers, scaling metadata, and untouched FP32 tensors. The packed resident model size is a storage-accounting estimate; it is **not** evidence of native low-bit GPU execution or an actual reduction of GPU VRAM use. GPU peak allocation is reported separately. Activation packed sizes are estimates for the measured activation tensor; runtime activations remain FP32 in this implementation.

## W&B

W&B project:
https://wandb.ai/keerthana_cs25m512-indian-institute-of-technology-madras/cs6886w-mobilenetv2-cifar10

## Assignment report

Prepare the final Q1–Q5 PDF separately and place it at `report/Assignment2_Report.pdf` before final submission. The assignment asks for a single PDF containing answers, figures, tables, and the GitHub repository link; upload that PDF through the official CODE submission Google Form.
