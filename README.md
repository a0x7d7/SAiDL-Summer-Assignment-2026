# SAiDL-Summer-Assignment-2026

This is my final submission repo for the SAiDL Summer Assignment 2026.

I took the Core ML route and the Sparsity \& Optimization route, and the final report is here:

[Final report PDF](reports/new/assignment_report.pdf)

## What is inside

- `core_ml/`: the modular Transformer stack for the Core ML experiments.
- `sparsity/`: LoRA, AdaLoRA, SoRA, the L1 optimization comparison, and the xLSTM/Mamba extension work.
- `configs/`: the final GPU configs used for the reported runs.
- `scripts/`: helper scripts, including the GLUE CoLA download script.
- `glue_data/CoLA/`: the CoLA files in the GLUE folder layout.
- `results/`: the final experiment outputs used in the report.
- `reports/new/`: the final report source and PDF.
- `requirements.txt`: the Python dependencies I used.

## How to run it

```powershell
python -m pip install -r requirements.txt
python scripts/download_glue_cola.py --data_dir glue_data
python -m core_ml.run_suite --config configs/core_ml_gpu.yaml
python -m sparsity.run_suite --config configs/sparsity_gpu.yaml
python -m sparsity.run_suite --config configs/sparsity_deberta_long.yaml
```

The repo was built and checked on an RTX 4060 Laptop GPU with CUDA-enabled PyTorch.
