# RunPod Integration for Steel Defect Detection

This directory contains files and instructions for running the Steel Defect Detection project on RunPod.io GPU instances.

## Files

- `runpod_guide.md`: Comprehensive guide for running on RunPod
- `runpod_setup.py`: Setup script for RunPod

## Quick Start

1. Upload the project zip file to RunPod
2. Extract the zip file
3. Run the setup script:

```bash
python runpod/runpod_setup.py
```

4. Install TensorBoard dependencies:

```bash
python install_tensorboard.py
```

5. Run a single experiment to verify everything works:

```bash
python scripts/debug_experiment.py --batch_size 32 --lr 0.001 --epochs 5
```

6. Run the full experiment suite:

```bash
python scripts/run_experiments.py
```

## Detailed Instructions

For detailed instructions on running the project on RunPod, please refer to `runpod_guide.md`.

## Troubleshooting

If you encounter any issues:

1. Check that all dependencies are installed:

```bash
python install.py
```

2. Verify GPU availability:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

3. Check for error messages in the output

4. Try running with a smaller batch size if you encounter CUDA out of memory errors

## Downloading Results

After training is complete, you can download the results:

```bash
zip -r results.zip results saved_models logs
```

Then download the `results.zip` file from RunPod.