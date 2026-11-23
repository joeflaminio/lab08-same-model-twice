# Lab 07: First Principles (Neural Network for Bike Data)

A modular PyTorch implementation predicting bicycle-related metrics from scratch using first principles.

## Project Structure
- `bikes_data.py`: Synthetic/processed dataset generation.
- `prepare.py`: Data splitting and PyTorch DataLoader utilities.
- `model.py`: `BikeNet` multi-layer perceptron architecture.
- `train.py`: Training loop, evaluation, baseline comparisons, and curve plotting.
- `tests/`: Unit tests verifying data pipelines and model forward passes.
- `reports/`: Generated training curve visualizations.

## Experimental Results
After training for 30 epochs ($\text{lr} = 1e-2$):
- **Mean Baseline Test MSE**: 198.2915
- **BikeNet Test MSE**: 113.0288

## Running the Pipeline
python train.py
pytest
