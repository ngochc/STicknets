# STicknet

## Installation

### Prerequisites

This project requires Python 3.10. The project uses `.tool-versions` to specify the Python version.

### Setup Instructions

1. **Install Python 3.10**
   ```bash
   # If using asdf (recommended)
   asdf install python 3.10.14
   asdf local python 3.10.14
   
   # Or download from python.org
   # https://www.python.org/downloads/release/python-31014/
   ```

2. **Create a virtual environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On macOS/Linux:
   source venv/bin/activate
   
   # On Windows:
   # venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   # Upgrade pip
   pip install --upgrade pip
   
   # Install requirements
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   # Check Python version
   python --version
   
   # Should output: Python 3.10.14
   ```

### Development Setup

For development, you may want to install additional tools:

```bash
# Install development dependencies (if any)
pip install -r requirements-dev.txt  # if exists

# Install pre-commit hooks (if configured)
pre-commit install
```

## Usage

### Training with S_TickNet_Dogs.py

The `S_TickNet_Dogs.py` script trains a Spatial TickNet model on the Stanford Dogs dataset. Here are some example usage patterns:

#### Basic Training
```bash
# Train with default settings (Stanford Dogs dataset)
python S_TickNet_Dogs.py

# Train with custom data path
python S_TickNet_Dogs.py --data-root /path/to/StanfordDogs

# Train with specific architecture type
python S_TickNet_Dogs.py --architecture-types basic
```

#### Advanced Training Options
```bash
# Train with custom hyperparameters
python S_TickNet_Dogs.py \
    --batch-size 32 \
    --epochs 100 \
    --learning-rate 0.01 \
    --gpu-id 0 \
    --workers 8

# Train with learning rate scheduling
python S_TickNet_Dogs.py \
    --learning-rate 0.1 \
    --schedule 50 75 90 \
    --epochs 100

# Train with different architecture configurations
python S_TickNet_Dogs.py \
    --architecture-types basic \
    --config 0
```

#### Evaluation Mode
```bash
# Evaluate a trained model
python S_TickNet_Dogs.py --evaluate
```

#### Dataset Options
```bash
# Download dataset automatically
python S_TickNet_Dogs.py --download

# Use different datasets (CIFAR-10, CIFAR-100, Stanford Dogs)
python S_TickNet_Dogs.py --dataset cifar10
python S_TickNet_Dogs.py --dataset cifar100
python S_TickNet_Dogs.py --dataset dogs  # default
```

#### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data-root` | Dataset root path | `../../../datasets/StanfordDogs` |
| `--dataset` | Dataset name (cifar10, cifar100, dogs) | `dogs` |
| `--architecture-types` | List of architecture types | `['basic']` |
| `--download` | Download dataset before training | `False` |
| `--gpu-id` | GPU ID to use (-1 for CPU) | `1` |
| `--workers` | Number of data loading workers | `4` |
| `--batch-size` | Batch size | `64` |
| `--epochs` | Number of training epochs | `200` |
| `--learning-rate` | Initial learning rate | `0.1` |
| `--schedule` | Learning rate schedule epochs | `[100, 150, 180]` |
| `--momentum` | SGD momentum | `0.9` |
| `--weight-decay` | SGD weight decay | `1e-4` |
| `--base-dir` | Base directory for checkpoints | `.` |
| `--evaluate` | Evaluate model on validation set | `False` |
| `--config` | Configuration index | `0` |

#### Output

The script will:
- Save model checkpoints in `checkpoints/StanfordDogs_S_TickNet_{architecture_type}_SE_config_{config}/`
- Log training progress to a `.txt` file
- Save results to a `.csv` file
- Print training metrics to console

### Interactive Examples with Jupyter Notebooks

For interactive exploration and experimentation, check out the notebooks in the `notebooks/` folder:

```bash
# Install Jupyter (if not already installed)
pip install jupyter

# Start Jupyter Notebook
jupyter notebook

# Navigate to notebooks/STickNet_Example.ipynb
```

The example notebook provides:
- Step-by-step model creation and training
- Interactive visualization of training progress
- Inference examples
- Model saving and loading demonstrations

See `notebooks/README.md` for detailed documentation about the available notebooks.

## Project Structure

- `models/` - Neural network model definitions
  - `common.py` - Common layers and utilities
  - `datasets.py` - Dataset handling
  - `SE_Attention.py` - Squeeze-and-Excitation attention mechanism
  - `TickNet.py` - Main TickNet model
- `notebooks/` - Jupyter notebooks with examples
  - `STickNet_Example.ipynb` - Comprehensive training and inference example
  - `README.md` - Notebook documentation
- `S_TickNet_Dogs.py` - Script for TickNet with dogs dataset
- `TickNet_Dogs.py` - Alternative TickNet implementation
- `writeLogAcc.py` - Logging and accuracy utilities