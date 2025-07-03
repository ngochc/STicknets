# STickNet Notebooks

This folder contains Jupyter notebooks demonstrating how to use the STickNet models.

## Available Notebooks

### STickNet_Cifar100.ipynb
A notebook demonstrating:

- **Model Creation**: How to create and configure STickNet models for CIFAR-100
- **Data Loading**: Setting up data transforms and loaders for CIFAR-100
- **Training Setup**: Configuring loss functions, optimizers, and schedulers
- **Training Loop**: Complete training and validation process
- **Visualization**: Plotting training curves and metrics
- **Inference**: Making predictions on new images
- **Model Persistence**: Saving and loading trained models

### STickNet_StanfordDogs.ipynb
A notebook demonstrating:

- **Model Creation**: How to create and configure STickNet models for Stanford Dogs
- **Data Loading**: Setting up data transforms and loaders for Stanford Dogs
- **Training Setup**: Configuring loss functions, optimizers, and schedulers
- **Training Loop**: Complete training and validation process
- **Visualization**: Plotting training curves and metrics
- **Inference**: Making predictions on new images
- **Model Persistence**: Saving and loading trained models

## Getting Started

1. **Install Jupyter** (if not already installed):
   ```bash
   pip install jupyter
   ```

2. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

3. **Open a notebook**:
   - Navigate to the `notebooks/` folder
   - Open `STickNet_Cifar100.ipynb` or `STickNet_StanfordDogs.ipynb`

## Prerequisites

Before running the notebooks, make sure you have:

1. **Installed all dependencies** from the main `requirements.txt`
2. **Downloaded the required dataset(s)** (update the data path in the notebook as needed)
3. **Set up your Python environment** as described in the main README

## Dataset Setup

The notebooks expect the relevant datasets to be available. You can:

1. **Download automatically** by setting `download=True` in the dataset creation (if supported)
2. **Use your own path** by updating the `data_root` variable in the notebook
3. **Use a different dataset** by modifying the dataset class and transforms

## Customization

Feel free to modify the notebooks to:

- Use different datasets (CIFAR-10, CIFAR-100, Stanford Dogs, etc.)
- Experiment with different model configurations
- Try different hyperparameters
- Add your own analysis and visualization

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're running the notebook from the project root directory
2. **CUDA out of memory**: Reduce the batch size in the notebook
3. **Dataset not found**: Update the `data_root` path to point to your dataset location

### Getting Help

- Check the main README for installation instructions
- Review the `S_TickNet_Dogs.py` script for advanced usage examples
- Ensure all dependencies are properly installed
