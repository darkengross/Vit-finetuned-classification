import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
import json, os, math
from torch.nn import functional as F

def save_model(model, config, exp_name, models_dir='models'):
    """
    Save the trained model and its configuration to the specified directory.
    
    Args:
    - model: The trained model to be saved.
    - config: Configuration object used to create the model.
    - exp_name: Name of the experiment or model.
    - models_dir: Directory where models will be saved.
    """
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Remove invalid characters from exp_name
    exp_name = exp_name.replace("*", "_")  # Replace '*' with '_'
    
    model_path = os.path.join(models_dir, f"{exp_name}.pt")
    torch.save({'model_state_dict': model.state_dict(), 'config': config}, model_path)
    print(f"Model saved at: {model_path}")

def load_model(model_class, exp_name, models_dir='models'):
    """
    Load a previously saved model.
    
    Args:
    - model_class: The class of the model to be loaded.
    - exp_name: Name of the experiment or model.
    - models_dir: Directory where models are saved.
    
    Returns:
    - model: The loaded model.
    """
    model_path = os.path.join(models_dir, f"{exp_name}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model '{exp_name}' not found in directory '{models_dir}'")
    
    checkpoint = torch.load(model_path)
    config = checkpoint['config']
    
    model = model_class(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from: {model_path}")
    return model