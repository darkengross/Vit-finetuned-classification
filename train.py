# Define a configuration dictionary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
import json, os, math
from torch.nn import functional as F 

device = "cuda" if torch.cuda.is_available() else "cpu"

config = {
    "patch_size": 4,  # Input image size: 32x32 -> 8x8 patches
    "hidden_size": 48,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 4 * 48, # 4 * hidden_size
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "image_size": 128,
    "num_classes": 6, # num_classes of Intel Dataset
    "num_channels": 3,
    "qkv_bias": True,
    "use_faster_attention": True,
}


exp_name = 'vit-10epochs_with_interpolation' #@param {type:"string"}
batch_size = 32 #@param {type: "integer"}
epochs = 10 #@param {type: "integer"}
lr = 1e-2  #@param {type: "number"}
save_model_every = 0 #@param {type: "integer"}

assert config["hidden_size"] % config["num_attention_heads"] == 0
assert config['intermediate_size'] == 4 * config['hidden_size']
assert config['image_size'] % config['patch_size'] == 0

save_model_every_n_epochs = save_model_every
model = ViTForClassfication(config)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
loss_fn = nn.CrossEntropyLoss()
trainer = Trainer(model, optimizer, loss_fn, exp_name, device=device)
output, train_losses, test_losses, accuracies = trainer.train(train_loader, test_loader, epochs, save_model_every_n_epochs=save_model_every_n_epochs)