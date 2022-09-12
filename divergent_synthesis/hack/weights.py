import torch, torch.nn as nn, numpy as np, pdb

def normal_(parameter, mean=0.0, std=0.1):
    parameter.data = parameter.data + mean + std * torch.randn_like(parameter.data)

def mask_(parameter, prob=0.2):
    mask = torch.full_like(parameter.data, prob).to(parameter.device)
    noise = torch.bernoulli(1-mask)
    parameter.data = parameter.data * noise

def scale_(parameter, weight=1.0):
    parameter.data = parameter.data * weight

def bias_(parameter, bias=0.0):
    parameter.data = parameter.data + bias

def permute_(parameter, axis=0):
    perm = torch.randperm(parameter.shape[axis])
    parameter.data = parameter.data.index_select(axis, perm)

def cos_(parameter, freq=1.0, phase=0.0):
    parameter.data = torch.cos(freq * parameter.data + phase)
