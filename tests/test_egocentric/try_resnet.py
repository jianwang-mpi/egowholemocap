import torch
import torch.nn as nn

proj = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
proj.fc = nn.Linear(2048, 768)

x = torch.zeros(4, 3, 32, 32)

res = proj(x)

print(res.shape)