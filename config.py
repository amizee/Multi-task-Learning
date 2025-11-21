import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASSES = [
    '__background__', 'red', 'yellow', 'green', 'off'
]