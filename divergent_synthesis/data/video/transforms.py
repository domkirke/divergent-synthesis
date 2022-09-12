import torch, torch.nn as nn

class VideoTransform(nn.Module):
    invertble = True
    scriptable = True
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
    def invert(self, x: torch.Tensor) -> torch.Tensor:
        return x