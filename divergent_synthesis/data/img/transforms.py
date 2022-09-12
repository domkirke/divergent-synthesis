from lib2to3.pytree import Base
from tkinter import E
import torch.nn as nn, torch
import torchvision.transforms as tv

class NotInvertibleError(BaseException):
    def __init__(self, obj):
        self.typeobj = type(obj)
        
    def __repr__(self):
        return "NotInvertibleError(%s)"%str(self.typeobj)


class Wrapper(nn.Module):
    def __init__(self, transform) -> None:
        super().__init__()
        self.transform = transform

    def forward(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    def invert(self, x):
        return x


class Compose(tv.Compose):
    def invert(self, x):
        for t in reversed(self.transforms):
            if hasattr(t, "invert"):
                x = t.invert(x)
            else:
                # raise NotInvertibleError(t)
                continue
        return x

class Binarize(nn.Module):
    def __init__(self, threshold=0.5) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(self, x):
        return (x > self.threshold).to(torch.get_default_dtype())

    def invert(self, x):
        return x

class Rescale(nn.Module):
    def __init__(self, mode="bipolar", invert_as_byte=False):
        super().__init__()
        self.mode = mode
        self.invert_as_byte = invert_as_byte

    def forward(self, x):
        if isinstance(x, torch.ByteTensor):
            x = x.float() / 255.
        if self.mode == "bipolar":
            x = (x * 2) - 1
        return x

    def invert(self, x):
        if self.mode == "bipolar":
            x = (x + 1) / 2
        if self.invert_as_byte:
            x = (x * 255).byte()
        return x            
        