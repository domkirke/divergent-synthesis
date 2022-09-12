import torch, torch.nn as nn, pdb

#%% Pixel Normalization

class PixelNorm_(nn.Module):
    eps = 10**(-8)
    def __init__(self, dim=None, eps=None):
        super().__init__()
        if dim is None:
            raise ValueError("dim keyword must not be none")
        self.dim = dim
        self.eps = eps if eps is not None else self.eps

    def forward(self, x):
        weight = (x.pow(2).sum(-(self.dim+1)) + self.eps).sqrt().unsqueeze(-(self.dim+1))
        return x / weight

class PixelNorm1d(PixelNorm_):
    def __init__(self, eps=None) -> None:
        super(PixelNorm1d, self).__init__(dim=1, eps=eps)

class PixelNorm2d(PixelNorm_):
    def __init__(self, eps=None) -> None:
        super(PixelNorm2d, self).__init__(dim=2, eps=eps)

class PixelNorm3d(PixelNorm_):
    def __init__(self, eps=None) -> None:
        super(PixelNorm3d, self).__init__(dim=3, eps=eps)


#%% Activation Normalization
#TODO 

class ActNorm_(nn.Module):
    eps = 10**(-8)
    def __init__(self, dim=None, eps=None):
        super().__init__()
        if dim is None:
            raise ValueError("dim keyword must not be none")
        self.dim = dim
        self.eps = eps if eps is not None else self.eps
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
        

class ActNorm1d(ActNorm_):
    def __init__(self, eps=None) -> None:
        super(ActNorm1d, self).__init__(dim=1, eps=eps)

class ActNorm2d(ActNorm_):
    def __init__(self, eps=None) -> None:
        super(ActNorm2d, self).__init__(dim=2, eps=eps)

class ActNorm3d(ActNorm_):
    def __init__(self, eps=None) -> None:
        super(ActNorm3d, self).__init__(dim=3, eps=eps)

#%% Batch Std Normalization
#TODO