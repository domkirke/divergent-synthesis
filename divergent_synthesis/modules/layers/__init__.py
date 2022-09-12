from .. import norm
from torch import nn

DEFAULT_NNLIN = "ELU"
dropout_hash = {1: nn.Dropout, 2: nn.Dropout2d, 3: nn.Dropout3d}
bn_hash = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}
in_hash = {1: nn.InstanceNorm1d, 2: nn.InstanceNorm2d, 3: nn.InstanceNorm3d}
pixel_hash = {1: norm.PixelNorm1d, 2: norm.PixelNorm2d, 3: norm.PixelNorm3d}
act_hash = {1: norm.ActNorm1d, 2: norm.ActNorm2d, 3: norm.ActNorm3d}
norm_hash = {'batch': bn_hash, 'instance': in_hash, 'pixel': pixel_hash}

from .conv import *
from .fc import *
from .seq import *
from .blocks import *