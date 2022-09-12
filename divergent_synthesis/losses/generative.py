from cgitb import reset
from inspect import formatannotationrelativeto
import torch
from torchmetrics.image import inception, fid
from .loss import Loss


class InceptionScore(Loss):
    def __repr__(self):
        return "InceptionScore(feature=%s, splits=%s)"%(self.feature, self.splits)
    
    def __init__(self, feature='logits_unbiased', splits=10):
        self.feature = feature
        self.splits = splits
        self.inception_module = inception.InceptionScore(feature=feature, splits=splits).eval()

    def forward(self, x: torch.Tensor, drop_detail: bool = False, **kwargs):
        kld_mean, kld_std = self.inception_module.compute(x)
        if drop_detail:
            return kld_mean, {'inception_mean': kld_mean.detach().cpu(), 'inception_std': kld_std.detach().cpu()}
        else:
            return kld_mean


class FID(Loss):
    def __repr__(self):
        return "FID()"

    def __init__(self, feature, reset_real_features = False):
        self.feature = feature
        self.reset_real_features = reset_real_features
        self.inception_module = fid.FrechetInceptionDistance(feature, reset_real_features=reset_real_features)

    def forward(self, x: torch.Tensor=None, target: torch.Tensor=None, drop_detail: bool = False, **kwargs):
        self.inception_module.update(target, real=True)
        self.inception_module.update(x, real=False)
        loss = self.inception_module.compute()
        if drop_detail:
            return loss, {'fid': loss.detach().cpu()}
        else:
            return loss