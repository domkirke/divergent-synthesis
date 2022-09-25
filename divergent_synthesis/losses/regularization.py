import sys, torch, torch.nn as nn, torch.distributions as dist
sys.path.append('../')
from divergent_synthesis.losses.loss import Loss
from torch.distributions import Distribution
from typing import Callable, Union


class KLD(Loss):
    def __repr__(self):
        return "KLD()"

    def forward(self, params1: dist.Distribution, params2: dist.Distribution, drop_detail=False, **kwargs) -> torch.Tensor:
        """
        Wrapper for Kullback-Leibler Divergence.
        Args:
            params1 (dist.Distribution): first distribution
            params2: (dist.Distribution) second distribution

        Returns:
            kld (torch.Tensor): divergence output

        """
        reduction = kwargs.get('reduction', self.reduction)
        assert isinstance(params1, dist.Distribution) and isinstance(params2, dist.Distribution), \
            "KLD only works with two distributions"
        ld = self.reduce(dist.kl_divergence(params1, params2), reduction=reduction)
        if drop_detail:
            return ld, {'kld': ld}
        else:
            return ld

def l2_kernel(x, y, scale=1.0):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    loss = torch.exp(-(x.unsqueeze(1).expand(x_size, y_size, dim) - y.unsqueeze(0).expand(x_size, y_size, dim)).pow(2) / float(scale))
    return loss

def l2_multiscale_kernel(x, y, scales=[0.2, 0.5, 0.9, 1.3]):
    loss = 0
    for scale in scales:
        loss = loss + l2_kernel(x, y, scale=scale)
    return loss

kernel_hash = {"l2": l2_kernel, "l2_multiscale": l2_multiscale_kernel}


class MMD(Loss):
    def __repr__(self):
        return "MMD(kernel=%s)"%self.kernel

    def __init__(self, kernel: Union[Callable, str] = l2_kernel, *args, reduction: bool = None, kernel_args={}, **kwargs):
        """
        Maximum Mean Discrepency (MMD) performs global distribution matching, in order to regularize q(z) rather that
        q(z|x). Used in Wasserstein Auto-Encoders.
        Args:
            kernel (Callable or str): kernel used (default: l2_kernel)
        """
        super(MMD, self).__init__(reduction=reduction)
        if isinstance(kernel, str):
            assert kernel in kernel_hash.keys(), "kernel keyword must be %s"%list(kernel_hash.keys())
            kernel = kernel_hash[kernel]
        self.kernel = kernel
        self.kernel_args = kernel_args

    def forward(self, params1: Distribution = None, params2: Distribution = None, drop_detail:bool = False, **kwargs) -> torch.Tensor:
        assert params1 is not None, params2 is not None
        reduction = kwargs.get('reduction', self.reduction)
        if isinstance(params1, Distribution):
            params1 = params1.sample() if not params1.has_rsample else params1.rsample()
        if isinstance(params2, Distribution):
            params2 = params2.sample() if not params2.has_rsample else params2.rsample()
        dim1 = params1.shape[-1]
        dim2 = params2.shape[-1]
        params1 = params1.view(-1, dim1)
        params2 = params2.view(-1, dim2)
        x_kernel = self.kernel(params1, params1, **self.kernel_args) / (dim1 * (dim1 - 1))
        y_kernel = self.kernel(params2, params2, **self.kernel_args) / (dim2 * (dim2 - 1))
        xy_kernel = self.kernel(params1, params2, **self.kernel_args) / (dim2 * dim1)
        loss = x_kernel.sum() + y_kernel.sum() - 2*xy_kernel.sum()
        if drop_detail:
            return loss, {'mmd': loss.detach().cpu()}
        else:
            return loss

