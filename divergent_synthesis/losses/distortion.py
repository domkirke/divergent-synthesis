import abc
from typing import Iterable, List
import torch, torch.nn, sys, pdb
sys.path.append('../')
from divergent_synthesis.losses.loss import Loss
from divergent_synthesis.losses import loss_utils as utils
from divergent_synthesis.utils import checklist
import torch.distributions as dist


class ReconstructionLoss(Loss):
    def __init__(self, reduction=None):
        super(ReconstructionLoss, self).__init__(reduction=reduction)

    @property
    @abc.abstractmethod
    def name(self):
        pass

    @abc.abstractmethod
    def forward(self, x, target, drop_detail=False, **kwargs):
        pass

class LogDensity(Loss):
    def __repr__(self):
        return "LogDensity()"

    def forward(self, x, target, drop_detail = False, sample=True):
        if isinstance(target, dist.Distribution):
            if sample:
                if hasattr(target, "rsample"):
                    target = target.rsample()
                else:
                    target = target.sample()
            else:
                target = target.mean
                if target is None:
                    raise ValueError('Could not sample distribution %s in LogDensity'%target)
        ld = self.reduce(-x.log_prob(target))
        if drop_detail:
            return ld, {"log_density": ld.detach().cpu()}
        else:
            return ld


class MSE(Loss):
    def __repr__(self):
        return "MSE()"

    def __init__(self, reduction=None):
        super().__init__(reduction=reduction)

    def forward(self, x, target, drop_detail = False, sample=False, **kwargs):
        if isinstance(x, dist.Distribution):
            if sample:
                if x.has_rsample:
                    x = x.rsample()
                else:
                    if x.grad_fn is not None:
                        print('[Warning] sampling a tensor in a graph will cut backpropagation' )
                    x = x.sample()
            else:
                if isinstance(x, dist.Normal):
                    x = x.mean
                elif isinstance(x, (dist.Bernoulli, dist.Categorical)):
                    x = x.probs
        mse_loss = self.reduce(torch.nn.functional.mse_loss(x, target, reduction="none"))
        if drop_detail:
            return mse_loss, {"mse": mse_loss.detach().cpu()}
        else:
            return mse_loss


## Reconstruction losses specific to audio

class LogCosh(Loss):
    def __repr__(self):
        return "LogCosh()"
    def __init__(self, freq_weight=2e-5, freq_exponent=2, nfft=1024, **kwargs):
        super().__init__(**kwargs)
        self.freq_weight = freq_weight
        self.freq_exponent = freq_exponent
        self.nfft = nfft

    def forward(self, x, target, drop_detail=False):
        loss = torch.cosh(x - target).log().sum(-1)
        x_fft = torch.stft(x, self.nfft, self.nfft//2, onesided=True, return_as_complex=True).abs()
        target_fft = torch.stft(target, self.nfft, self.nfft//2, onesided=True, return_as_complex=True).abs()
        l1_mag = torch.abs(x_fft - target_fft)
        freq_bins = torch.arange(target_fft.shape[-1]).pow(self.freq_exponent)
        freq_bins = freq_bins.view(*(1,)*(len(freq_bins.shape)-1))
        loss = loss + freq_bins * l1_mag
        if drop_detail:
            return loss, {"log_cosh": loss.detach().cpu()}
        else:
            return loss

class ESR(Loss):
    """Error-Signal Rate"""
    def __repr__(self):
        return "ESR()"
    def forward(self, x, target, drop_detail=False):
        loss = self.reduce(torch.abs(x - target).pow(2).sum(-1) / target.pow(2).sum(-1))
        if drop_detail:
            return loss, {"esr": loss.detach().cpu()}
        else:
            return loss

class SpectralLoss(Loss):
    def __repr__(self):
        repr = "SpectralLoss("
        for i, l in enumerate(self.losses):
            if i == len(self.losses) - 1:
                repr += str(self.weights[i]) + "*" + l + ")"
            else:
                repr +=( str(self.weights[i]) + "*" + l + ", ")
        return repr


    def __init__(self, input_type: str = "raw", losses: Iterable[str] = ['mag_l2'], weights: Iterable[float] = [1.0],
                 losses_args: Iterable[dict] = None, nfft: int=512, hop_size:int=None, normalized:bool=False, reduction:str=None):
        super().__init__(reduction=reduction)
        self.input_type = input_type
        self.losses = losses
        self.weights = weights
        losses_args = losses_args or checklist({}, n=len(losses))
        self.losses_args = list(map(lambda x: x or {}, losses_args))
        self.nfft = nfft
        self.hop_size = hop_size or self.nfft // 4
        self.normalized = normalized

    def forward(self, x, target, drop_detail=False):
        if self.input_type == "raw":
            x_f = torch.stft(x, self.nfft, self.hop_size, normalized=self.normalized, onesided=True, return_complex=True)
            target_f = torch.stft(target, self.nfft, self.hop_size, normalized=self.normalized, onesided=True, return_complex=True)
            x_f = x_f.transpose(-2, -1); target_f = target_f.transpose(-2, -1)
        elif self.input_type == "mag":
            x_f = x * torch.exp(torch.zeros_like(x) * 1j)
            target_f = x * torch.exp(torch.zeros_like(target) * 1j)
        elif self.input_type == "phase_channel":
            x_f = x[..., 0, :] * torch.exp(x[..., 1, :] * 1j)
            target_f = target[..., 0, :] * torch.exp(target[..., 1, :] * 1j)
        elif self.input_type == "complex":
            x_f = x; target_f = target
        losses = []
        losses_dict = {}
        for i, loss_name in enumerate(self.losses):
            current_loss = self.reduce(getattr(utils, loss_name)(x_f, target_f, **self.losses_args[i]))
            losses.append(self.weights[i] * current_loss)
            losses_dict[loss_name] = current_loss
        if drop_detail:
            return sum(losses), losses_dict
        else:
            return sum(losses)

class MultiResSpectralLoss(Loss):
    def __init__(self, nffts=[256, 1024, 2048], hop_sizes=None, weights = None, reduction=None, **kwargs):
        super(MultiResSpectralLoss, self).__init__(reduction=reduction)
        hop_size = checklist(hop_sizes, n=len(nffts))
        self.weights = checklist(weights or 1, n=len(nffts))
        self.spectral_losses = []
        self.nffts = nffts
        for i, nfft in enumerate(nffts):
            hs = hop_size[i] or nfft // 4
            self.spectral_losses.append(SpectralLoss(nfft=nfft, hop_size=hs, **kwargs))

    def forward(self, x, target, drop_detail=False):
        losses = []
        losses_dict = {"stft-"+n for n in self.nffts}
        for i, loss in enumerate(self.spectral_losses):
            l_tmp = loss(x, target, drop_detail = drop_detail)
            if drop_detail:
                l_tmp, ls_tmp = l_tmp
                losses_dict["stft-"+self.nffts[i]] = ls_tmp
            losses.append(l_tmp)
        if self.drop_detail:
            return sum(losses), losses_dict
        else:
            return sum(losses)

#TODO Perceptual weighting (pre-emphasis)

