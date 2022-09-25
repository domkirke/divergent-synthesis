import torch, torch.nn as nn, torch.distributions as dist
from acids_transforms import OneHot
from .base import Discriminator
from omegaconf import OmegaConf
from ..utils import checklist
from torch.nn.functional import softmax
from .. import modules

class MLPDiscriminator(Discriminator):
    default_module = "MLPEncoder"
    def __init__(self, config: OmegaConf):
        super(MLPDiscriminator, self).__init__(config)
        self._init_parameters(config)
        
    def _init_parameters(self, config: OmegaConf):
        self.y_config = config.get('classes')
        disc_config = config.get('discriminator')
        # check output shape
        self.input_shape = config.input_shape
        # checking input shape
        disc_config.args.input_shape = config.input_shape
        disc_config.args.target_shape = (1,)
        self.target_shape = (1, )
        # make disc
        disc_type = getattr(modules.encoders, disc_config.args.get('module', self.default_module))
        self.discriminator = disc_type(disc_config.args)

    def forward(self, x: torch.Tensor = None, trace = None, return_hidden = False, **kwargs):
        disc_out = self.discriminator(x, trace=trace, return_hidden=return_hidden)
        if return_hidden:
            disc_out, hidden_out = disc_out
        domain_disc = {}
        domain_disc['domain'] = disc_out[..., 0]
        if return_hidden:
            return domain_disc, hidden_out
        else:
            return domain_disc