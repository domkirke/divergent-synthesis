import torch, torch.nn as nn, torch.distributions as dist
from acids_transforms import OneHot
from .base import Discriminator
from omegaconf import OmegaConf
from ..utils import checklist
from torch.nn.functional import softmax
from .. import modules

class ConvDiscriminator(Discriminator):
    default_module = "ConvEncoder"
    def __init__(self, config: OmegaConf):
        super(ConvDiscriminator, self).__init__(config)
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

class ConvAuxDiscriminator(Discriminator):
    default_module = "ConvEncoder"
    def __init__(self, config: OmegaConf):
        super(ConvAuxDiscriminator, self).__init__(config)
        self._init_parameters(config)
        
    def _init_parameters(self, config: OmegaConf):
        self.y_config = config.get('classes')
        disc_config = config.get('discriminator')
        # check output shape
        self.input_shape = config.input_shape
        # checking input shape
        class_dim = 0
        onehot_embeddings = {}
        for task, task_config in self.y_config.items():
            if "cat" in checklist(task_config.get('mode', 'cat')):
                class_dim += task_config['dim']
                onehot_embeddings[task] = OneHot(n_classes=class_dim)
        self.onehot_embeddings = nn.ModuleDict(onehot_embeddings)
        disc_config.args.input_shape = config.input_shape
        disc_config.args.target_shape = (class_dim + 1,)
        self.target_shape = (class_dim + 1, )
        # make disc
        disc_type = getattr(modules.encoders, disc_config.args.get('module', self.default_module))
        self.discriminator = disc_type(disc_config.args)

    def forward(self, x: torch.Tensor = None, trace = None, return_hidden = False, **kwargs):
        disc_out = self.discriminator(x, trace=trace, return_hidden=return_hidden)
        if return_hidden:
            disc_out, hidden_out = disc_out
        domain_disc = {}
        domain_disc['domain'] = disc_out[..., 0]
        current_idx = 1
        for task, task_config in self.y_config.items():
            current_dim = task_config['dim']
            domain_disc[task] = dist.Categorical(probs=softmax(disc_out[..., current_idx:current_idx+current_dim], -1))
        if return_hidden:
            return domain_disc, hidden_out
        else:
            return domain_disc

    def get_features(self, x:torch.Tensor):
        disc_out, hidden = self.discriminator(x, return_hidden=True)
        return hidden[-1]