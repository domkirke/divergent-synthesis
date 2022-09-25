import torch, torch.nn as nn
from acids_transforms import OneHot
from .base import Generator
from omegaconf import OmegaConf
from ..modules import encoders
from divergent_synthesis.utils import checklist


class ConvGenerator(Generator):
    default_module = "DeconvEncoder"
    def __init__(self, config: OmegaConf):
        super(ConvGenerator, self).__init__(config)
        self._init_parameters(config)
        
    def _init_parameters(self, config: OmegaConf):
        self.y_config = config.get('classes')
        self.z_config = config.get('latent')
        generator_config = config.get('generator')
    
        # check output shape
        self.target_shape = config.input_shape
        
        # checking input shape
        latent_dim = self.z_config.get('dim')
        input_dim = latent_dim
        if self.y_config is not None:
            class_dim = 0
            onehot_embeddings = {}
            for task, task_config in self.y_config.items():
                if "cat" in checklist(task_config.get('mode', 'cat')):
                    class_dim += task_config['dim']
                    onehot_embeddings[task] = OneHot(n_classes=class_dim)
            self.onehot_embeddings = nn.ModuleDict(onehot_embeddings)
            input_dim += class_dim
        generator_config.args.input_shape = input_dim
        generator_config.args.target_shape = config.input_shape
        self.input_shape = (input_dim,)

        # make generator
        generator_type = getattr(encoders, generator_config.args.get('module', self.default_module))
        self.generator = generator_type(generator_config.args)

    def forward(self, z: torch.Tensor = None, y: torch.Tensor = None, trace = None, **kwargs):
        assert z is not None
        decoder_input = [z]
        if self.y_config is not None:
            for task, embed in self.onehot_embeddings.items():
                decoder_input.append(embed(y[task]))
        decoder_input = torch.cat(decoder_input, -1)
        return self.generator(decoder_input, trace=trace)