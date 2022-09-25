import abc, torch, torch.nn as nn
from omegaconf import OmegaConf


class Generator(nn.Module):
    def __init__(self, config: OmegaConf):
        """General class for generators. Automatically parses configuration

        Args:
            config (Union[str, OmegaConf, dict]): generator configuration
        """
        super(Generator, self).__init__()
        if isinstance(config, str):
            self.config = OmegaConf.load(str)
        elif config is None:
            self.config = OmegaConf.create()
        else:
            self.config = OmegaConf.create(config)

    @abc.abstractmethod
    def forward(self,
                t: torch.Tensor = None,
                z: torch.Tensor = None,
                y: torch.Tensor = None,
                parameters = None,
                data: torch.Tensor = None):
        raise NotImplementedError
        