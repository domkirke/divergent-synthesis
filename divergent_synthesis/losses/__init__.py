from omegaconf import OmegaConf, ListConfig
from .loss import Loss, LossContainer


# distortion losses
from . import distortion
from .distortion import *
def get_distortion_loss(config):
    if isinstance(config, ListConfig):
        losses =[get_distortion_loss(l) for l in config]
        return sum(losses[1:], losses[0])
    else:
        loss_type = config.get('type')
        if loss_type is None:
            raise ValueError("key missing in loss configuration : type")
        loss_args = config.get('args', {})
        loss = getattr(distortion, loss_type)(**loss_args)
        if config.get('weight'):
            loss = config.weight * loss
        return loss


# regularization losses
from . import regularization
from .regularization import *
def get_regularization_loss(config):
    if isinstance(config, ListConfig):
        losses =[get_regularization_loss(l) for l in config]
        return sum(losses[1:], losses[0])
    else:
        loss_type = config.get('type')
        if loss_type is None:
            raise ValueError("key missing in loss configuration : type")
        loss_args = config.get('args', {})
        loss = getattr(regularization, loss_type)(**loss_args)
        if config.get('weight'):
            loss = config.weight * loss
        return loss

# generative losses
from . import generative
from .generative import *