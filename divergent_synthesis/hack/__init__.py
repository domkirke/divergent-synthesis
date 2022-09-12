from . import weights
from . import hooks
import re, torch, pdb
from omegaconf import OmegaConf

def hack_model(model, hack_config):
    if isinstance(hack_config, dict):
        hack_config = OmegaConf.create(hack_config)
    if not hack_config.get('weights'):
        return
    parameters_regex = list(hack_config.weights.keys())
    for param_name, param_value in model.named_parameters():
        for target_param in parameters_regex:
            if re.match(target_param, param_name):
                for callback_config in hack_config.weights.get(target_param):
                    print("corrputing", param_name)
                    callback_name = list(callback_config.keys())[0]
                    callback_args = callback_config.get(callback_name, {})
                    getattr(weights, callback_name)(param_value, **callback_args)

def hook_model(model, hack_config):
    if not hack_config.get('hooks'):
        return
    for module_name in list(hack_config.hooks.keys()):
        for hook_name, hook_args in hack_config.hooks.get(module_name).items():
            hooks.record_hook(model, module_name, getattr(hooks, hook_name)(**hook_args))
    torch.nn.modules.module.register_module_forward_pre_hook(hooks.apply_hooks)