import re, os, dill, torch
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule
from divergent_synthesis import models
from divergent_synthesis.utils import checkdir

def get_model_versions(folder):
    models_path = os.path.join(folder, "models")
    name = os.path.basename(folder)
    current_files = list(filter(lambda x: os.path.splitext(x)[1] == ".ckpt", os.listdir(models_path)))
    versions = list(filter(lambda x: x is not None, [re.match(f"{name}-v(\d+).ckpt", f) for f in current_files]))
    if len(versions) != 0:
        versions = list(map(int, [v[1] for v in versions]))
    return versions

def save_trainig_config(config: OmegaConf, data: LightningDataModule, path: str = None, name: str = None):
    """Saves training configurations and transforms in the training directory."""
    if path is None:
        path = config.rundir
    if name is None:
        name = config.name
    config_path = os.path.join(path, name)
    models_path = os.path.join(path, name, "models")
    checkdir(models_path)
    versions = get_model_versions(config_path)
    current_version = 1 if len(versions) == 0 else max(versions) + 1
    name = f'{name}-v{current_version}'
    checkdir(os.path.join(config_path, "configs"))
    with open(os.path.join(config_path, "configs", f"{name}.yaml"), "w+") as f:
        f.write(OmegaConf.to_yaml(config))
    if hasattr(data, "full_transforms"):
        checkdir(os.path.join(config_path, "transforms"))
        with open(os.path.join(config_path, "transforms", f"{name}.ct"), 'wb') as f:
            dill.dump(data.full_transforms, f)
    
    