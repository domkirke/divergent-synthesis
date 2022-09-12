import torch, os, dill
from omegaconf import OmegaConf
from divergent_synthesis import models
from divergent_synthesis.utils.save import get_model_versions


def load_model_from_run(run_path, get_last=False, version=None, map_location=torch.device("cpu")):
    run_name = os.path.basename(run_path)
    if get_last:
        assert version is None, "get last and version keywords are incompatible. Choose only one."
    if version is None:
        versions = get_model_versions(run_path)
        if len(versions) > 0:
            version = max(versions)
            config_name = transform_name = model_name = f"{run_name}-v{version}"
        else:
            version = 1
            if os.path.isfile(os.path.join(run_path, "models", f"{run_name}-v1.ckpt")):
                config_name = transform_name = model_name = f"{run_name}-v1"
            else:
                model_name = f"{run_name}"
                config_name = transform_name = f"{run_name}-v1"
    if get_last:
        model_path = os.path.join(run_path, "models", f"last.ckpt")
    else:
        model_path = os.path.join(run_path, "models", f"{model_name}.ckpt")
    config_path = os.path.join(run_path, "configs", f"{config_name}.yaml")
    transform_path = os.path.join(run_path, "transforms", f"{transform_name}.ct")

    config = OmegaConf.load(config_path)
    model_type = getattr(models, config.model.type)
    model = model_type.load_from_checkpoint(model_path, map_location=map_location)
    with open(transform_path, "rb") as f:
        transform = dill.load(f)
    return model, config, transform


def load_vae(name: str, root: str = "models/vae/"):
    model_path = os.path.join(root, name)
    model, config, transform = load_model_from_run(model_path)
    model.eval()
    if torch.cuda.is_available():
        cuda_id = int(os.environ.get('CUDA_DEVICE', 0))
        model = model.to("cuda:%d"%cuda_id)
    return model, config, transform

def load_gan(name: str, root: str = "models/gan/"):
    model_path = os.path.join(root, name)
    model, config, transform = load_model_from_run(model_path)
    model.eval()
    if torch.cuda.is_available():
        cuda_id = int(os.environ.get('CUDA_DEVICE', 0))
        model = model.to("cuda:%d"%cuda_id)
    return model, config, transform