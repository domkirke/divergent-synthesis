import torch, os, dill
from omegaconf import OmegaConf
from divergent_synthesis import models

def get_model_from_path(model_path: str):
    if os.path.isfile(model_path):
        assert os.path.splitext(model_path)[1] == ".ckpt", "%s does not seem to be a valid file"%model_path
        yaml_paths = [os.path.dirname(model_path), os.path.abspath(os.path.dirname(model_path)+"/..")]
    else:
        model_path = os.path.join(model_path, "last.ckpt")
        yaml_paths = [os.path.dirname(model_path)]
        assert os.path.isfile(model_path), "model not found in folder %s"%model_path
    yaml_files = []
    for yp in yaml_paths:
        yaml_files.extend([os.path.join(yp, f) for f in os.listdir(yp)])
    yaml_files = list(filter(lambda x: os.path.splitext(x)[1] == ".yaml", yaml_files))
    if len(yaml_files) > 1:
        print('[Warning] ambiguous ; retrieved yaml file %s'%yaml_files[0])
    config = OmegaConf.load(os.path.abspath(yaml_files[0]))
    model = getattr(models, config.model.type)(config=config.model)
    model = model.load_from_checkpoint(model_path, config=config.model)
    return model, config