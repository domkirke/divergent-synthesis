import pdb, torch, torch.nn as nn, re
hooks_dict = {}

def apply_hooks(module, input):
    if id(module) in hooks_dict.keys():
        for hook_func in hooks_dict[id(module)]:
            input = hook_func(module, input)
    return input

def record_hook(module, module_name, hook_function):
    attributes = module_name.split('.')
    if module_name != "":
        for i, a in enumerate(attributes):
            list_index = re.match(r"(.*)\[([0-9]+)\]", a)
            if list_index is None:
                module = getattr(module, a)
            else:
                a, index = list_index.groups()
                module = getattr(module, a).__getitem__(int(index))
            if isinstance(module, nn.ModuleList):
                [record_hook(m, ".".join(attributes[i+1:]), hook_function) for m in module]
    if id(module) in hooks_dict:
        hooks_dict[id(module)].append(hook_function)
    else:
        hooks_dict[id(module)] = [hook_function]

def hook_test(*args, **kwargs):
    def hook_test(module, input):
        pass
    return hook_test

def normal_(mean=0.0, std=0.1):
    def normal_(module, input):
        input = input + mean + std * torch.randn_like(input)
        return input
    return normal_

def mask_(prob=0.2):
    def salt_and_pepper_(module, input):
        input = list(input)
        for i in range(len(input)):
            mask = torch.full_like(input[i], prob).to(input[i].device)
        noise = torch.bernoulli(1-mask)
        input[i] = input[i] * noise
        return tuple(input)
    return salt_and_pepper_

def scale_(weight=1.0):
    def scale_(module, input):
        input = [i * weight for i in input]
        return tuple(input)
    return scale_

def setattr_(name=None, value=None):
    def setattr_(module, input):
        setattr(module, name, value)
        return input
    return setattr_

def bias_(bias=0.0):
    def bias_(module, input):
        input = input + bias
        return input
    return bias_

def permute_(axis=0):
    def permute_(module, input):
        input = list(input)
        for i, inp in enumerate(input):
            perm = torch.randperm(inp.shape[axis])
            input[i] = inp.index_select(axis, perm)
        return tuple(input)
    return permute_

def cos_(freq=1.0, phase=0.0):
    def cos_(module, input):
        input = [torch.cos(freq * i + phase) for i in input]
        return tuple(input)
    return cos_