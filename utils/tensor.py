import torch

from definitions import config

next_device = 0


def get_device(device_id=None):
    global next_device

    if device_id is None:
        device_to_use = next_device
    else:
        device_to_use = device_id

    device_name = "cpu"
    if torch.cuda.is_available():
        device_name = "cuda:{}".format(device_to_use)
        print('Cuda device:', device_to_use)
    else:
        print('Cuda not available')

    device = torch.device(device_name)
    if device.type == 'cuda':
        capability = torch.cuda.get_device_capability(device)
        device = device if capability[0] > 3 or (capability[0] == 3 and capability[1] >= 5) else torch.device("cpu")

    if device.type == 'cuda' and device_id is None:
        next_device = (next_device + 1) % torch.cuda.device_count()

    return device


class TensorWrapper:

    def __init__(self, device_id=None):
        self.device = get_device(device_id)

    def __call__(self, *args, **kwargs):
        t = args[0]
        if isinstance(t, torch.Tensor):
            return t.to(self.device)
        return torch.tensor(args[0], device=self.device)


tw = TensorWrapper()


def to_numpy(*tensors):
    if len(tensors) == 1:
        return tensors[0].detach().cpu().numpy()
    return tuple(map(lambda t: t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t, tensors))


def num_workers():
    workers = config.get('torch', 'num_workers') if tw.device.type == 'cuda' else 0
    return workers
