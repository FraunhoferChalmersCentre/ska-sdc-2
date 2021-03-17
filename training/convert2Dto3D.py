from torch import nn
import numpy as np
import collections.abc
from itertools import repeat
import re
import torch


def _ntuple_same(n):
    def parse(x):
        if isinstance(x, int):
            return tuple(repeat(x, n))
        elif isinstance(x, collections.abc.Iterable):
            assert len(set(x)) == 1, 'the size of kernel must be the same for each side'
            return tuple(repeat(x[0], n))

    return parse


_triple_same = _ntuple_same(3)


class BaseConverter(object):
    """
    base class for converters
    """
    converter_attributes = []
    target_conv = None

    def __init__(self, model):
        """ Convert the model to its corresponding counterparts and deal with original weights if necessary """
        self.model = model
        self.linear = False

    def convert_module(self, module):
        """
        A recursive function.
        Treat the entire model as a tree and convert each leaf module to
            target_conv if it's Conv2d,
            3d counterparts if it's a pooling or normalization module,
            trilinear mode if it's a Upsample module.
        """
        for child_name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                arguments = nn.Conv2d.__init__.__code__.co_varnames[1:]
                kwargs = {k: getattr(child, k) for k in arguments}
                kwargs = self.convert_conv_kwargs(kwargs)
                setattr(module, child_name, self.__class__.target_conv(**kwargs))
            elif hasattr(nn, child.__class__.__name__) and \
                    ('pool' in child.__class__.__name__.lower() or
                     'norm' in child.__class__.__name__.lower()):
                if hasattr(nn, child.__class__.__name__.replace('2d', '3d')):
                    TargetClass = getattr(nn, child.__class__.__name__.replace('2d', '3d'))
                    arguments = TargetClass.__init__.__code__.co_varnames[1:]
                    kwargs = {k: getattr(child, k) for k in arguments}
                    if 'adaptive' in child.__class__.__name__.lower():
                        for k in kwargs.keys():
                            kwargs[k] = _triple_same(kwargs[k])
                    setattr(module, child_name, TargetClass(**kwargs))
                else:
                    raise Exception('No corresponding module in 3D for 2d module {}'.format(child.__class__.__name__))
            elif isinstance(child, nn.Upsample):
                arguments = nn.Upsample.__init__.__code__.co_varnames[1:]
                kwargs = {k: getattr(child, k) for k in arguments}
                kwargs['mode'] = 'trilinear' if kwargs['mode'] == 'bilinear' else kwargs['mode']
                setattr(module, child_name, nn.Upsample(**kwargs))
            elif isinstance(child, nn.Linear):
                self.linear = True
                self.convert_module(child)
            else:
                self.convert_module(child)
        return module

    def convert_conv_kwargs(self, kwargs):
        """
        Called by self.convert_module. Transform the original Conv2d arguments
        to meet the arguments requirements of target_conv.
        """
        raise NotImplementedError

    def __getattr__(self, attr):
        return getattr(self.model, attr)

    def __setattr__(self, name, value):
        if name in self.__class__.converter_attributes:
            return object.__setattr__(self, name, value)
        else:
            return setattr(self.model, name, value)

    def __call__(self, x):
        return self.model(x)

    def __repr__(self):
        return self.__class__.__name__ + '(\n' + self.model.__repr__() + '\n)'


class Conv3dConverter(BaseConverter):
    """
    Decorator class for converting 2d convolution modules
    to corresponding 3d version in any networks.

    Args:
        model (torch.nn.module): model that needs to be converted
    Warnings:
        Functions in torch.nn.functional involved in data dimension are not supported
    """
    converter_attributes = ['model']
    target_conv = nn.Conv3d

    def __init__(self, model, i3d_repeat_axis=None, r_input=None):
        super().__init__(model)
        preserve_state_dict = None
        if i3d_repeat_axis is not None:
            preserve_state_dict = model.state_dict()
        self.model = model
        self.model = self.convert_module(self.model)
        if i3d_repeat_axis is not None:
            self.load_state_dict(preserve_state_dict, strict=True, i3d_repeat_axis=i3d_repeat_axis)
        if self.linear and r_input is not None:
            fix_dim_linear(self.model, r_input)

    def convert_conv_kwargs(self, kwargs):
        kwargs['bias'] = True if kwargs['bias'] is not None else False
        for k in ['kernel_size', 'stride', 'padding', 'dilation']:
            kwargs[k] = _triple_same(kwargs[k])
        return kwargs

    def load_state_dict(self, state_dict, strict=True, i3d_repeat_axis=None):
        if i3d_repeat_axis is not None:
            return load_state_dict_from_2d_to_i3d(self.model, state_dict, strict, repeat_axis=i3d_repeat_axis)
        else:
            return self.model.load_state_dict(state_dict, strict)


def load_state_dict_from_2d_to_i3d(model_3d, state_dict_2d, strict=True, repeat_axis=-1):
    present_dict = model_3d.state_dict()
    for key in list(state_dict_2d.keys()):
        if state_dict_2d[key].dim() == 4:
            repeat_times = present_dict[key].shape[repeat_axis]
            state_dict_2d[key] = torch.stack([state_dict_2d[key]] * repeat_times, dim=repeat_axis) / repeat_times
    return model_3d.load_state_dict(state_dict_2d, strict=strict)


def is_int(val):
    try:
        num = int(val)
    except ValueError:
        return False
    return True


def find_linear(model):
    for module in model.modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                return child_name, child, module


def fix_dim_linear(model, r_input):
    try:
        output = model(r_input)
    except RuntimeError as e:
        exp_dim = [int(v) for v in re.split('\[|\]| ', str(e)) if is_int(v)][1]
        child_name, child, module = find_linear(model)
        arguments = nn.Linear.__init__.__code__.co_varnames[1:]
        kwargs = {k: getattr(child, k) for k in arguments}
        kwargs['in_features'] = exp_dim
        kwargs['bias'] = True if kwargs['bias'] is not None else False
        setattr(module, child_name, nn.Linear(**kwargs))
        with torch.no_grad():
            weights = next(child.parameters())
            new_weights = next(getattr(module, child_name).parameters())
            repeats = np.ceil(new_weights.shape[-1] / weights.shape[-1]).astype(int)
            new_weights.copy_(torch.cat([weights] * repeats, dim=-1)[:, :new_weights.shape[-1]])
