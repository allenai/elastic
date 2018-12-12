from __future__ import print_function
import math
import random
import torchvision.transforms as transforms
import warnings
from torch.nn import functional as F
import shutil
import torch.utils.data as data
import os
from PIL import Image
import torch
import numpy as np


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    def __init__(self):
        self.val = None
        self.sum = None
        self.cnt = None
        self.avg = None
        self.ema = None
        self.initialized = False

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def initialize(self, val, n):
        self.val = val
        self.sum = val * n
        self.cnt = n
        self.avg = val
        self.ema = val
        self.initialized = True

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        self.ema = self.ema * 0.99 + self.val * 0.01


def inter_and_union(pred, mask, num_class):
    pred = np.asarray(pred, dtype=np.uint8).copy()
    mask = np.asarray(mask, dtype=np.uint8).copy()

    # 255 -> 0
    pred += 1
    mask += 1
    pred = pred * (mask > 0)

    inter = pred * (pred == mask)
    (area_inter, _) = np.histogram(inter, bins=num_class, range=(1, num_class))
    (area_pred, _) = np.histogram(pred, bins=num_class, range=(1, num_class))
    (area_mask, _) = np.histogram(mask, bins=num_class, range=(1, num_class))
    area_union = area_pred + area_mask - area_inter

    return (area_inter, area_union)


def preprocess(image, mask, flip=False, scale=None, crop=None):
    if flip:
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    if scale:
        w, h = image.size
        rand_log_scale = math.log(scale[0], 2) + random.random() * (math.log(scale[1], 2) - math.log(scale[0], 2))
        random_scale = math.pow(2, rand_log_scale)
        new_size = (int(round(w * random_scale)), int(round(h * random_scale)))
        image = image.resize(new_size, Image.ANTIALIAS)
        mask = mask.resize(new_size, Image.NEAREST)

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = data_transforms(image)
    mask = torch.LongTensor(np.array(mask).astype(np.int64))

    if crop:
        h, w = image.shape[1], image.shape[2]
        ori_h, ori_w = image.shape[1], image.shape[2]

        pad_tb = max(0, int((1 + crop[0] - h) / 2))
        pad_lr = max(0, int((1 + crop[1] - w) / 2))
        image = torch.nn.ZeroPad2d((pad_lr, pad_lr, pad_tb, pad_tb))(image)
        mask = torch.nn.ConstantPad2d((pad_lr, pad_lr, pad_tb, pad_tb), 255)(mask)

        h, w = image.shape[1], image.shape[2]
        i = random.randint(0, h - crop[0])
        j = random.randint(0, w - crop[1])
        image = image[:, i:i + crop[0], j:j + crop[1]]
        mask = mask[i:i + crop[0], j:j + crop[1]]

    return image, mask, pad_tb - j, pad_lr - i, ori_h, ori_w


# pascal dataloader
class VOCSegmentation(data.Dataset):
    CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
        'tv/monitor'
    ]

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, crop_size=None):
        self.root = root
        _voc_root = os.path.join(self.root, 'VOC2012')
        _list_dir = os.path.join(_voc_root, 'list')
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.crop_size = crop_size

        if download:
            self.download()

        if self.train:
            _list_f = os.path.join(_list_dir, 'train_aug.txt')
        else:
            _list_f = os.path.join(_list_dir, 'val.txt')
        self.images = []
        self.masks = []
        with open(_list_f, 'r') as lines:
            for line in lines:
                _image = _voc_root + line.split()[0]
                _mask = _voc_root + line.split()[1]
                assert os.path.isfile(_image)
                assert os.path.isfile(_mask)
                self.images.append(_image)
                self.masks.append(_mask)

    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.masks[index])

        _img, _target, a, b, h, w = preprocess(_img, _target,
                                               flip=True if self.train else False,
                                               scale=(0.5, 2.0) if self.train else None,
                                               crop=(self.crop_size, self.crop_size))

        if self.transform is not None:
            _img = self.transform(_img)

        if self.target_transform is not None:
            _target = self.target_transform(_target)

        return _img, _target, a, b, h, w  # used for visualizing

    def __len__(self):
        return len(self.images)

    def download(self):
        raise NotImplementedError('Automatic download not yet implemented.')


# flops counter
def add_flops_counting_methods(net_main_module):
    """Adds flops counting functions to an existing model. After that
        the flops count should be activated and the model should be run on an input
        image.

        Example:

        fcn = add_flops_counting_methods(fcn)
        fcn = fcn.cuda().train()
        fcn.start_flops_count()


        _ = fcn(batch)

        fcn.compute_average_flops_cost() / 1e9 / 2 # Result in GFLOPs per image in batch

        Important: dividing by 2 only works for resnet models -- see below for the details
        of flops computation.

        Attention: we are counting multiply-add as two flops in this work, because in
        most resnet models convolutions are bias-free (BN layers act as bias there)
        and it makes sense to count muliply and add as separate flops therefore.
        This is why in the above example we divide by 2 in order to be consistent with
        most modern benchmarks. For example in "Spatially Adaptive Computatin Time for Residual
        Networks" by Figurnov et al multiply-add was counted as two flops.

        This module computes the average flops which is necessary for dynamic networks which
        have different number of executed layers. For static networks it is enough to run the network
        once and get statistics (above example).

        Implementation:
        The module works by adding batch_count to the main module which tracks the sum
        of all batch sizes that were run through the network.

        Also each convolutional layer of the network tracks the overall number of flops
        performed.

        The parameters are updated with the help of registered hook-functions which
        are being called each time the respective layer is executed.

        Parameters
        ----------
        net_main_module : torch.nn.Module
            Main module containing network

        Returns
        -------
        net_main_module : torch.nn.Module
            Updated main module with new methods/attributes that are used
            to compute flops.
        """

    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    net_main_module.start_flops_count = start_flops_count.__get__(net_main_module)
    net_main_module.stop_flops_count = stop_flops_count.__get__(net_main_module)
    net_main_module.reset_flops_count = reset_flops_count.__get__(net_main_module)
    net_main_module.compute_average_flops_cost = compute_average_flops_cost.__get__(net_main_module)

    net_main_module.reset_flops_count()

    # Adding variables necessary for masked flops computation
    net_main_module.apply(add_flops_mask_variable_or_reset)

    return net_main_module


def compute_average_flops_cost(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Returns current mean flops consumption per image.

    """

    batches_count = self.__batch_counter__
    flops_sum = 0
    for module in self.modules():
        if hasattr(module, '__flops__'):  # is_supported_instance(module)
            flops_sum += module.__flops__

    return flops_sum / batches_count


def start_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Activates the computation of mean flops consumption per image.
    Call it before you run the network.

    """
    add_batch_counter_hook_function(self)
    self.apply(add_flops_counter_hook_function)


def stop_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Stops computing the mean flops consumption per image.
    Call whenever you want to pause the computation.

    """
    remove_batch_counter_hook_function(self)
    self.apply(remove_flops_counter_hook_function)


def reset_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Resets statistics computed so far.

    """
    add_batch_counter_variables_or_reset(self)
    self.apply(add_flops_counter_variable_or_reset)


def add_flops_mask(module, mask):
    def add_flops_mask_func(module):
        if isinstance(module, torch.nn.Conv2d):
            module.__mask__ = mask
    module.apply(add_flops_mask_func)


def remove_flops_mask(module):
    module.apply(add_flops_mask_variable_or_reset)


# ---- Internal functions
def is_supported_instance(module):
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.ReLU) \
       or isinstance(module, torch.nn.PReLU) or isinstance(module, torch.nn.ELU) \
       or isinstance(module, torch.nn.LeakyReLU) or isinstance(module, torch.nn.ReLU6) \
       or isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.MaxPool2d) \
       or isinstance(module, torch.nn.AvgPool2d) or isinstance(module, torch.nn.BatchNorm2d):
        return True

    return False


def empty_flops_counter_hook(module, input, output):
    module.__flops__ += 0


def relu_flops_counter_hook(module, input, output):
    input = input[0]
    batch_size = input.shape[0]
    active_elements_count = batch_size
    for val in input.shape[1:]:
        active_elements_count *= val

    module.__flops__ += active_elements_count


def linear_flops_counter_hook(module, input, output):
    input = input[0]
    batch_size = input.shape[0]
    module.__flops__ += batch_size * input.shape[1] * output.shape[1]


def pool_flops_counter_hook(module, input, output):
    input = input[0]
    module.__flops__ += np.prod(input.shape)

def bn_flops_counter_hook(module, input, output):
    module.affine
    input = input[0]

    batch_flops = np.prod(input.shape)
    if module.affine:
        batch_flops *= 2
    module.__flops__ += batch_flops

def conv_flops_counter_hook(conv_module, input, output):
    # Can have multiple inputs, getting the first one
    input = input[0]

    batch_size = input.shape[0]
    output_height, output_width = output.shape[2:]

    kernel_height, kernel_width = conv_module.kernel_size
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = kernel_height * kernel_width * in_channels * filters_per_channel

    active_elements_count = batch_size * output_height * output_width

    if conv_module.__mask__ is not None:
        # (b, 1, h, w)
        flops_mask = conv_module.__mask__.expand(batch_size, 1, output_height, output_width)
        active_elements_count = flops_mask.sum()

    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0

    if conv_module.bias is not None:

        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops

    conv_module.__flops__ += overall_flops


def batch_counter_hook(module, input, output):
    # Can have multiple inputs, getting the first one
    input = input[0]
    batch_size = input.shape[0]
    module.__batch_counter__ += batch_size


def add_batch_counter_variables_or_reset(module):

    module.__batch_counter__ = 0


def add_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        return

    handle = module.register_forward_hook(batch_counter_hook)
    module.__batch_counter_handle__ = handle


def remove_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        module.__batch_counter_handle__.remove()
        del module.__batch_counter_handle__


def add_flops_counter_variable_or_reset(module):
    if is_supported_instance(module):
        module.__flops__ = 0


def add_flops_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops_handle__'):
            return

        if isinstance(module, torch.nn.Conv2d):
            handle = module.register_forward_hook(conv_flops_counter_hook)
        elif isinstance(module, torch.nn.ReLU) or isinstance(module, torch.nn.PReLU) \
             or isinstance(module, torch.nn.ELU) or isinstance(module, torch.nn.LeakyReLU) \
             or isinstance(module, torch.nn.ReLU6):
            handle = module.register_forward_hook(relu_flops_counter_hook)
        elif isinstance(module, torch.nn.Linear):
            handle = module.register_forward_hook(linear_flops_counter_hook)
        elif isinstance(module, torch.nn.AvgPool2d) or isinstance(module, torch.nn.MaxPool2d):
            handle = module.register_forward_hook(pool_flops_counter_hook)
        elif isinstance(module, torch.nn.BatchNorm2d):
            handle = module.register_forward_hook(bn_flops_counter_hook)
        else:
            handle = module.register_forward_hook(empty_flops_counter_hook)
        module.__flops_handle__ = handle


def remove_flops_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops_handle__'):
            module.__flops_handle__.remove()
            del module.__flops_handle__
# --- Masked flops counting


# Also being run in the initialization
def add_flops_mask_variable_or_reset(module):
    if is_supported_instance(module):
        module.__mask__ = None
