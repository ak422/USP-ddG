import os
import time
import random
import logging
from typing import OrderedDict
import torch
import torch.linalg
import numpy as np
import yaml
import json
import re
from easydict import EasyDict
from glob import glob
import shutil


class BlackHole(object):
    def __setattr__(self, name, value):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return self


class Counter(object):
    def __init__(self, start=0):
        super().__init__()
        self.now = start

    def step(self, delta=1):
        prev = self.now
        self.now += delta
        return prev


def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    formatter = logging.Formatter('[%(asctime)s] %(message)s',"%m-%d %H:%M:%S")
    # formatter = logging.Formatter('%(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)

    if log_dir is not None:
        date_time = re.search(r'\d{2}_\d{2}_\d{2}_\d{2}_\d{2}', log_dir)
        file_handler = logging.FileHandler(os.path.join(log_dir, '%s.txt' % date_time.group()))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_new_dir(root='./logs', prefix='', tag=''):
    fn = time.strftime('%m_%d_%H_%M_%S', time.localtime())
    if prefix != '':
        fn = prefix + '_' + fn
    if tag != '':
        fn = fn + '_' + tag
    log_dir = os.path.join(root, fn)
    os.makedirs(log_dir)
    return log_dir


def seed_all(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def should_ignore_dir(dir_path, allowed_extensions=('.py', '.yml')):
    """检查目录是否包含目标扩展名的文件（递归）"""
    for root, _, files in os.walk(dir_path):
        if any(f.endswith(allowed_extensions) for f in files):
            return False  # 目录包含目标文件，不忽略
    return True  # 目录不包含目标文件，忽略


def ignore_non_py_and_yml(dirname, filenames):
    allowed_extensions = ('.py', '.yml')
    ignored = []

    for f in filenames:
        full_path = os.path.join(dirname, f)
        if os.path.isfile(full_path):
            # 如果不是 .py 或 .yml，加入忽略列表
            if not f.endswith(allowed_extensions):
                ignored.append(f)
        elif os.path.isdir(full_path):
            # 如果目录不包含目标文件，递归检查并忽略
            if should_ignore_dir(full_path, allowed_extensions):
                ignored.append(f)

    return ignored

def save_code(save_dir):
    save_code_dir = os.path.join(save_dir, 'CATH_code/')

    shutil.copytree('./configs', save_code_dir + 'configs', ignore=ignore_non_py_and_yml)
    shutil.copytree('./src', save_code_dir + 'src', ignore=ignore_non_py_and_yml)
    # shutil.copyfile('./data/SKEMPI2/skempi_v2_cache/entries_cache/entries.pkl', save_code_dir + 'entries.pkl')
    shutil.copyfile('./data/SKEMPI2/skempi_v2_cache/skempi_v2.csv', save_code_dir + 'skempi_v2.csv')
    for file in os.listdir("./"):
        if '.py' in file:
            shutil.copyfile('./' + file, save_code_dir + file)


def inf_iterator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def log_hyperparams(writer, args):
    from torch.utils.tensorboard.summary import hparams
    vars_args = {k: v if isinstance(v, str) else repr(v) for k, v in vars(args).items()}
    exp, ssi, sei = hparams(vars_args, {})
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)


def int_tuple(argstr):
    return tuple(map(int, argstr.split(',')))


def str_tuple(argstr):
    return tuple(argstr.split(','))


def get_checkpoint_path(folder, it=None):
    if it is not None:
        return os.path.join(folder, '%d.pt' % it), it
    all_iters = list(map(lambda x: int(os.path.basename(x[:-3])), glob(os.path.join(folder, '*.pt'))))
    all_iters.sort()
    return os.path.join(folder, '%d.pt' % all_iters[-1]), all_iters[-1]


class Loader(yaml.SafeLoader):
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream):
        """Initialise Loader."""

        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)


def construct_include(loader: Loader, node: yaml.Node):
    """Include file referenced at node."""

    filename = os.path.abspath(os.path.join(loader._root, loader.construct_scalar(node)))
    extension = os.path.splitext(filename)[1].lstrip('.')

    with open(filename, 'r') as f:
        if extension in ('yaml', 'yml'):
            return yaml.load(f, Loader)
        elif extension in ('json', ):
            return json.load(f)
        else:
            return ''.join(f.readlines())


yaml.add_constructor('!include', construct_include, Loader)


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.load(f, Loader))
        # config = OmegaConf.create(yaml.load(f, Loader))
    config_name = os.path.basename(config_path)[:os.path.basename(config_path).rfind('.')]
    return config, config_name


def extract_weights(weights: OrderedDict, prefix):
    extracted = OrderedDict()
    for k, v in weights.items():
        if k.startswith(prefix):
            extracted.update({
                k[len(prefix):]: v
            })
    return extracted


def current_milli_time():
    return round(time.time() * 1000)

