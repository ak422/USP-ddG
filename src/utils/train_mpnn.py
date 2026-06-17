import numpy as np
import torch
import os
import math
import matplotlib.pyplot as plt

from src.utils.protein.constants import chi_pi_periodic, AA
from src.utils.misc import BlackHole
from src.utils.early_stopping import EarlyStopping
from transformers import EsmModel

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer, step):
        self.optimizer = optimizer
        # for i, p in enumerate(self.optimizer.param_groups):  # zero grad of esm2
        #     if i < 3:
        #         for param in p['params']:
        #             param.requires_grad = False
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for i,p in enumerate(self.optimizer.param_groups):
            if i < 3:
                # for param in p['params']:
                #     param.requires_grad = True
                p['lr'] = rate * 1.0e-2
            else:
                p['lr'] = rate      # 设置全局lr
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) *min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()
    def load_state_dict(self):
        return self.optimizer.load_state_dict()

def get_std_opt(params, d_model, warmup, step, factor, weight_decay):
    # d_model：模型特征维度，gradual warmup学习率自适应优化器
    return NoamOpt(
        d_model, factor, warmup, torch.optim.Adam(params, lr=0, weight_decay= weight_decay, betas=(0.9, 0.98), eps=1e-9), step
    )

def get_optimizer(cfg, model):
    # 1. 定义模块→超参映射（严格对应你的需求）
    module_hyperparams = {
        'esm_embed': {'lr': cfg.lr_2, 'initial_lr': cfg.lr_2, 'weight_decay': cfg.weight_decay_2},
        'foldx_ddg': {'lr': cfg.lr_3, 'initial_lr': cfg.lr_3, 'weight_decay': cfg.weight_decay_3},
        # 'MoE_blocks': {'lr': cfg.lr_4, 'initial_lr': cfg.lr_4, 'weight_decay': cfg.weight_decay_4},
    }
    # 2. 初始化参数集合
    grouped_params = {k: set() for k in module_hyperparams}  # 关键字组
    other_params = set()    # 其他组

    # 3. 遍历所有参数并分类
    for name, param in model.named_parameters():
        matched = False
        # 3.1 匹配类属性子模块（esm_embed / adaptmlp_list）
        # for module_name in ['esm_embed', 'MoE_blocks']:
        for module_name in ['esm_embed']:
            if f'.{module_name}.' in name or name.startswith(f'{module_name}.'):
                grouped_params[module_name].add(param)
                matched = True
                break
        if matched:
            continue

        # 3.2 匹配 Sequential 模块（foldx_ddg）
        if f'.foldx_ddg.' in name or name.startswith('foldx_ddg.'):
            grouped_params['foldx_ddg'].add(param)
            matched = True
            continue

        # 3.3 剩余参数归入其他组
        other_params.add(param)

    # 4. 构造参数组（带独立超参）
    params_to_update = []

    # 4.1 添加关键字组（带各自超参）
    for module_name, hp in module_hyperparams.items():
        if grouped_params[module_name]:
            params_to_update.append({
                'params': list(grouped_params[module_name]),
                **hp,
            })

    # 4.2 添加其他组（默认超参）
    if other_params:
        params_to_update.append({
            'params': list(other_params),
            'lr': cfg.lr,
            'initial_lr': cfg.lr,
            'weight_decay': cfg.weight_decay,
        })

    if cfg.type == 'adam':
        optimizer = torch.optim.Adam(
            # model.parameters(),
            params=params_to_update,
            lr=cfg.lr,      # global lr
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2, )
        )
        return optimizer
    elif cfg.type == 'adamw':
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2, )
        )
        return optimizer
    elif cfg.type == 'warm_up':
        optimizer = get_std_opt(
            # params=model.parameters(),
            params=params_to_update,
            d_model=cfg.d_model,
            warmup=cfg.warmup,
            step=cfg.step,
            factor=cfg.factor,
            weight_decay=cfg.weight_decay)
        return optimizer
    else:
        raise NotImplementedError('Optimizer not supported: %s' % cfg.type)

def warmup_CosineAnneal(warm_up_iters, T_iters, lr_max, lr_min, lr_2_max, lr_2_min, lr_3_max, lr_3_min):
    # 自定义warm up AND Cosine Anneal学习率scheduler
    #  param_groups[0] to param_groups[2]
    _lambda = lambda iter: iter / warm_up_iters if iter < warm_up_iters else \
        (lr_min + 0.5 * (lr_max - lr_min) *(1.0 + math.cos((iter - warm_up_iters) /
             (T_iters - warm_up_iters) * math.pi))) / lr_max
    _lambda2 = lambda iter: iter / warm_up_iters if iter < warm_up_iters else \
        (lr_2_min + 0.5 * (lr_2_max - lr_2_min) *(1.0 + math.cos((iter - warm_up_iters) /
             (T_iters - warm_up_iters) * math.pi))) / lr_2_max
    _lambda3 = lambda iter: iter / warm_up_iters if iter < warm_up_iters else \
        (lr_3_min + 0.5 * (lr_3_max - lr_3_min) *(1.0 + math.cos((iter - warm_up_iters) /
             (T_iters - warm_up_iters) * math.pi))) / lr_3_max
    return [_lambda2, _lambda3, _lambda]

def get_scheduler(cfg, optimizer):
    if cfg.type is None:
        return BlackHole()
    elif cfg.type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=cfg.factor,
            patience=cfg.patience,
            min_lr=cfg.min_lr,
        )
    elif cfg.type == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.milestones,
            gamma=cfg.gamma,
        )
    elif cfg.type == 'exp':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=cfg.gamma,
        )
    elif cfg.type == 'lambdaLR' and isinstance(optimizer, torch.optim.Adam):
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=warmup_CosineAnneal(
                warm_up_iters=cfg.warm_up_iters,
                T_iters=cfg.T_iters,
                lr_max=cfg.lr_max,
                lr_min=cfg.lr_min,
                lr_2_max=cfg.lr_2_max,
                lr_2_min=cfg.lr_2_min,
                lr_3_max=cfg.lr_3_max,
                lr_3_min=cfg.lr_3_min,
            ),
            last_epoch=0,
        )
    elif cfg.type is None:
        return BlackHole()
    # else:
    #     raise NotImplementedError('Scheduler not supported: %s' % cfg.type)


def log_losses(loss, loss_dict, scalar_dict, it, tag, logger=BlackHole(), writer=BlackHole()):
    logstr = '[%s] Iter %05d' % (tag, it)
    logstr += ' | loss %.4f' % loss.item()

    for k, v in loss_dict.items():
        logstr += ' | loss(%s) %.4f' % (k, v.item())
    for k, v in scalar_dict.items():
        logstr += ' | %s %.4f' % (k, v.item() if isinstance(v, torch.Tensor) else v)
    logger.info(logstr)

    writer.add_scalar('%s/loss' % tag, loss, it)

    for k, v in loss_dict.items():
        writer.add_scalar('%s/loss_%s' % (tag, k), v, it)

    for k, v in scalar_dict.items():
        writer.add_scalar('%s/%s' % (tag, k), v, it)
    writer.flush()


class ScalarMetricAccumulator(object):

    def __init__(self):
        super().__init__()
        self.accum_dict = {}
        self.count_dict = {}

    @torch.no_grad()
    def add(self, name, value, batchsize=None, mode=None):
        assert mode is None or mode in ('mean', 'sum')

        if mode is None:
            delta = value.sum()
            count = value.size(0)
        elif mode == 'mean':
            delta = value * batchsize
            count = batchsize
        elif mode == 'sum':
            delta = value
            count = batchsize
        delta = delta.item() if isinstance(delta, torch.Tensor) else delta

        if name not in self.accum_dict:
            self.accum_dict[name] = 0
            self.count_dict[name] = 0
        self.accum_dict[name] += delta
        self.count_dict[name] += count

    def log(self, epoch, tag, logger=BlackHole(), writer=BlackHole()):
        summary = {k: self.accum_dict[k] / self.count_dict[k] for k in self.accum_dict}
        logstr = '[%s] Epoch %05d' % (tag, epoch)
        for k, v in summary.items():
            logstr += ' | %s %.4f' % (k, v)
            writer.add_scalar('%s/%s' % (tag, k), v, epoch)
        logger.info(logstr)

    def get_average(self, name):
        return self.accum_dict[name] / self.count_dict[name]


def recursive_to(obj, device):
    if isinstance(obj, torch.Tensor):
        try:
            return obj.cuda(device=device, non_blocking=True)
        except RuntimeError:
            return obj.to(device)
    elif isinstance(obj, list):
        return [recursive_to(o, device=device) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to(o, device=device) for o in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to(v, device=device) for k, v in obj.items()}

    else:
        return obj


def sum_weighted_losses(losses, weights):
    """
    Args:
        losses:     Dict of scalar tensors.
        weights:    Dict of weights.
    """
    loss = 0
    for k in losses.keys():
        if weights is None:
            loss = loss + losses[k]
        else:
            loss = loss + weights[k] * losses[k]
    return loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def aggregate_sidechain_accuracy(aa, chi_pred, chi_native, chi_mask):
    aa = aa.reshape(-1)
    chi_mask = chi_mask.reshape(-1, 4)
    diff = torch.min(
        (chi_pred - chi_native) % (2 * np.pi),
        (chi_native - chi_pred) % (2 * np.pi),
    )   # (N, L, 4)
    diff = torch.rad2deg(diff)
    diff = diff.reshape(-1, 4)

    diff_flip = torch.min(
        ( (chi_pred + np.pi) - chi_native) % (2 * np.pi),
        (chi_native - (chi_pred + np.pi) ) % (2 * np.pi),
    )
    diff_flip = torch.rad2deg(diff_flip)
    diff_flip = diff_flip.reshape(-1, 4)
    
    acc = [{j:[] for j in range(1, 4+1)} for i in range(20)]
    for i in range(aa.size(0)):
        for j in range(4):
            chi_number = j+1
            if not chi_mask[i, j].item(): continue
            if chi_pi_periodic[AA(aa[i].item())][chi_number-1]:
                diff_this = min(diff[i, j].item(), diff_flip[i, j].item())
            else:
                diff_this = diff[i, j].item()
            acc[aa[i].item()][chi_number].append(diff_this)
    
    table = np.full((20, 4), np.nan)
    for i in range(20):
        for j in range(1, 4+1):
            if len(acc[i][j]) > 0:
                table[i, j-1] = np.mean(acc[i][j])
    return table


def make_sidechain_accuracy_table_image(tag: str, diff: np.ndarray):
    from Bio.PDB.Polypeptide import index_to_three
    columns = ['chi1', 'chi2', 'chi3', 'chi4']
    rows = [index_to_three(i) for i in range(20)]
    cell_text = diff.tolist()
    fig, ax = plt.subplots(dpi=200)
    ax.axis('tight')
    ax.axis('off')
    ax.set_title(tag)
    ax.table(
        cellText=cell_text,
        colLabels=columns,
        rowLabels=rows,
        loc='center'
    )
    return fig


def load_model_from_checkpoint(ckpt_path, return_ckpt=False):
    from src.models import get_model
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model = get_model(ckpt['config'].model)
    model.load_state_dict(ckpt['model'])
    if return_ckpt:
        return model, ckpt
    else:
        return model
    

class CrossValidation(object):
    def __init__(self, model_factory, config, num_cvfolds, early_stoppingdir='./early_stopping', logger=None):
        super().__init__()
        self.num_cvfolds = num_cvfolds
        self.config = config
        self.early_stoppingdir = early_stoppingdir

        self.models = [
            model_factory(config.model)
            for _ in range(num_cvfolds)
        ]

        self.optimizers = []
        self.schedulers = []
        self.early_stoppings = []
        for model in self.models:
            optimizer = get_optimizer(config.train.optimizer, model)
            scheduler = get_scheduler(config.train.scheduler, optimizer)
            early_stopping = EarlyStopping(early_stoppingdir, logger, patience=5, verbose=False)
            self.optimizers.append(optimizer)
            self.schedulers.append(scheduler)
            self.early_stoppings.append(early_stopping)

    def get(self, fold):
        return self.models[fold], self.optimizers[fold], self.schedulers[fold], self.early_stoppings[fold]

    def to(self, device):
        for m in self.models:
            m.to(device)
        return self

    def state_dict(self):
        return {
            'models': [m.state_dict() for m in self.models],
            # 'optimizers': [o.state_dict() for o in self.optimizers],
            # 'schedulers': [s.state_dict() for s in self.schedulers],
        }

    def save_state_dict(self, args, config, early_stoppingdir, checkpoint):
        models = []
        # optimizers = []
        for fold in range(self.num_cvfolds):
            early_stopping_path = os.path.join(early_stoppingdir, f'Fold_{fold+1}_best_network.pt')
            model = torch.load(early_stopping_path, map_location=args.device)
            models.append(model)

        ckpt_path = os.path.join(early_stoppingdir, checkpoint)
        torch.save({
            'config': self.config,
            'model': {
                        'models': [m for m in models]
                    }
        }, ckpt_path)

    def load_state_dict(self, state_dict):
        for sd, obj in zip(state_dict['models'], self.models):
            obj.load_state_dict(sd)


