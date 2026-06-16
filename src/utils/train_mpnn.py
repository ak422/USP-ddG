import numpy as np
import torch
import os
import math
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau, MultiStepLR, ExponentialLR

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
        return {
            'step': self._step,
            'rate': self._rate,
            'optimizer': self.optimizer.state_dict(),
        }
    def load_state_dict(self):
        self._step = state_dict['step']
        self._rate = state_dict['rate']
        self.optimizer.load_state_dict(state_dict['optimizer'])

def get_std_opt(params, d_model, warmup, step, factor, weight_decay):
    # d_model：模型特征维度，gradual warmup学习率自适应优化器
    return NoamOpt(
        d_model, factor, warmup, torch.optim.Adam(params, lr=0, weight_decay= weight_decay, betas=(0.9, 0.98), eps=1e-9), step
    )

def get_optimizer(cfg, model):
    """创建带分组学习率的优化器"""
    # 1. 定义模块→超参映射
    module_hyperparams = {
        'esm_embed': {'lr': cfg.lr_2, 'weight_decay': cfg.weight_decay_2},
        'foldx_ddg': {'lr': cfg.lr_3, 'weight_decay': cfg.weight_decay_3},
    }

    # 2. 初始化参数集合
    grouped_params = {k: [] for k in module_hyperparams}  # 使用list而不是set
    other_params = []

    # 3. 遍历所有参数并分类
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # 跳过不需要梯度的参数

        matched = False

        # 3.1 匹配esm_embed
        if '.esm_embed.' in name or name.startswith('esm_embed.'):
            grouped_params['esm_embed'].append(param)
            matched = True

        # 3.2 匹配foldx_ddg
        if not matched and ('.foldx_ddg.' in name or name.startswith('foldx_ddg.')):
            grouped_params['foldx_ddg'].append(param)
            matched = True

        # 3.3 剩余参数
        if not matched:
            other_params.append(param)

    # 4. 构造参数组（注意顺序！）
    params_to_update = []

    # 重要：记录参数组对应的模块名称，用于调度器
    param_group_names = []
    # 4.1 添加esm_embed组
    if grouped_params['esm_embed']:
        params_to_update.append({
            'params': grouped_params['esm_embed'],
            'lr': module_hyperparams['esm_embed']['lr'],
            'weight_decay': module_hyperparams['esm_embed']['weight_decay'],
        })
        param_group_names.append('esm_embed')

    # 4.2 添加foldx_ddg组
    if grouped_params['foldx_ddg']:
        params_to_update.append({
            'params': grouped_params['foldx_ddg'],
            'lr': module_hyperparams['foldx_ddg']['lr'],
            'weight_decay': module_hyperparams['foldx_ddg']['weight_decay'],
        })
        param_group_names.append('foldx_ddg')

    # 4.3 添加其他参数组（主模型）
    if other_params:
        params_to_update.append({
            'params': other_params,
            'lr': cfg.lr,
            'weight_decay': cfg.weight_decay,
        })
        param_group_names.append('main')

    # 5. 创建优化器
    if cfg.type == 'adam':
        optimizer = torch.optim.Adam(
            params=params_to_update,
            betas=(cfg.beta1, cfg.beta2),
            eps=float(cfg.get('eps', 1e-8)),
        )
        # 存储参数组名称供调度器使用
        optimizer.param_group_names = param_group_names
        return optimizer
    elif cfg.type == 'adamw':
        optimizer = torch.optim.AdamW(
            params=params_to_update,  # 修正：使用分组参数
            betas=(cfg.beta1, cfg.beta2),
            eps=float(cfg.get('eps', 1e-8)),
        )
        optimizer.param_group_names = param_group_names
        return optimizer
    elif cfg.type == 'warm_up':
        optimizer = get_std_opt(
            # params=model.parameters(),
            params=params_to_update,
            d_model=cfg.d_model,
            warmup=cfg.warmup,
            step=cfg.step,
            factor=cfg.factor,
            weight_decay=cfg.weight_decay
        )
        optimizer.param_group_names = param_group_names
        return optimizer

    else:
        raise NotImplementedError(f'Optimizer not supported: {cfg.type}')

def warmup_CosineStable(
        warm_up_iters,
        T_0,
        param_group_names,

        lr_max, lr_stable,
        lr_2_max, lr_2_stable,
        lr_3_max, lr_3_stable,
):
    """
    Warmup + Cosine Decay + Stable LR

    特点:
    1. 前期线性 warmup
    2. 中期按周期 T_0 进行 cosine 平滑下降
    3. 后期稳定在 lr_stable
    4. 无 restart，更适合 protein representation learning

    Args:
        warm_up_iters: warmup步数
        T_0: cosine 衰减周期（步数）

        param_group_names:
            ['main', 'esm_embed', 'foldx_ddg']

        lr_max/lr_stable:
            不同参数组的最大学习率和稳定学习率
    """

    def create_lambda(lr_max, lr_stable):
        # 稳定学习率比例
        stable_ratio = lr_stable / lr_max

        def _lambda(iter):
            # =========================
            # 1. Warmup
            # =========================
            if iter < warm_up_iters:
                return float(iter) / float(max(1, warm_up_iters))

            # =========================
            # 2. Cosine Decay over T_0
            # =========================
            progress = (
                (iter - warm_up_iters)
                / float(max(1, T_0))
            )
            # 防止超过1
            progress = min(progress, 1.0)

            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

            # =========================
            # 3. Stable LR
            # =========================
            multiplier = (
                stable_ratio
                + (1.0 - stable_ratio) * cosine_decay
            )

            return multiplier

        return _lambda

    # =========================
    # 根据参数组创建 scheduler
    # =========================
    lr_lambdas = []
    for name in param_group_names:
        if name == 'main':
            lr_lambdas.append(
                create_lambda(
                    lr_max,
                    lr_stable
                )
            )
        elif name == 'esm_embed':
            lr_lambdas.append(
                create_lambda(
                    lr_2_max,
                    lr_2_stable
                )
            )
        elif name == 'foldx_ddg':
            lr_lambdas.append(
                create_lambda(
                    lr_3_max,
                    lr_3_stable
                )
            )
        else:
            raise ValueError(
                f"Unknown param group name: {name}"
            )
    return lr_lambdas


def get_scheduler(cfg, optimizer):
    """创建学习率调度器"""
    if cfg.type is None or cfg.type == 'none':
        return BlackHole()

    elif cfg.type == 'plateau':
        return ReduceLROnPlateau(
            optimizer,
            factor=cfg.factor,
            patience=cfg.patience,
            min_lr=cfg.min_lr,
        )

    elif cfg.type == 'multistep':
        return MultiStepLR(
            optimizer,
            milestones=cfg.milestones,
            gamma=cfg.gamma,
        )

    elif cfg.type == 'exp':
        return ExponentialLR(
            optimizer,
            gamma=cfg.gamma,
        )

    elif cfg.type == 'lambdaLR':
        # 检查优化器类型
        if not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
            print(f"Warning: lambdaLR scheduler with {type(optimizer)} may not work properly")

        # 获取参数组名称（在get_optimizer中设置）
        if not hasattr(optimizer, 'param_group_names'):
            raise AttributeError(
                "Optimizer does not have 'param_group_names' attribute. "
                "Make sure you're using the fixed get_optimizer function."
            )

        # 创建lambda函数列表
        lr_lambdas = warmup_CosineStable(
            warm_up_iters=cfg.warm_up_iters,
            T_0=cfg.T_0,
            param_group_names=optimizer.param_group_names,
            lr_max=cfg.lr_max,
            lr_stable=cfg.lr_stable,
            lr_2_max=cfg.lr_2_max,
            lr_2_stable=cfg.lr_2_stable,
            lr_3_max=cfg.lr_3_max,
            lr_3_stable=cfg.lr_3_stable,
        )

        # 验证数量匹配
        if len(lr_lambdas) != len(optimizer.param_groups):
            raise ValueError(
                f"Number of lambdas ({len(lr_lambdas)}) != "
                f"number of param groups ({len(optimizer.param_groups)})"
            )

        return LambdaLR(
            optimizer,
            lr_lambda=lr_lambdas,
            last_epoch=cfg.get('last_epoch', -1),  # -1表示从当前状态开始
        )

    else:
        raise NotImplementedError(f'Scheduler not supported: {cfg.type}')


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


