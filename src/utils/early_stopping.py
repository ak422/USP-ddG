import numpy as np
import torch
import os

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, logger, patience=7, verbose=False, delta=0.1):
        """
        Args:
            save_path : save path
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.logger = logger
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = np.inf
        self.delta = delta

    def __call__(self, val_score, model, fold):

        score = -val_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model, fold)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.logger.info(f'EarlyStopping counter: [Fold-{fold+1}] {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # self.logger.info(f'EarlyStopping Reset of [Fold-{fold+1}]')
            self.best_score = score
            self.save_checkpoint(val_score, model, fold)
            self.counter = 0

    def save_checkpoint(self, val_score, model, fold):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation score decreased ({self.val_score_min:.6f} --> {val_score:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, f'Fold_{fold+1}_best_network.pt')
        torch.save(model.state_dict(), path)
        self.val_score_min = val_score