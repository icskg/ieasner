
import numpy as np
import torch

from utils.logs import Logger


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0, min_lr = 1e-4):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.min_lr = min_lr
        self.val_loss_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name
        self.loss_descend = False

    def __call__(self, metric_score, model, optimizer, path):

        score = -metric_score if self.verbose else metric_score
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(metric_score, model, path)
            self.loss_descend = True

        elif score < self.best_score + self.delta:
            self.counter += 1
            Logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                if self.get_current_learning_rate(optimizer) <= self.min_lr:
                    self.early_stop = True
                else:
                    self.change_learning_rate(optimizer)
                    self.counter = 0
            self.loss_descend = False
        else:
            self.best_score = score
            self.save_checkpoint(metric_score, model, path)
            self.counter = 0
            self.loss_descend = True

    @staticmethod
    def change_learning_rate(optimizer, decay_factor = 0.1):
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            param_group['lr'] = current_lr * decay_factor
        Logger.info(f"Learning rate changed to {optimizer.param_groups[0]['lr']}")

    @staticmethod
    def get_current_learning_rate(optimizer):
        return optimizer.param_groups[0]['lr']

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            Logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if path is not None:
            torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss