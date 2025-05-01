import math
from torch.optim.lr_scheduler import _LRScheduler

class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizer, warm_start_epochs, cosine_epochs, eta_min=0, last_epoch=-1, max_epoch=None):
        '''
        Args:
            optimizer: (Optimizer) - Wrapped optimizer.
            warm_start_epochs: (int, float) - (fraction of) Number of epochs for warm start.
            cosine_epochs: (int, float) - (fraction of) Number of epochs for cosine annealing.
            eta_min: (float) - Minimum learning rate.
            last_epoch: (int) - The index of last epoch. Default: -1.
            max_epoch : (int) - The number of epochs to train. needed if we have fractional args.            
        '''
        self.warm_start_epochs = warm_start_epochs
        self.cosine_epochs = cosine_epochs
        self.eta_min = eta_min

        if cosine_epochs > 0 and cosine_epochs < 1:
            assert max_epoch is not None, "max_epoch must be provided if cosine_epochs is fractional"
            self.cosine_epochs = int(cosine_epochs * max_epoch)
        
        if warm_start_epochs > 0 and warm_start_epochs < 1:
            assert max_epoch is not None, "max_epoch must be provided if warm_start_epochs is fractional"
            self.warm_start_epochs = int(warm_start_epochs * max_epoch)


        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warm_start_epochs:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi * self.last_epoch / self.warm_start_epochs)) / 2
                    for base_lr in self.base_lrs]
        elif self.last_epoch < self.warm_start_epochs + self.cosine_epochs:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * (self.last_epoch - self.warm_start_epochs) / self.cosine_epochs)) / 2
                    for base_lr in self.base_lrs]
        else:
            return [self.eta_min for _ in self.base_lrs]