#Training, Validation, Checkpoint, Logging, Scheduler, AMP, Gradient Clipping

class trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, criterion, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.config  = config
    def train_one_epoch(self, model):
        model.train()
        for batch in self.train_loader:
            

    