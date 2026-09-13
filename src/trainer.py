import torch


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        config
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.config = config

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)

    def train_one_epoch(self):

        self.model.train()

        total_loss = 0.0

        for inputs, targets in self.train_loader:

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(inputs)

            loss = self.criterion(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1)
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0
            )

            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()

        average_loss = total_loss / len(self.train_loader)

        return average_loss

    @torch.no_grad()
    def validate(self):

        self.model.eval()

        total_loss = 0.0

        for inputs, targets in self.val_loader:

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            logits = self.model(inputs)

            loss = self.criterion(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1)
            )

            total_loss += loss.item()

        average_loss = total_loss / len(self.val_loader)

        return average_loss