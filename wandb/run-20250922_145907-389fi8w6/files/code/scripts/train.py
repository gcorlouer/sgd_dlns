import torch
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm

from torch import nn
from teacher import *
from models import *
from pathlib import Path
import wandb

wandb.init(
    project="sgd-dln",
    entity="geeom"   # your personal username
)

# Code training loop class with SGD
class Trainer():
    def __init__(self, teacher: Teacher, 
                 dataset: TeacherDataset,
                 model: DLN,
                 lr: float = 0.01,
                 batch_size: int = 1,
                 num_epochs: int = 1000,
                 device: str = 'cpu'):
        
        self.teacher = teacher 
        self.dataset = dataset
        self.model = model.to(device)
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()

        self.train_losses = []
        self.test_losses = []
        self.grad_norms = []

    def online_training_loop(self):
        train, test = self.dataset.train_test_split()
        train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test, batch_size=self.batch_size, shuffle=False)
        for x, y in tqdm(train_loader, desc="Training", unit="batch"):
            x, y = x.to(self.device), y.to(self.device)
            # Forward
            self.model.train()
            target = self.model(x)
            train_loss = self.loss(target, y)
            self.train_losses.append(train_loss.item())
            self.test_losses.append(self.evaluate(test_loader).item())
            # Backward
            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()
            grad_norm = torch.sqrt(sum([p.grad.norm()**2 for p in model.parameters() if p.grad is not None]))
            self.grad_norms.append(grad_norm)
    
    def epoch_training_loop(self, train_loader: DataLoader):
        epoch_loss = 0
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            # Forward
            self.model.train()
            target = self.model(x)
            train_loss = self.loss(target, y)
            # backward
            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()
            epoch_loss += train_loss.item()
        return epoch_loss/len(train_loader)
    

    def training_epochs(self):
        train, test = self.dataset.train_test_split()
        for i in tqdm(range(self.num_epochs), desc="Training", unit="epoch"):
            train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test, batch_size=self.batch_size, shuffle=False)
            train_loss = self.epoch_training_loop(train_loader)
            self.train_losses.append(train_loss)
            self.test_losses.append(self.evaluate(test_loader).item())
            wandb.log({
                        "train_loss": train_loss,
                        "test_loss": self.evaluate(test_loader).item(),
                        "epoch": i
                    })

    def evaluate(self, test_loader: DataLoader):
        test_loss = 0
        self.model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                # Forward
                target = self.model(x)
                test_loss += self.loss(target, y)
        return test_loss/len(test_loader)
# Get interesting observables: modes, loss, end of training distribution


class Observable():
    def __init__(self, teacher_matrix: Teacher, model: DLN):
        self.teacher_matrix = teacher_matrix
        self.model = model

    def mode_matrix(self):
        U, _, V = self.teacher_matrix.components
        params = model.parameters()
        product = 1
        product *= [w for w in params]
        return U.T @ product @ V
    
# Plotting functions for losses, modes etc


if __name__ == '__main__':
    output_dim = 10
    hidden_dim = 100
    input_dim = 10
    rank = 4
    whiten_inputs = True
    progression = 'linear'
    noise_std = 1
    num_hidden_layers = 3
    gamma = 2.5  # \sigma^2 = w^(-gamma)
    lr = 1e-4
    batch_size = 1
    max_singular_value = 100
    decay_rate = 10
    n_samples = 20
    num_epochs = 100000
    teacher = Teacher(output_dim=output_dim, input_dim=input_dim, rank=rank,
                      max_singular_value=max_singular_value,
                      min_singular_value=1e-12,
                      decay_rate=decay_rate,
                      progression=progression)
    # Generate dataset
    dataset = TeacherDataset(teacher, n_samples=n_samples, noise_std=noise_std, whiten_inputs=whiten_inputs)
    model = DLN(input_dim=input_dim, 
                hidden_dims=hidden_dim,
                output_dim=output_dim, 
                num_hidden_layers=num_hidden_layers,
                gamma=gamma)
    obs = Observable(teacher, model)
    modes = obs.mode_matrix()
    print(f"mode shape is {modes.shape} should be {teacher.matrix().shape}")
    trainer = Trainer(teacher, dataset, model, lr=lr, batch_size=batch_size, num_epochs=num_epochs)
    trainer.training_epochs()
    train_loss = trainer.train_losses
    test_loss = trainer.test_losses
    iterations = np.arange(0, len(train_loss))
    # Get the script's directory, then navigate to results
    script_dir = Path(__file__).parent  # /Users/guime/projects/bias_sgd/sgd_dlns_code/scripts/
    results_dir = script_dir.parent / "results"  # /Users/guime/projects/bias_sgd/sgd_dlns_code/results/
    fpath = results_dir
    fname = f"iter_{len(train_loss)}_max_singular_value_{max_singular_value}_gamma_{gamma}_lr_{lr}_batch_{batch_size}_loss.png"
    fpath = fpath.joinpath(fname)
    plt.figure()
    plt.plot(iterations, train_loss, label="Train loss")
    plt.plot(iterations, test_loss, label="Test loss")
    plt.legend()
    plt.savefig(fpath)
    plt.show()
    wandb.log({"loss_curve": wandb.Image(plt)})
