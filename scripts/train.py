import torch
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm

from torch import nn
from teacher import *
from models import *


# Code training loop class with SGD
class Trainer():
    def __init__(self, teacher: Teacher, 
                 dataset: TeacherDataset,
                 model: DLN,
                 lr: float = 0.01,
                 batch_size: int = 1,
                 device: str = 'cpu'):
        
        self.teacher = teacher 
        self.dataset = dataset
        self.model = model.to(device)
        self.lr = lr
        self.batch_size = batch_size
        self.device = device

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()

        self.train_losses = []
        self.test_losses = []
        self.grad_norms = []

    def training_loop(self):
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

# Plotting functions for losses, modes etc


if __name__ == '__main__':
    output_dim = 10
    hidden_dim = 100
    input_dim = 10
    rank = 3
    whiten_inputs = True
    noise_std = 0
    num_hidden_layers = 3
    gamma = 0.8  # \sigma^2 = w^(-gamma)
    lr = 1e-3
    batch_size = 1
    max_singular_value = 1
    decay_rate = 1.5
    n_samples = 1000
    teacher = Teacher(output_dim=output_dim, input_dim=input_dim, rank=rank,
                      max_singular_value=max_singular_value,
                      min_singular_value=1e-12,
                      decay_rate=decay_rate)
    # Generate dataset
    dataset = TeacherDataset(teacher, n_samples=n_samples, noise_std=noise_std, whiten_inputs=whiten_inputs)
    model = DLN(input_dim=input_dim, 
                hidden_dims=hidden_dim,
                output_dim=output_dim, 
                num_hidden_layers=num_hidden_layers,
                gamma=gamma)
    trainer = Trainer(teacher, dataset, model, lr=lr, batch_size=batch_size)
    trainer.training_loop()
    train_loss = trainer.train_losses
    test_loss = trainer.test_losses
    samples = np.arange(0, len(train_loss))
    plt.figure()
    plt.plot(samples, train_loss)
    plt.plot(samples, test_loss)
    plt.show()