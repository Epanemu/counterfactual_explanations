import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# progress bar
from tqdm import tqdm

class SimpleDataset(Dataset):
  def __init__(self, X, y):
    self.X = torch.tensor(X,dtype=torch.float32)
    self.y = torch.tensor(y,dtype=torch.float32)

  def __len__(self):
    return len(self.y)

  def __getitem__(self,idx):
    return self.X[idx], self.y[idx]

class NNModel:
    def __init__(self, input_size, hidden_sizes, output_size):
        layers = []
        prev_size = input_size
        for curr_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, curr_size))
            layers.append(nn.ReLU())
            prev_size = curr_size
        layers.append(nn.Linear(prev_size, output_size))
        # no ReLU on the output

        self.model = nn.Sequential(*layers)
        self.loss_f = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters())

    def predict(self,x):
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32)
            res = self.model(x)
        return res

    def train(self, X_train, y_train, epochs=50, batch_size=64):
        dataset = SimpleDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        print("Training:")
        self.model.train()
        for j in tqdm(range(epochs)):
            for i, (X, y) in enumerate(dataloader):
                y_pred = self.model(X)

                loss = self.loss_f(y_pred, y.reshape(-1,1))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def test(self, X_test, y_test):
        dataset = SimpleDataset(X_test, y_test)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        print("Testing:")
        self.model.eval()
        losses = []
        correct = []
        with torch.no_grad():
            for i, (X, y) in enumerate(dataloader):
                y_pred = self.model(X)
                losses.append(self.loss_f(y_pred, y.reshape(-1,1)).item())
                correct.append(((y_pred > 0) == y).item())
        print(f"Accuracy: {sum(correct) / y_test.shape[0] * 100:.2f}%")
        print("Average loss:", sum(losses) / y_test.shape[0])


    def get_params(self):
        types = []
        weights = []
        biases = []
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                types.append("linear")
                weights.append(np.array(layer.state_dict()['weight']))
                biases.append(np.array(layer.state_dict()['bias']))
            elif isinstance(layer, nn.ReLU):
                types.append("ReLU")
                weights.append(None)
                biases.append(None)
            else:
                types.append(None)
                weights.append(None)
                biases.append(None)

        return types, weights, biases
