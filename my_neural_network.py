import torch
import torch.nn as nn
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.input_layer = nn.Linear(2, 3, bias=False)
        self.out_layer = nn.Linear(3, 1, bias=False)
        self._initialize_weights()

    def forward(self, x):
        h1 = self.input_layer(x)
        out = self.out_layer(h1)
        return h1, out

    def _initialize_weights(self):
        with torch.no_grad():
            K = torch.Tensor([[0.5 , 0.25], [0.2, 0.4], [0.9, 0.8]])
            self.input_layer.weight = torch.nn.Parameter(K)
            K = torch.Tensor([0.1 , 0.5, 0.3])
            self.out_layer.weight = torch.nn.Parameter(K)