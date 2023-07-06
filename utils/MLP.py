import torch
from utils.modules import downLayer

class MLP(torch.nn.Module):
    def __init__(self, inputDim, outputDim, num_downLayer):
        super().__init__()
        self.dnn = torch.nn.Sequential(
            *[downLayer(inputDim//(2**i)) for i in range(num_downLayer)],
            torch.nn.Linear(inputDim//(2**num_downLayer), outputDim)
        )
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        out = self.dnn(x)
        out = self.softmax(out)
        return out
