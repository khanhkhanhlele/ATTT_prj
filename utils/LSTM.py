import torch

class LSTM(torch.nn.Module):
    def __init__(self, vocabSize, inputDim, outputDim, hiddenDim, num_layers=3):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocabSize, inputDim)
        self.lstm = torch.nn.LSTM(input_size=inputDim, hidden_size=hiddenDim, num_layers=num_layers, batch_first=True)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hiddenDim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, outputDim)
        )
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        emb = self.embedding(x)
        hid, (hidn, cn) = self.lstm(emb)
        out = self.fc(hidn[-1, :, :])
        out = self.softmax(out)
        return out
