import torch
from transformers import DistilBertModel

class DistilBERT(torch.nn.Module):
    def __init__(self, outputDim):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.embedding = self.distilbert.embeddings.requires_grad_(True)
        self.transformer = torch.nn.ModuleList([self.distilbert.transformer.layer[i].requires_grad_(True) for i in range(6)])
        self.output_bert = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            torch.nn.ReLU(),
            torch.nn.Linear(768, outputDim)
        )
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, token):
        out = self.embedding(token['input_ids'])
        for i in range(6):
            out = self.transformer[i](out, token['attention_mask'])[-1]
        out = self.output_bert(out[:, 0, :])
        out = self.softmax(out)
        return out
