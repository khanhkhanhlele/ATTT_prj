import torch

def confusionMatrix(model, c, dataloader, device):
    cm = torch.zeros((c, c)).int()
    for i, (x, l) in enumerate(dataloader):
        x = x.to(device)
        l = l.to(device)
        pred = model(x).detach()
        for ii in range(pred.shape[0]):
            cm[pred[ii].argmax(), l[ii].argmax()] += 1
    return cm

def accuracy(cm):
    return cm.trace()/cm.sum()

def precision(cm):
    return (cm.diag()/cm.sum(dim=1)).mean()

def recall(cm):
    return (cm.diag()/cm.sum(dim=0)).mean()

def f1(cm):
    return 2*(precision(cm)*recall(cm))/(precision(cm) + recall(cm))
