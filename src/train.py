import torch
import matplotlib.pyplot as plt
from utils.metric import *

def train(model, lossFunction, dataloader_train, dataloader_val=None, num_epochs=50, lr=1e-5, milestones_lr=[36], gamma=0.1, device=torch.device('cpu'), display=False):
    criterion = lossFunction
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones_lr, gamma=gamma)
    model.train()
    for epoch in range(num_epochs):
        for i, (x, l) in enumerate(dataloader_train):
            x = x.to(device)
            l = l.to(device)
            pred = model(x)
            loss = criterion(pred, l)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if display:
                print('epoch {}/{}: [{}/{}] -------> loss: {}'.format(
                    epoch + 1,
                    num_epochs,
                    i + 1,
                    len(dataloader_train),
                    loss.item()
                ))
        if display and dataloader_val != None:
            cm = confusionMatrix(model, pred.shape[1], dataloader_val, device)
            plt.matshow(cm)
            for ii in range(cm.shape[0]):
                for iii in range(cm.shape[0]):
                    plt.text(iii, ii, str(cm[ii, iii].item()), va='center', ha='center')
            plt.show()
            print('accuracy: {}'.format(accuracy(cm).item()))
            print('precision: {}'.format(precision(cm).item()))
            print('recall: {}'.format(recall(cm).item()))
            print('f1: {}'.format(f1(cm).item()))
        scheduler.step()
