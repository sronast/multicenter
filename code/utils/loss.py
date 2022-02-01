#python

#libs

#pytorch
import torch.nn as nn

###

###

crossentropy_loss = nn.CrossEntropyLoss()

def celoss(pred, label):
    return crossentropy_loss(pred, label)