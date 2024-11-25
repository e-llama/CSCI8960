
import torch.optim as optim


def sdg_default(model):
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0)
    return optimizer


def sgd_momentum(model, mome):
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=mome)
    return optimizer

def sgd_momentum_negmax(model, mome):
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=mome, maximize=True)
    return optimizer



def sgd_momentum_damp(model, mome, damp):
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=mome, dampening=damp)
    return optimizer


def sgd_momentum_nesterov(model, mome):
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=mome, nesterov=True)
    return optimizer

def sgd_decay(model, decay):
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay = decay)
    return optimizer

def sgd_decay_max(model, decay):
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay = decay, maximize=True)
    return optimizer