"""
    This is code from:
    https://github.com/CuthbertCai/pytorch_resnet
"""


MAX_GRAD_NORM = 1.2
EPSILON = 50.0
DELTA = 1e-5
EPOCHS = 60
LR = 1

BATCH_SIZE = 512
MAX_PHYSICAL_BATCH_SIZE = 128



import warnings
warnings.simplefilter("ignore")


from opacus.validators import ModuleValidator
import torch
from torchvision import datasets, transforms
from torch.utils.data import dataloader
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import time
import torch.nn.init as init

device = torch.device('cuda')

#preprocess the images

import torch
import torchvision
import torchvision.transforms as transforms


# These values, specific to the CIFAR10 dataset, are assumed to be known.
# If necessary, they can be computed with modest privacy budgets.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV),
])

#make dataloader for training and testing
train_data = datasets.CIFAR10(root='./data', train= True, transform= transform, download= True)
test_data = datasets.CIFAR10(root='./data', train= False, transform= transform, download= True)


train_data_size = len(train_data)
test_data_size = len(test_data)
train_loader = dataloader.DataLoader(dataset= train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers= 2)
test_loader = dataloader.DataLoader(dataset= test_data, batch_size=BATCH_SIZE, num_workers= 2)

#use_gpu = torch.cuda.is_available()
use_gpu = True
device = torch.device("cuda")


from wrn_imp import wrn16_4




#------------------------
# Training
#------------------------
import pandas as pd

def steps_to_csv(step, loss, duration):
  data = {'Step': [step], 'Loss': [loss], 'Duration': [duration]}
  df = pd.DataFrame(data)

  try:

    existing_df = pd.read_csv('steps.csv')

    combined_df = pd.concat([existing_df, df])

    combined_df.to_csv('steps.csv', index=False)

  except FileNotFoundError:

    df.to_csv('steps.csv', index=False)

def epochs_to_csv(epoch, loss, duration):
  data = {'Epoch': [epoch], 'Loss': [loss], 'Duration': [duration]}
  df = pd.DataFrame(data)

  try:
    existing_df = pd.read_csv('epoch.csv')

    combined_df = pd.concat([existing_df, df])

    combined_df.to_csv('epoch.csv', index=False)

  except FileNotFoundError:
    df.to_csv('epoch.csv', index=False)


def epochs_to_csv(epoch, loss, accuracy, epsilon, delta, timen):
  data = {
      'Epoch': [epoch], 
      'Loss': [loss], 
      'Accuracy': [accuracy], 
      'Epsilon': [epsilon],
      'Delta': [delta],
      'time': [timen]
      }
  df = pd.DataFrame(data)

  try:
    existing_df = pd.read_csv('epoch.csv')

    combined_df = pd.concat([existing_df, df])

    combined_df.to_csv('epoch.csv', index=False)

  except FileNotFoundError:
    df.to_csv('epoch.csv', index=False)


#train

import numpy as np
from opacus.utils.batch_memory_manager import BatchMemoryManager





def train(model, train_loader, optimizer, epoch, device):
    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    top1_acc = []
    
    with BatchMemoryManager(
        data_loader=train_loader, 
        max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE, 
        optimizer=optimizer
    ) as memory_safe_data_loader:

        for i, (images, target) in enumerate(memory_safe_data_loader):   
            optimizer.zero_grad()
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()

            # measure accuracy and record loss
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

            loss.backward()
            optimizer.step()

            if (i+1) % 200 == 0:
                epsilon = privacy_engine.get_epsilon(DELTA)
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
                    f"(ε = {epsilon:.2f}, δ = {DELTA})"
                )
    epochs_to_csv(epoch, np.mean(losses), np.mean(top1_acc)*100, epsilon, DELTA, time.time())





def accuracy(preds, labels):
    return (preds == labels).mean()



#------------------------
# Privacy Engine
#------------------------

from opacus.validators import ModuleValidator




from opacus.privacy_engine import PrivacyEngine

# enter PrivacyEngine
privacy_engine = PrivacyEngine()

wrn = wrn16_4()


if not ModuleValidator.is_valid(wrn):
    errors = ModuleValidator.validate(wrn, strict=False)
    print("errors: ", errors[-5:])
    wrn = ModuleValidator.fix(wrn)
    
    
# optimizer
from optimizers import sgd_default

optimizer = sgd_default(wrn);



wrn, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=wrn,
    optimizer=optimizer,
    data_loader=train_loader,
    epochs=EPOCHS,
    target_epsilon=EPSILON,
    target_delta=DELTA,
    max_grad_norm=MAX_GRAD_NORM,
)


#if use_gpu:
#    res_net = res_net.to(device)
device = torch.device('cuda')
wrn.to(device)
#res_net = ModuleValidator.fix(res_net)

criterion = torch.nn.CrossEntropyLoss()




def test(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top1_acc = []

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

    top1_avg = np.mean(top1_acc)

    print(
        f"\tTest set:"
        f"Loss: {np.mean(losses):.6f} "
        f"Acc: {top1_avg * 100:.6f} "
    )
    return np.mean(top1_acc)



#res_net = train(res_net, lr_optimizer, optimizer, criterion, train_loader)
#res_net = train(res_net, train_loader, optimizer, 20, device)


from tqdm import tqdm

for epoch in tqdm(range(EPOCHS), desc="Epoch", unit="epoch"):
    train(wrn, train_loader, optimizer, epoch + 1, device)



#calculate the accuracy of the trained model


top1_acc = test(wrn, test_loader, device)

torch.save(wrn.state_dict(), "dpsgd_1_wrn_20.pt");