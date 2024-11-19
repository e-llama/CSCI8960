"""
    This is code from:
    https://github.com/CuthbertCai/pytorch_resnet
"""

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


BATCH_SIZE = 512
MAX_PHYSICAL_BATCH_SIZE = 128

# These values, specific to the CIFAR10 dataset, are assumed to be known.
# If necessary, they can be computed with modest privacy budgets.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV),
])
"""
train_transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]) """

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


class ResBlock(nn.Module):
    """The block of residual network"""

    def __init__(self, in_channel, out_channel, size):
        super(ResBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.size = size
        self.conv1 = torch.nn.Conv2d(in_channel, out_channel, (3, 3), padding= 1)
        self.conv1_ = torch.nn.Conv2d(in_channel, out_channel, (3, 3), stride= 2, padding= 1)
        self.conv2 = torch.nn.Conv2d(out_channel, out_channel, (3, 3), padding= 1)
        self.bn = torch.nn.BatchNorm2d(out_channel)
        self.relu = torch.nn.PReLU()
        self.max_pool = torch.nn.MaxPool2d(2, 2)

    def forward(self, input):
        identity = input.to(device)
        #if the channels and image size would not change after the block
        if self.in_channel == self.out_channel:
            output = self.conv1(input)
            output = self.bn(output)
            output = self.relu(output)
            output = self.conv2(output)
            output = self.bn(output)
            output = torch.add(output, identity)
            output = self.relu(output)
        else:
            #if the channels and image size would change after the block
            identity = self.max_pool(identity.to(device))
            identity = torch.cat((identity.to(device), Variable(torch.zeros(identity.size())).to(device)), 1)
            identity = identity.to(device)
            
            output = self.conv1_(input)
            output = self.bn(output)
            output = self.relu(output)
            output = self.conv2(output)
            output = self.bn(output)
            output = torch.add(output, identity)
            output = self.relu(output)

        return output

class ResNet(nn.Module):
    """The architecture of residual network"""

    def __init__(self):
        super(ResNet, self).__init__()
        self.relu = torch.nn.PReLU()
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.conv1 = torch.nn.Conv2d(3, 16, (3, 3), padding= 1)
        self.res_block1 = ResBlock(16, 16, 32)
        self.res_block2 = ResBlock(16, 16, 32)
        self.res_block3 = ResBlock(16, 16, 32)
        self.res_block4 = ResBlock(16, 32, 32)
        self.res_block5 = ResBlock(32, 32, 16)
        self.res_block6 = ResBlock(32, 32, 16)
        self.res_block7 = ResBlock(32, 64, 16)
        self.res_block8 = ResBlock(64, 64, 8)
        self.res_block9 = ResBlock(64, 64, 8)
        self.global_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(64, 10)
        self.softmax = torch.nn.Softmax(dim=1)
        self.init_params()

    def init_params(self):
        """Initialize the parameters in the ResNet model"""

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, std= 0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):
        
        input = input.to(device)
        
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.res_block1(output)
        output = self.res_block2(output)
        output = self.res_block3(output)
        output = self.res_block4(output)
        output = self.res_block5(output)
        output = self.res_block6(output)
        output = self.res_block7(output)
        output = self.res_block8(output)
        output = self.res_block9(output)
        output = self.global_pool(output)
        output = output.view(-1, 64)
        output = self.fc(output)
        #print("output: ", output)
        output = self.softmax(output)

        return output


from resnet_imp import resnet20




#--------------------------
# learning rate optimizer
#--------------------------
def lr_optimizer(optimizer, step):
    """The schedule of changing the learning rate

       When the step is 32000 or 48000, the learning rate would be divided by 10

    :param: optimizer: the optimizer to update the parameters
    :param: step: the training iterations
    """

    if step == 32000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 10
        print('Learning rate is set to {: 4f}'.format(optimizer.param_groups[0]['lr']))
    elif step == 48000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 10
        print('Learning rate is set to {: 4f}'.format(optimizer.param_groups[0]['lr']))
    return optimizer




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
"""
def train(res_net, lr_optimizer, optimizer, criterion, train_loader, max_epoch = 20):
    #Train the ResNet model

    step = 0
    
    start_time = time.time()
    for epoch in range(max_epoch):
        
        epoch_time_start = time.time();
        running_loss = 0.0
        for data in train_loader:
            step += 1
            train_input, train_label = data
            if use_gpu:
                train_input, train_label = Variable(train_input.to(device)), Variable(train_label.to(device))
            else:
                train_input, train_label = Variable(train_input), Variable(train_label)

            #optimizer = lr_optimizer(optimizer, step)
            optimizer.zero_grad()
            train_outputs = res_net(train_input)
            loss = criterion(train_outputs, train_label)
            loss.backward()
            optimizer.step()
            running_loss += loss.data

            if step % 10 == 0:
                end_time = time.time()
                duration = end_time - start_time
                start_time = time.time()
                print('Step: {}, Loss: {: 4f}, Durantion per 10 steps: {: 2f}'.format(step, loss.data, duration))
                steps_to_csv(step, loss.cpu().data, duration)
                
        epoch_time_end = time.time();
        epoch_duration = epoch_time_end-epoch_time_start
        epoch_loss = running_loss / (train_data_size / 128)
        print('Epoch: {}, Loss: {: 4f}'.format(epoch, epoch_loss))
        epochs_to_csv(epoch, epoch_loss.cpu(), epoch_duration)
        
    return res_net

"""


import numpy as np
from opacus.utils.batch_memory_manager import BatchMemoryManager

MAX_GRAD_NORM = 1.2
EPSILON = 50.0
DELTA = 1e-5
EPOCHS = 20
LR = 1e-3
BATCH_SIZE = 512
MAX_PHYSICAL_BATCH_SIZE = 128


"""
def train(model, train_loader, optimizer, max_epoch, device):
        #Train the ResNet model

    #step = 0
    
    #start_time = time.time()
    for epoch in range(max_epoch):
        epoch_time_start = time.time();
        
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
        epoch_time_end = time.time();
        epoch_duration = epoch_time_end - epoch_time_start
        epochs_to_csv(epoch, np.mean(losses), np.mean(top1_acc) * 100, epsilon, DELTA, epoch_duration)

"""

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



# optimizer = optim.SGD(res_net.parameters(), lr = 0.1, momentum= 0.9, weight_decay= 0.0001)

#the parameters of PRelu could not have weight decay
"""
optimizer = optim.SGD([
    {'params': res_net.bn1.parameters(), 'weight_decay': 0.0001},
    {'params': res_net.conv1.parameters(), 'weight_decay': 0.0001},
    {'params': res_net.res_block1.parameters(), 'weight_decay': 0.0001},
    {'params': res_net.res_block2.parameters(), 'weight_decay': 0.0001},
    {'params': res_net.res_block3.parameters(), 'weight_decay': 0.0001},
    {'params': res_net.res_block4.parameters(), 'weight_decay': 0.0001},
    {'params': res_net.res_block5.parameters(), 'weight_decay': 0.0001},
    {'params': res_net.res_block6.parameters(), 'weight_decay': 0.0001},
    {'params': res_net.res_block7.parameters(), 'weight_decay': 0.0001},
    {'params': res_net.res_block8.parameters(), 'weight_decay': 0.0001},
    {'params': res_net.res_block9.parameters(), 'weight_decay': 0.0001},
    {'params': res_net.relu.parameters()},
], lr= 0.1, momentum= 0.9)
"""
#optimizer = optim.SGD(res_net.parameters(), lr=0.01, momentum=0.9)


def accuracy(preds, labels):
    return (preds == labels).mean()



#------------------------
# Privacy Engine
#------------------------

from opacus.validators import ModuleValidator

#errors = ModuleValidator.validate(res_net, strict=False)
#rint("errors: ", errors[-5:])


from opacus.privacy_engine import PrivacyEngine

# enter PrivacyEngine
privacy_engine = PrivacyEngine()

#model = res_net
#data_loader = train_loader

#res_net = ResNet()
res_net = resnet20()

errors = ModuleValidator.validate(res_net, strict=False)
print("errors: ", errors[-5:])

if not ModuleValidator.is_valid(res_net):
    res_net = ModuleValidator.fix(res_net)

optimizer = optim.SGD(res_net.parameters(), lr=0.01, momentum=0)

MAX_GRAD_NORM = 1.2
EPSILON = 50.0
DELTA = 1e-5
EPOCHS = 20
LR = 1e-3

res_net, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=res_net,
    optimizer=optimizer,
    data_loader=train_loader,
    epochs=EPOCHS,
    target_epsilon=EPSILON,
    target_delta=DELTA,
    max_grad_norm=MAX_GRAD_NORM,
)


#if use_gpu:
#    res_net = res_net.to(device)
device = torch.device('cuda:0')
res_net.to(device)
#res_net = ModuleValidator.fix(res_net)

criterion = torch.nn.CrossEntropyLoss()


print()
print()
print("=================================")
print("new errors")
print("=================================")

errors = ModuleValidator.validate(res_net, strict=False)
print("errors: ", errors[-5:])

"""
privacy_engine = PrivacyEngine()

model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    epochs=EPOCHS,
    target_epsilon=EPSILON,
    target_delta=DELTA,
    max_grad_norm=MAX_GRAD_NORM,
)
"""


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
    train(res_net, train_loader, optimizer, epoch + 1, device)



#calculate the accuracy of the trained model

"""
corrects = 0
for data in test_loader:
    test_input, test_label = data

    test_input, test_label = Variable(test_input.to(device)), Variable(test_label.to(device))


    test_outputs = res_net(test_input)
    _, preds = torch.max(test_outputs.data, 1)
    corrects += torch.sum(preds == test_label.data)

accuracy = corrects / test_data_size
print('The accuracy in the test dataset is {: 4f}'.format(accuracy))

"""

top1_acc = test(res_net, test_loader, device)

torch.save(res_net.state_dict(), "dpsgd_1_resnet_20.pt");