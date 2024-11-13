#------------------------
# Preprocessing
#------------------------
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torch
from torchvision import datasets, transforms
from torch.utils.data import dataloader
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import time
import torch.nn.init as init

transform = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]);

train_dataset = CIFAR10(root=args.data_root, train=True, download=True, transform=transform);
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=int(args.sample_rate * len(train_dataset)),
    generator=generator,
    num_workers=args.workers,
    pin_memory=True,
);

test_dataset = CIFAR10(root=args.data_root, train=False, download=True, transform=transform);
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.batch_size_test,
    shuffle=False,
    num_workers=args.workers,
)


#------------------------
# Optimzer
#------------------------
import torch.optim as optim

optimizer = optim.SGD(
    model.parameters(),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay,
)


#------------------------
# Privacy Engine
#------------------------
from opacus import PrivacyEngine

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

print(f"Using sigma={optimizer.noise_multiplier} and C={MAX_GRAD_NORM}")


#------------------------
# Model
#------------------------
import torch.nn as nn

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
        self.softmax = torch.nn.Softmax()
        self.init_params()

    def init_params(self):
        """Initialize the parameters in the ResNet model"""

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            if isinstance(m, nn.Linear):
                init.normal(m.weight, std= 0.01)
                if m.bias is not None:
                    init.constant(m.bias, 0)

    def forward(self, input):
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
        output = self.softmax(output)

        return output

device = torch.device("cuda")
model = ResNet()
model = model.to(device)
args = ""

#------------------------
# Training
#------------------------
def train(args, model, train_loader, optimizer, privacy_engine, epoch, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    losses, top1_acc = [], []
    
    step = 0
    start_time = time.time()
    
    for images, target in train_loader:
        images = images.to(device)
        target = target.to(device)
        
        # run forward step
        output = model(images)
        loss = criterion(output, target) # compute the loss
        
        # choose the largest class probability
        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = target.detach().cpu().numpy()
        
        # measure accuracy and record loss
        acc1 = (preds == labels).mean()
        
        # run backward step (computing gradients)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()

    print("accuracy: ", acc1)



#------------------------
# Running
#------------------------
res_net = ResNet()
#if use_gpu:
#    res_net = res_net.to(device)
device = torch.device('cuda')
res_net.to(device)
epoch=50;

res_net = train(args, model, train_loader, optimizer, privacy_engine, epoch, device);

torch.save(res_net.state_dict(), "dp_resnet_20-original.pt");