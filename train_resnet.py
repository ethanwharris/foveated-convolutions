import torch
from torch import nn, optim
from torchvision import transforms

import torchbearer
from torchbearer.callbacks import UnpackState

from main import FoveaNet
from resnet import ResNet18
from scattered_cifar import ScatteredCIFAR10


foveanet = FoveaNet(pool=True)
foveanet.load_state_dict(torch.load('foveated5.49.pt')[torchbearer.MODEL])

# standardnet = Net()
# standardnet.load_state_dict(torch.load('standard5.49.pt')[torchbearer.MODEL])

# msnet = MultiScaleNet()
# msnet.load_state_dict(torch.load('multi5.49.pt', map_location='cpu')[torchbearer.MODEL])

resnet = ResNet18()


class PreResNet(nn.Module):
    def __init__(self, resnet, basenet):
        super(PreResNet, self).__init__()

        self.resnet = resnet
        self.basenet = basenet
        for param in self.basenet.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.basenet.stn(x)[:, :3, (x.size(2) // 2) - 16: (x.size(2) // 2) + 16, (x.size(3) // 2) - 16: (x.size(3) // 2) + 16].detach()
        return self.resnet(x)


net = PreResNet(resnet, foveanet)

# net = nn.DataParallel(resnet)

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.ToTensor()
])

train_set = ScatteredCIFAR10('./data', download=True, transform=transforms.Compose([transform_train]))
# train_set = CIFAR10('./data', download=True, transform=transforms.Compose([transform_train]))

train_gen = torch.utils.data.DataLoader(train_set, pin_memory=True, batch_size=128, shuffle=True, num_workers=4)

test_set = ScatteredCIFAR10('./data', train=False, download=True, transform=transforms.Compose([transform_test]))
# test_set = CIFAR10('./data', train=False, download=True, transform=transforms.Compose([transform_test]))

test_gen = torch.utils.data.DataLoader(test_set, pin_memory=True, batch_size=128, shuffle=False, num_workers=4)

optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

from torchbearer.callbacks import MultiStepLR, TensorBoard
from torchbearer import Trial

from datetime import datetime
current_time = datetime.now().strftime('%b%d_%H-%M-%S')
trial = Trial(net, optimizer, nn.CrossEntropyLoss(), metrics=['acc', 'loss'], callbacks=[UnpackState(), TensorBoard(write_graph=False, comment=current_time), MultiStepLR([100, 150])])
trial.with_generators(train_generator=train_gen, val_generator=test_gen).to('cuda')
trial.run(200, verbose=1)
