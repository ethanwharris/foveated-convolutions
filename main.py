import torch
import torch.nn.functional as F
import torch.nn as nn

import torchbearer
from torchbearer import Trial
import torchbearer.callbacks as callbacks
import torchbearer.callbacks.imaging as imaging
from scattered_cifar import ScatteredCIFAR10
import torchvision.transforms as transforms

TRANSFORMED = torchbearer.state_key('transformed')


class FoveatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, input_size=96, pool=False):
        super().__init__()
        out_channels = int(out_channels / 4)

        if pool:
            self.conv1 = nn.Sequential(
                nn.AvgPool2d(2),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, dilation=2, padding=2)
        self.border1 = 0

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.border2 = int((input_size - (input_size / 2)) / 2)

        self.conv3 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.border3 = int((input_size - (input_size / 4)) / 2)

        self.conv4 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=4, padding=0, output_padding=1)
        self.border4 = int((input_size - (input_size / 8)) / 2)

    def crop(self, x, border):
        return x[:, :, border:x.size(2) - border, border:x.size(3) - border]

    def forward(self, x):
        x1 = self.conv1(self.crop(x, self.border1))
        x2 = self.conv2(self.crop(x, self.border2))
        x3 = self.conv3(self.crop(x, self.border3))
        x4 = self.conv4(self.crop(x, self.border4))
        return torch.cat((x1, x2, x3, x4), dim=1)


class Net(nn.Module):
    def __init__(self, full_affine=False):
        super(Net, self).__init__()
        self.full_affine = full_affine
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)

        self.fc1 = nn.Linear(64 * 11 * 11, 128)
        self.fc2 = nn.Linear(128, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.ReLU(True)
        )

        if full_affine:
            self.fc_loc = nn.Sequential(
                nn.Linear(32 * 4 * 4, 32),
                nn.ReLU(True),
                nn.Linear(32, 6)
            )
        else:
            self.fc_loc = nn.Sequential(
                nn.Linear(32 * 4 * 4, 32),
                nn.ReLU(True),
                nn.Linear(32, 2)
            )

        # Initialize the weights/bias with identity transformation
        if full_affine:
            self.fc_loc[2].weight.data.zero_()
            self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        else:
            self.fc_loc[2].weight.data.zero_()
            self.fc_loc[2].bias.data.zero_()

        self.id = nn.Parameter(torch.tensor([[1, 0], [0, 1]], dtype=torch.float), requires_grad=False)

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 32 * 4 * 4)
        theta = self.fc_loc(xs)

        if self.full_affine:
            theta = theta.view(-1, 2, 3)
        else:
            theta = torch.cat((self.id.unsqueeze(0).repeat(theta.size(0), 1, 1), theta.unsqueeze(2)), dim=2)

        grid = F.affine_grid(theta, x.size())
        mode = 'bilinear' if self.training else 'nearest'
        x = F.grid_sample(x, grid, mode=mode)

        return x

    def forward(self, x, state=None):
        # transform the input
        x = self.stn(x)

        if state is not None:
            state[TRANSFORMED] = x.detach()

        # Perform the usual forward pass
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class FoveaNet(Net):
    def __init__(self, full_affine=False, pool=False):
        super().__init__(full_affine=full_affine)
        self.conv1 = FoveatedConv2d(3, 32, pool=pool)


class MultiScaleNet(Net):
    def __init__(self, full_affine=False):
        super().__init__(full_affine=full_affine)
        self.conv1 = FoveatedConv2d(9, 32)

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 32 * 4 * 4)
        theta = self.fc_loc(xs)

        if self.full_affine:
            theta = theta.view(-1, 2, 3)
        else:
            theta = torch.cat((self.id.unsqueeze(0).repeat(theta.size(0), 1, 1), theta.unsqueeze(2)), dim=2)

        grid = F.affine_grid(theta, x.size())
        mode = 'bilinear' if self.training else 'nearest'
        x1 = F.grid_sample(x, grid, mode=mode)
        x2 = F.grid_sample(F.interpolate(x, scale_factor=2), grid, mode=mode)
        x3 = F.grid_sample(F.interpolate(x, scale_factor=4), grid, mode=mode)

        tmp = torch.cat((x1, x2, x3), dim=1)
        tmp.detach = lambda: x1.detach()  # HACK!
        return tmp


class StandardNet(Net):
    def stn(self, x):
        return x


class PoolNet(Net):
    def __init__(self):
        super().__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.MaxPool2d(11)
        )

        self.fc1 = nn.Linear(64 * 1 * 1, 128)

    def stn(self, x):
        return x


if __name__ == '__main__':
    model = FoveaNet(full_affine=False, pool=True)
    # model = StandardNet()
    # model = PoolNet()
    # model = Net(full_affine=True)
    # model = MultiScaleNet(full_affine=False)
    train_set = ScatteredCIFAR10('./data', download=True, transform=transforms.Compose([transforms.ToTensor()]))
    train_gen = torch.utils.data.DataLoader(train_set, pin_memory=True, batch_size=128, shuffle=True, num_workers=10)

    test_set = ScatteredCIFAR10('./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    test_gen = torch.utils.data.DataLoader(test_set, pin_memory=True, batch_size=128, shuffle=False, num_workers=10)

    comment = 'foveated5'

    trial = Trial(model, torch.optim.Adam(model.parameters(), lr=0.0001), nn.CrossEntropyLoss(), metrics=['acc', 'loss'],
                  callbacks=[
                      callbacks.MostRecent(comment + '.{epoch:02d}.pt'),
                      imaging.MakeGrid(key=TRANSFORMED, num_images=16, nrow=16, pad_value=1).to_tensorboard(name='transformed', comment=comment).on_val().to_file(comment + '_transformed.png'),
                      imaging.MakeGrid(key=torchbearer.INPUT, num_images=16, nrow=16, pad_value=1).to_tensorboard(name='input', comment=comment).on_val().to_file(comment + '_inputs.png')
                  ])
    trial = trial.with_generators(train_gen, test_gen).to('cuda')
    trial.run(50)
