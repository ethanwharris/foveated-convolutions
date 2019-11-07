import torch
from torchvision import transforms
from torchvision.utils import save_image

import torchbearer

from main import MultiScaleNet, Net, FoveaNet, TRANSFORMED
from scattered_cifar import ScatteredCIFAR10

# Load batch
transform_test = transforms.Compose([
    transforms.ToTensor()
])
test_set = ScatteredCIFAR10('./data', train=False, download=True, transform=transforms.Compose([transform_test]))
test_gen = torch.utils.data.DataLoader(test_set, pin_memory=True, batch_size=128, shuffle=False, num_workers=4)
batch, _ = next(iter(test_gen))
inputs = batch[:16]

# Load models
stn = Net()
stn.load_state_dict(torch.load('standard1.49.pt')[torchbearer.MODEL])

multi = MultiScaleNet()
multi.load_state_dict(torch.load('multi1.49.pt')[torchbearer.MODEL])

fovea = FoveaNet(pool=True)
fovea.load_state_dict(torch.load('foveated1.49.pt')[torchbearer.MODEL])

stn_fa = Net(full_affine=True)
stn_fa.load_state_dict(torch.load('standardFA1.49.pt')[torchbearer.MODEL])

fovea_fa = FoveaNet(full_affine=True, pool=True)
fovea_fa.load_state_dict(torch.load('foveatedFA1.49.pt')[torchbearer.MODEL])

# Obtain transformed results
state = {}

_ = stn(inputs, state)
x1 = state[TRANSFORMED]

_ = multi(inputs, state)
x2 = state[TRANSFORMED]

_ = fovea(inputs, state)
x3 = state[TRANSFORMED]

_ = stn_fa(inputs, state)
x4 = state[TRANSFORMED]

_ = fovea_fa(inputs, state)
x5 = state[TRANSFORMED]

# Make grid and save
save_image(torch.cat((inputs, x1, x2, x3, x4, x5), dim=0), 'figure.png', nrow=16, pad_value=1)
