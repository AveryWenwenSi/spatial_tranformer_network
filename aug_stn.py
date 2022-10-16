import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
import pdb

import os
import numpy as np
import pdb

os.environ["TORCH_HOME"] = "./data"

device_no = 2
device = torch.device(device_no if torch.cuda.is_available() else "cpu")

# Training dataset
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=64, shuffle=True, num_workers=4)
# Test dataset
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=64, shuffle=True, num_workers=4)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(10, 20, kernel_size=(5, 5))
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def spatial_sampler(self, x, theta):
        grid = torch.nn.functional.affine_grid(theta, size=x.size())
        sampled_data = torch.nn.functional.grid_sample(x, grid)

        return sampled_data

    def forward(self, x, theta):
        x = self.spatial_sampler(x, theta)

        x = self.conv1(x)
        x = nn.functional.max_pool2d(x, (2, 2))
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = nn.functional.max_pool2d(x, (2, 2))
        x = nn.functional.relu(x)
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = nn.functional.dropout(x, training=self.training)  # module param maybe.
        x = self.fc2(x)

        return nn.functional.log_softmax(x, dim=1)


net = Net().to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
eps = 0.01   # theta step size

def train(epoch):
    theta = Variable(torch.Tensor([[1.,0.,0.],[0.,1.,0.]]), requires_grad=True).to(device)
    theta.retain_grad()
    net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = net(data, theta.repeat(data.shape[0],1,1))

        loss = nn.functional.nll_loss(output, target)
        loss.backward()

        grad = theta.grad
        # apply FGSM on affine matrix
        # Q1: is it cumulative?
        theta -= eps * torch.sign(grad)

        # this time update net with the new transferred images
        optimizer.zero_grad()
        output = net(data, theta.repeat(data.shape[0],1,1))
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # if batch_idx % 500 == 0:
    # print(theta)
    # print(grad)
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    return theta


def test(theta):
    net.eval()
    t_test_loss = 0
    n_correct = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        output = net(data, theta.repeat(data.shape[0],1,1))

        test_loss = nn.functional.nll_loss(output, target)
        t_test_loss += test_loss
        # print("test_loss : ", test_loss.item())

        pred = output.max(1, keepdim=True)[1]
        n_correct += (pred.flatten() == target).sum()

    avg_loss = n_correct / len(test_loader.dataset)
    print("test_acc : ", avg_loss.item())


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


# We want to visualize the output of the spatial transformers layer
# after the training, we visualize a batch of input images and
# the corresponding transformed batch using STN.


def visualize_stn(theta, epoch):
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(test_loader))[0].to(device)

        input_tensor = data.cpu()
        transformed_input_tensor = net.spatial_sampler(data, theta.repeat(data.shape[0],1,1)).cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')
        plt.show()
        plt.savefig('vis_{}.png'.format(epoch))


for epoch in range(1, 10 + 1):
    theta = train(epoch)
    test(theta)
    # Visualize the STN transformation on some input batch
    visualize_stn(theta, epoch)

plt.show()
