import torch
from torchvision import datasets, transforms
from flopth import flopth
import torch.nn as nn

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = get_device()
print(device)

parameters = {
    'batch_size': 64,
    'classes': 10,
    'epochs': 5,
    'learning_rate': 0.01
}

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.FashionMNIST('data/MNIST/', train=True, download=True, transform=data_transform)
test_dataset = datasets.FashionMNIST('data/MNIST/', train=False, download=True, transform=data_transform)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=parameters['batch_size'])
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=parameters['batch_size'])


class ResNextBlock(nn.Module):
    expansion_factor = 2

    def __init__(self, input_channels, cardinality, bottleneck_width, downsample=None, stride=1):
        super(ResNextBlock, self).__init__()
        output_channels = cardinality * bottleneck_width
        self.shortcut = downsample
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=stride, padding=1, groups=cardinality),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels * self.expansion_factor, kernel_size=1),
            nn.BatchNorm2d(output_channels * self.expansion_factor),
            nn.ReLU()
        )

    def forward(self, x):
        shortcut = x
        out = self.conv_layers(x)

        if self.shortcut is not None:
            shortcut = self.shortcut(shortcut)

        return nn.ReLU()(out + shortcut)


class ResNext(nn.Module):
    def __init__(self, block, layers, cardinality, bottleneck_width, input_channels, num_classes):
        super(ResNext, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, layers[0])
        self.layer2 = self._make_layer(block, layers[1], stride=2)
        self.layer3 = self._make_layer(block, layers[2], stride=2)
        self.layer4 = self._make_layer(block, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.cardinality * self.bottleneck_width * 2, num_classes)

    def _make_layer(self, block, blocks, stride=1):
        downsample = None
        output_channels = self.cardinality * self.bottleneck_width * 2

        if stride != 1 or self.in_channels != output_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, output_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(output_channels)
            )

        layers = []
        layers.append(block(self.in_channels, self.cardinality, self.bottleneck_width, downsample, stride))
        self.in_channels = output_channels

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, self.cardinality, self.bottleneck_width))

        self.bottleneck_width *= 2

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


network = ResNext(ResNextBlock, [2, 2, 2, 2], cardinality=32, bottleneck_width=4, input_channels=1, num_classes=parameters['classes'])

loss_function = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(network.parameters(), lr=parameters['learning_rate'], weight_decay=0.001, momentum=0.9)

total_steps = len(train_dataloader)

# same functions for computing accuracy, training and evaluation go here

for epoch in range(parameters['epochs']):
    train_loss = train_one_epoch(epoch)

accuracy_list = test_model()

total_accuracy = sum([acc[0][0].item() for acc in accuracy_list]) / len(accuracy_list)
print(total_accuracy)

model_flops, model_params = flopth(network, in_size=((1, 28, 28),))
print('FLOPs: ', model_flops)
print('Params: ', model_params)
