
import torch.nn as nn

class ResNet:

    def __init__(self, num_classes=1000):
        super(ResNet, self).__init__()
        #populate the layers with your custom functions or pytorch
        #functions.
        
        self.conv1 = nn.Conv2d(3, 64, (7, 7), padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d((2, 2))
        
        # TODO: investigate adding residuals
        
        self.layer1 = new_block(64, 64)
        self.layer2 = new_block(64, 128) 
        self.layer3 = new_block(128, 256) 
        self.layer4 = new_block(256, 512) 
        
        self.avgpool = nn.AvgPool2d((1,1))
        self.fc = nn.Linear(131072, num_classes) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def new_block(self, in_planes, out_planes, stride = 1):
        layers = [nn.Conv2d(in_planes, out_planes, (3, 3), stride=stride, padding=1),
                  nn.Conv2d(out_planes, out_planes, (3, 3), stride=stride, padding=1)]
        return nn.Sequential(*layers)