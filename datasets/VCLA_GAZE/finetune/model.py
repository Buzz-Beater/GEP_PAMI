"""
Created on 10/29/18

@author: Baoxiong Jia

Description:

"""

import torch
import torchvision
from datasets.VCLA_GAZE.metadata import VCLA_METADATA
metadata = VCLA_METADATA()

model_dict = {
    'resnet': lambda num_classes, feature_dims : ResNet152(num_classes=num_classes, feature_dim=feature_dims),
    'densenet': lambda num_classes, feature_dims : DenseNet(num_classes=num_classes, feature_dim=feature_dims),
    'vgg16': lambda num_classes, feature_dims : VGG16(num_classes=num_classes, feature_dim=feature_dims)
}

class VGG16(torch.nn.Module):
    def __init__(self, num_classes=metadata.ACTION_NUM, feature_dim=200):
        super(VGG16, self).__init__()
        self.features = torchvision.models.vgg16(pretrained=True).features
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512 * 7 * 7, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.Dropout(),
            torch.nn.Linear(4096, feature_dim)
        )
        self.last = torch.nn.Linear(feature_dim, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        output = self.last(x)
        return x, output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.zeros_(m.bias)

class ResNet152(torch.nn.Module):
    def __init__(self, num_classes=metadata.ACTION_NUM, feature_dim=200):
        super(ResNet152, self).__init__()
        self.features = torchvision.models.resnet152(pretrained=True)
        self.fc_ = torch.nn.Linear(1000, feature_dim)
        self.fc = torch.nn.Linear(feature_dim, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc_(x)
        output = self.fc(x)
        return x, output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.zeros_(m.bias)

class DenseNet(torch.nn.Module):
    def __init__(self, num_classes=metadata.ACTION_NUM, feature_dim=200):
        super(DenseNet, self).__init__()
        self.features = torchvision.models.densenet161(pretrained=True)
        self.fc_ = torch.nn.Linear(1000, feature_dim)
        self.fc = torch.nn.Linear(feature_dim, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc_(x)
        output = self.fc(x)
        return x, output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.zeros_(m.bias)

class AffordanceNet(torch.nn.Module):
    def __init__(self, num_classes, name='resnet', feature_dim=200):
        super(AffordanceNet, self).__init__()
        self.network = model_dict[name](num_classes, feature_dim)

    def forward(self, x):
        return self.network(x)

class ActivityNet(torch.nn.Module):
    def __init__(self, num_classes, name='resnet', feature_dim=500, obj_feature_dim=200):
        super(ActivityNet, self).__init__()
        self.network = model_dict[name](num_classes, 2 * feature_dim)
        self.pooling = torch.nn.MaxPool2d(3, stride=1, padding=1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2 * feature_dim + 3 * obj_feature_dim, 2 * feature_dim),
            torch.nn.BatchNorm1d(2 * feature_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * feature_dim, feature_dim)
        )
        self.fc_ = torch.nn.Linear(feature_dim, num_classes)

    def forward(self, x, affordance_features):
        feature, x = self.network(x)
        u_feature = self.pooling(affordance_features).view(affordance_features.size(0), -1)
        features = self.fc(torch.cat((feature, u_feature), 1))
        output = self.fc_(features)
        return features, output

class TaskNet(torch.nn.Module):
    def __init__(self, task='affordance', name='resnet', feature_dim=1500, obj_feature_dim=1000):
        super(TaskNet, self).__init__()
        self.task = task
        if task == 'affordance':
            self.network = AffordanceNet(num_classes=metadata.AFFORDANCE_NUM, name=name, feature_dim=obj_feature_dim)
        else:
            self.network = ActivityNet(num_classes=metadata.ACTION_NUM, name=name, feature_dim=feature_dim, obj_feature_dim=obj_feature_dim)

    def forward(self, x, features=None):
        if self.task == 'affordance':
            return self.network(x)
        else:
            return self.network(x, features)

# For test purposes only
def main():
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        normalize,
    ])

if __name__ == '__main__':
    main()