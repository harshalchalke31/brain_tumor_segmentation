from torchvision import models
import torch.nn as nn
import torch

class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNet50Classifier, self).__init__()
        # Load the pretrained ResNet50 backbone
        self.model = models.resnet50(pretrained=True)  #changed
        num_features = self.model.fc.in_features        #changed
        # Replace the final fully connected layer with one that outputs the desired number of classes
        self.model.fc = nn.Linear(num_features, num_classes)  #changed

    def forward(self, x):
        return self.model(x)