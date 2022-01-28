import torch 
import torch.nn as nn
import torchvision.models as models 


class CalibNet(nn.Module):

    def __init__(self, pretrained = True, hidden_size = 128, outputs = 2, loss = nn.SmoothL1Loss()):
        super(CalibNet, self).__init__()

        self.resnet = models.resnet50(pretrained=pretrained)
        fc_inputs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
                            nn.Linear(fc_inputs, hidden_size),
                            nn.ReLU(),
                            nn.Linear(hidden_size, 2) 
                        )
        self.loss = loss 


    def forward(self, x):
        return self.resnet(x)