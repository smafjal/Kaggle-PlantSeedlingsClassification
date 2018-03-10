import numpy as np
import torchvision
import torch
import logging
import torch.nn as nn
from torch.autograd import Variable

logger = logging.getLogger('model-resnet-preconvfeat')
logger.setLevel(logging.INFO)

USE_GPU = torch.cuda.is_available()
VERBOSE = 5


class MyResnet(nn.Module):
    def __init__(self, num_class=1000, pretrained=True):
        super(MyResnet, self).__init__()
        self.num_class = num_class
        
        resnet18 = torchvision.models.resnet18(pretrained=pretrained)
        for param in resnet18.parameters():
            param.requires_grad = False
        
        self.features = nn.Sequential(*list(resnet18.children())[:-1])
        self.trained_features = nn.Linear(512, num_class)
    
    def forward(self, inputs):
        inp = inputs.view(inputs.size(0), -1)
        out = self.trained_features(inp)
        return out


def preconvfeat(dataloaders, image_datasets, model, phase):
    conv_fet = []
    labels_list = []
    
    logger.info("ConvNet fetures generating phase: %s" % phase)
    batch_epoch = 0
    for data in dataloaders[phase]:
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        
        w = model.features(inputs)
        conv_fet.extend(w.data.numpy())
        labels_list.extend(labels.data.numpy())
        batch_epoch += 1
        if batch_epoch % VERBOSE == 0:
            logger.info("Gen CovNet Feture! {} Epoch {}/{}".format(
                phase, batch_epoch, len(image_datasets[phase]) // inputs.size(0)))
    
    conv_fet = np.concatenate([[feat] for feat in conv_fet])
    return conv_fet, labels_list
