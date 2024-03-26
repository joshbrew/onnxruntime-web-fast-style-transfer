from collections import namedtuple

import torch
from torchvision import models

class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        # Load pre-trained VGG-16 model features. PyTorch models module provides a pre-trained VGG-16
        # that is trained on ImageNet dataset. Only the features portion of the model (convolutional layers)
        # is utilized for feature extraction, ignoring the classifier part.
        vgg_pretrained_features = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        
        # Initialize four sequential containers. Each will hold a 'slice' of the VGG-16 features,
        # corresponding to different depths in the network. These slices are used to obtain the output
        # from intermediate layers of the network.
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        
        # Populate the sequential containers with the VGG-16 layers.
        # The slices are defined up to specific layers to capture the output
        # after activation functions (ReLU layers), following common practice in style transfer.
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        
        # If gradients are not required (e.g., in inference mode, such as feature extraction for style transfer),
        # set requires_grad = False to avoid unnecessary computation during backpropagation.
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        # Define the forward pass. The input image X is processed through each slice sequentially.
        # At the output of each slice, the feature map is saved to a variable corresponding
        # to the depth at which features are extracted.
        h = self.slice1(X)
        h_relu1_2 = h  # Output after the ReLU activation following the second convolutional layer.
        h = self.slice2(h)
        h_relu2_2 = h  # Similar, output after the second slice.
        h = self.slice3(h)
        h_relu3_3 = h  # Output after the third slice.
        h = self.slice4(h)
        h_relu4_3 = h  # Output after the fourth slice, capturing deeper features.
        
        # A namedtuple is used for convenience to access the outputs by name.
        # This encapsulates the four feature maps obtained from the slices in an output tuple.
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        
        # The function returns a namedtuple containing feature maps from the specified layers,
        # making it easy to access them for further processing.
        return out