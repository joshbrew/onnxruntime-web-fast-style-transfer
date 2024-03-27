import torch

class TransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # Initial convolution layers
        # ConvLayer is a custom convolutional layer defined later in the code.
        # It applies padding before convolution to maintain the image size.
        # These layers progressively increase the number of channels while reducing image size,
        # preparing the input for the residual blocks.
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)  # Normalizes the output of conv1 to stabilize training.
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)  # Each InstanceNorm layer normalizes across spatial dimensions.
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)

        # Residual layers
        # These layers allow the network to learn identity functions, making training of deep networks easier.
        # They help in propagating gradients through multiple layers without diminishing them.

        self.resblocks = torch.nn.Sequential(*[ResidualBlock(128) for _ in range(5)])

        # Upsampling Layers
        # These layers increase the size of the image back to its original dimensions,
        # while reducing the number of channels back to 3 (RGB).
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = UpsampleConvLayer(32, 3, kernel_size=9, stride=1)

        # Non-linearities
        # ReLU (Rectified Linear Unit) introduces non-linearity, helping the network learn complex patterns.
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        # Define the forward pass through the network.
        # Applies convolutional layers, residual blocks, and upsampling layers in sequence.
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        # Residual blocks do not change the dimensions of their input.
        y = self.resblocks(y)
        # Upsampling layers increase the spatial dimensions of the input.
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)  # Final convolution to produce the output.
        return y


# Results weren't even a 2X speedup over the bigger model, we need to go deeper!
class SmallTransformerNet(torch.nn.Module):
    def __init__(self):
        super(SmallTransformerNet, self).__init__()
        # Initial convolution layers with less complexity and efficient design
        self.conv1 = ConvLayer(3, 16, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(16, affine=True)
        self.conv2 = ConvLayer(16, 32, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv3 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(64, affine=True)

        # A single, more efficient residual block
        self.resblock = ResidualBlock(64)

        # Upsampling Layers with efficient techniques
        self.deconv1 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv2 = UpsampleConvLayer(32, 16, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(16, affine=True)
        self.deconv3 = ConvLayer(16, 3, kernel_size=9, stride=1)
       
        # Non-linearities
        # ReLU (Rectified Linear Unit) introduces non-linearity, helping the network learn complex patterns.
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        # Define the forward pass through the network.
        # Applies convolutional layers, residual blocks, and upsampling layers in sequence.
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        # Residual blocks do not change the dimensions of their input.
        y = self.resblock(y)
        # Upsampling layers increase the spatial dimensions of the input.
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)  # Final convolution to produce the output.
        return y
    
class ConvLayer(torch.nn.Module):
    """
    This layer is a building block of the neural network, specifically designed for processing images. 
    It applies a convolution operation, which is fundamental in learning features from images. 
    Convolution applies a filter (or kernel) to an image to create a feature map, highlighting certain aspects like edges, textures, or patterns.

    Parameters:
    - in_channels: The number of channels in the input image (e.g., 3 for RGB images).
    - out_channels: The number of filters to apply, which becomes the number of channels in the output feature map.
    - kernel_size: The size of the filter (e.g., 9x9 pixels). Larger kernels cover more pixels in one operation, capturing larger features but at a computational cost.
    - stride: The step size the filter moves across the image. A stride of 1 means moving the filter one pixel at a time, leading to high-resolution feature maps.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(ConvLayer, self).__init__()
        self.upsample = upsample
        padding = kernel_size // 2  # Calculate padding based on the kernel size.
        self.reflection_pad = torch.nn.ReflectionPad2d(padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.interpolate = torch.nn.functional.interpolate
        
    def forward(self, x):
        if self.upsample:
            x = self.interpolate(x, scale_factor=self.upsample, mode='nearest')
        
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html

    Residual Blocks are designed to allow for deeper neural networks by alleviating the vanishing gradient problem.
    It does this by adding the input (or a 'shortcut connection') to the output of the block, helping to preserve the gradient signal through the network.
    This concept was introduced by He et al. in their paper on Residual Networks (ResNets).

    Each ResidualBlock consists of two convolutional layers with normalization and non-linearity (ReLU).

    Parameters:
    - channels: Number of channels in the input and output. This is fixed to ensure the input can be added to the output.
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        return self.relu(out + residual)

class UpsampleConvLayer(torch.nn.Module):
    """
    An efficient upsampling layer that combines nearest-neighbor upsampling and a convolutional layer.
    This approach can reduce checkerboard artifacts compared to transposed convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample, mode='nearest')
        self.reflection_pad = torch.nn.ReflectionPad2d(kernel_size // 2)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = self.upsample_layer(x)
        x = self.reflection_pad(x)
        x = self.conv2d(x)
        return x

