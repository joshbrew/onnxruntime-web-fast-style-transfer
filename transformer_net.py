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
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)

        # Upsampling Layers
        # These layers increase the size of the image back to its original dimensions,
        # while reducing the number of channels back to 3 (RGB).
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)  # Final layer to produce the output image.

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
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
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
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        # Reflection padding is used to pad the image with a mirror image of its border. 
        # This helps in dealing with the issue where convolution reduces the size of the image.
        # It also helps in reducing artifacts at the image edges, making the transition at the border smoother.
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        # The forward pass applies the reflection padding followed by the convolution operation.
        # This process extracts significant features from the input image, preparing it for further processing in the network.
        out = self.reflection_pad(x)  # Apply reflection padding to maintain image size after convolution.
        out = self.conv2d(out)  # Perform the convolution operation.
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
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)  # Instance normalization helps in stabilizing the learning in style transfer tasks.
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()  # ReLU introduces non-linearity, allowing the network to learn complex patterns.

    def forward(self, x):
        # The input is saved to a variable 'residual'. 
        # After processing through convolutional layers, the residual (input) is added back to the output, forming the final output of the block.
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual  # Add the input directly to the output.
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/

    This layer first upsamples the input image to a larger size and then applies a convolution.
    Upsampling is a technique to increase the spatial dimensions (width and height) of the image.
    This is particularly useful in tasks like style transfer, where we need to increase the resolution of the transformed image.

    Unlike ConvTranspose2d, which directly combines upsampling and convolution and can introduce checkerboard artifacts,
    this approach separately upsamples using nearest neighbor interpolation, followed by a standard convolution,
    reducing the risk of such artifacts.

    Parameters:
    - in_channels: Number of input channels.
    - out_channels: Number of output channels.
    - kernel_size: Size of the convolutional kernel.
    - stride: Stride for the convolution.
    - upsample: Factor by which to upsample the input.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample  # The factor by which the input is to be upscaled. If None, no upsampling is performed.
        reflection_padding = kernel_size // 2
        # Uses reflection padding prior to convolution to maintain spatial dimensions and reduce edge artifacts.
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        # A standard convolutional layer that will process the upsampled image.

    def forward(self, x):
        x_in = x
        if self.upsample:
            # If an upsample factor is specified, the input is first upscaled using nearest neighbor interpolation.
            # Nearest neighbor interpolation is chosen for its simplicity and effectiveness in many scenarios.
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)  # Apply reflection padding to the upsampled input.
        out = self.conv2d(out)  # Perform convolution on the padded, upsampled input.
        return out
