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
        self.deconv1 = ConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = ConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)

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



# Architecture differences based on: https://blog.unity.com/engine-platform/real-time-style-transfer-in-unity-using-deep-neural-networks
class SmallTransformerNet(torch.nn.Module):
    def __init__(self):
        super(SmallTransformerNet, self).__init__()
        # Initial convolution layers with less complexity and efficient design
        self.conv1 = ConvLayer(3, 16, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(16, affine=True)
        self.conv2 = ConvLayer(16, 32, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(32, affine=True)

        # A single, more efficient residual block
        self.resblocks = torch.nn.Sequential(*[OptimizedResidualBlock(32) for _ in range(5)])

        # Upsampling Layers with efficient techniques
        self.deconv1 = ConvLayer(32, 16, kernel_size=3, stride=1, upsample=4)
        self.in3 = torch.nn.InstanceNorm2d(16, affine=True)
        self.deconv2 = ConvLayer(16, 3, kernel_size=9, stride=1)
       
        # Non-linearities
        # ReLU (Rectified Linear Unit) introduces non-linearity, helping the network learn complex patterns.
        self.relu = torch.nn.ReLU()


    def forward(self, X):
        # Define the forward pass through the network.
        
        # Applies convolutional layers, residual blocks, and upsampling layers in sequence.
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        
        # Residual blocks do not change the dimensions of their input.
        y = self.resblocks(y)
        
        # Upsampling layers increase the spatial dimensions of the input.
        y = self.relu(self.in3(self.deconv1(y)))
        y = self.deconv2(y)  # Final convolution to produce the output.
        return y
    

# Architecture differences based on: https://blog.unity.com/engine-platform/real-time-style-transfer-in-unity-using-deep-neural-networks
class SmallTransformerNet48(torch.nn.Module):
    def __init__(self):
        super(SmallTransformerNet48, self).__init__()
        # Initial convolution layers with less complexity and efficient design
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 48, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(48, affine=True)

        # A single, more efficient residual block
        self.resblocks = torch.nn.Sequential(*[OptimizedResidualBlock(48) for _ in range(5)])

        # Upsampling Layers with efficient techniques
        self.deconv1 = ConvLayer(48, 32, kernel_size=3, stride=1, upsample=4)
        self.in3 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv2 = ConvLayer(32, 3, kernel_size=9, stride=1)
       
        # Non-linearities
        # ReLU (Rectified Linear Unit) introduces non-linearity, helping the network learn complex patterns.
        self.relu = torch.nn.ReLU()

        

    def forward(self, X):
        # Define the forward pass through the network.
        
        # Applies convolutional layers, residual blocks, and upsampling layers in sequence.
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        
        # Residual blocks do not change the dimensions of their input.
        y = self.resblocks(y)
        
        # Upsampling layers increase the spatial dimensions of the input.
        y = self.relu(self.in3(self.deconv1(y)))
        y = self.deconv2(y)  # Final convolution to produce the output.
        return y
   
# Architecture based on: https://medium.com/@jamesonthecrow/creating-a-17kb-style-transfer-model-with-layer-pruning-and-quantization-864d7cc53693
class MobileTransformerNet(torch.nn.Module):
    def __init__(self):
        super(MobileTransformerNet, self).__init__()
        # Initial convolution layers with less complexity and efficient design
        self.conv1 = ConvLayer(3, 9, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(9, affine=True)
        self.conv2 = ConvLayer(9, 9, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(9, affine=True)
        self.conv3 = ConvLayer(9, 9, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(9, affine=True)

        # A single, more efficient residual block
        self.resblocks = torch.nn.Sequential(*[OptimizedResidualBlock(9) for _ in range(3)])

        # Upsampling Layers with efficient techniques
        self.deconv1 = ConvLayer(9, 9, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(9, affine=True)
        self.deconv2 = ConvLayer(9, 9, kernel_size=9, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(9, affine=True)
        self.deconv3 = ConvLayer(9, 3, kernel_size=3, stride=1)
       
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
   

# has some issue allocating operators for efficiency in onnx we found so it runs worse but is more compressed
class EfficientTransformerNet(torch.nn.Module):
    def __init__(self):
        super(EfficientTransformerNet, self).__init__()
        # Initial convolution layers
        # ConvLayer is a custom convolutional layer defined later in the code.
        # It applies padding before convolution to maintain the image size.
        # These layers progressively increase the number of channels while reducing image size,
        # preparing the input for the residual blocks.
        self.conv1 = DepthwiseConvLayer(3, 32, kernel_size=9, stride=1, padding=0)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)  # Normalizes the output of conv1 to stabilize training.
        self.conv2 = DepthwiseConvLayer(32, 64, kernel_size=3, stride=2, padding=0)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)  # Each InstanceNorm layer normalizes across spatial dimensions.
        self.conv3 = DepthwiseConvLayer(64, 128, kernel_size=3, stride=2, padding=0)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)

        # Residual layers
        # These layers allow the network to learn identity functions, making training of deep networks easier.
        # They help in propagating gradients through multiple layers without diminishing them.

        self.resblocks = torch.nn.Sequential(*[OptimizedResidualBlock(128) for _ in range(5)])

        # Upsampling Layers
        # These layers increase the size of the image back to its original dimensions,
        # while reducing the number of channels back to 3 (RGB).
        self.deconv1 = DepthwiseConvLayer(128, 64, kernel_size=3, stride=1, upsample=2, padding=0)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = DepthwiseConvLayer(64, 32, kernel_size=3, stride=1, upsample=2, padding=0)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = DepthwiseConvLayer(32, 3, kernel_size=9, stride=1, padding=0)

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


# Architecture differences based on: https://blog.unity.com/engine-platform/real-time-style-transfer-in-unity-using-deep-neural-networks
class SmallEfficientTransformerNet(torch.nn.Module):
    def __init__(self):
        super(SmallEfficientTransformerNet, self).__init__()
        # Initial convolution layers with less complexity and efficient design
        self.conv1 = DepthwiseConvLayer(3, 16, kernel_size=9, stride=1, padding=0)
        self.in1 = torch.nn.InstanceNorm2d(16, affine=True)
        self.conv2 = DepthwiseConvLayer(16, 32, kernel_size=3, stride=2, padding=0)
        self.in2 = torch.nn.InstanceNorm2d(32, affine=True)

        # A single, more efficient residual block
        self.resblocks = torch.nn.Sequential(*[OptimizedResidualBlock(32) for _ in range(5)])

        # Upsampling Layers with efficient techniques
        self.deconv1 = DepthwiseConvLayer(32, 16, kernel_size=3, stride=1, upsample=4, padding=0) #OptimizedUpsampleConvLayer # test next
        self.in5 = torch.nn.InstanceNorm2d(16, affine=True)
        self.deconv2 = DepthwiseConvLayer(16, 3, kernel_size=9, stride=1, padding=0)
       
        # Non-linearities
        # ReLU (Rectified Linear Unit) introduces non-linearity, helping the network learn complex patterns.
        self.relu = torch.nn.ReLU()

        

    def forward(self, X):
        # Define the forward pass through the network.
        # Applies convolutional layers, residual blocks, and upsampling layers in sequence.
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        # Residual blocks do not change the dimensions of their input.
        y = self.resblocks(y)
        # Upsampling layers increase the spatial dimensions of the input.
        y = self.relu(self.in5(self.deconv1(y)))
        y = self.deconv2(y)  # Final convolution to produce the output.
        return y
    
# Results weren't even a 2X speedup over the bigger model, we need to go deeper!
# Architecture differences based on: https://blog.unity.com/engine-platform/real-time-style-transfer-in-unity-using-deep-neural-networks
class SmallEfficientTransformerNet48(torch.nn.Module):
    def __init__(self):
        super(SmallEfficientTransformerNet48, self).__init__()
        # Initial convolution layers with less complexity and efficient design
        self.conv1 = DepthwiseConvLayer(3, 32, kernel_size=9, stride=1, padding=0)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = DepthwiseConvLayer(32, 48, kernel_size=3, stride=2, padding=0)
        self.in2 = torch.nn.InstanceNorm2d(48, affine=True)

        # A single, more efficient residual block
        self.resblocks = torch.nn.Sequential(*[OptimizedResidualBlock(48) for _ in range(5)])

        # Upsampling Layers with efficient techniques
        self.deconv1 = DepthwiseConvLayer(48, 32, kernel_size=3, stride=1, upsample=4, padding=0) #OptimizedUpsampleConvLayer # test next
        self.in3 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv2 = DepthwiseConvLayer(32, 3, kernel_size=9, stride=1, padding=0)
       
        # Non-linearities
        # ReLU (Rectified Linear Unit) introduces non-linearity, helping the network learn complex patterns.
        self.relu = torch.nn.ReLU()

        

    def forward(self, X):
        # Define the forward pass through the network.
        # Applies convolutional layers, residual blocks, and upsampling layers in sequence.
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        # Residual blocks do not change the dimensions of their input.
        y = self.resblocks(y)
        # Upsampling layers increase the spatial dimensions of the input.
        y = self.relu(self.in3(self.deconv1(y)))
        y = self.deconv2(y)  # Final convolution to produce the output.
        return y
    
   
# Architecture based on: https://medium.com/@jamesonthecrow/creating-a-17kb-style-transfer-model-with-layer-pruning-and-quantization-864d7cc53693
class EfficientMobileTransformerNet(torch.nn.Module):
    def __init__(self):
        super(EfficientMobileTransformerNet, self).__init__()
        # Initial convolution layers with less complexity and efficient design
        self.conv1 = DepthwiseConvLayer(3, 9, kernel_size=9, stride=1, padding=0)
        self.in1 = torch.nn.InstanceNorm2d(9, affine=True)
        self.conv2 = DepthwiseConvLayer(9, 9, kernel_size=3, stride=2, padding=0)
        self.in2 = torch.nn.InstanceNorm2d(9, affine=True)
        self.conv3 = DepthwiseConvLayer(9, 9, kernel_size=3, stride=2, padding=0)
        self.in3 = torch.nn.InstanceNorm2d(9, affine=True)

        # A single, more efficient residual block
        self.resblocks = torch.nn.Sequential(*[OptimizedResidualBlock(9) for _ in range(3)])

        # Upsampling Layers with efficient techniques
        self.deconv1 = DepthwiseConvLayer(9, 9, kernel_size=3, stride=1, upsample=2, padding=0) #upsample=4
        self.in4 = torch.nn.InstanceNorm2d(9, affine=True)
        self.deconv2 = DepthwiseConvLayer(9, 9, kernel_size=9, stride=1, upsample=2, padding=0) #OptimizedUpsampleConvLayer
        self.in5 = torch.nn.InstanceNorm2d(9, affine=True)
        self.deconv3 = DepthwiseConvLayer(9, 3, kernel_size=3, stride=1, padding=0)
       
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

class DepthwiseSeparableConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=None):
        super(DepthwiseSeparableConv, self).__init__()

        self.depthwise = torch.nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, groups=in_channels, padding=padding or kernel_size//2)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
class DepthwiseConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None, padding=None):
        super(DepthwiseConvLayer, self).__init__()
        self.upsample = upsample

        if padding == None:
            # adjust padding for odd channel counts 
            if(int(in_channels/2) != in_channels/2 and int(in_channels/3) == in_channels/3):
                padding = (kernel_size - 1) // 2 
            else:
                padding = kernel_size // 2
        
        self.padding = torch.nn.ReflectionPad2d(padding)

        # Use a depthwise separable convolution to process input
        self.depthwise_separable_conv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size, stride, padding=0)  # No additional padding applied here, using reflection padding instead.

    def forward(self, x):
        # Apply upsampling first if specified
        if self.upsample:
            x = torch.nn.functional.interpolate(x, scale_factor=self.upsample, mode='nearest')
        
        # Apply reflection padding to maintain spatial dimensions without introducing border artifacts
        x = self.padding(x)

        # Perform the depthwise separable convolution
        x = self.depthwise_separable_conv(x)
        return x

class GroupedConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None, groups=None):
        super(GroupedConvLayer, self).__init__()
        self.upsample = upsample
 
        if groups == None:
            if(int(in_channels/3) == in_channels/3):
                groups = 3
            else:
                groups = 2 if in_channels <= 32 else 4

        self.padding = torch.nn.ReflectionPad2d(kernel_size // 2)
        
        # Ensure the number of groups is a divisor of both in_channels and out_channels
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError("in_channels and out_channels must be divisible by groups")
        
        # Grouped convolution
        self.group_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, groups=groups)
        
        # Pointwise convolution to mix channels from different groups
        self.pointwise_conv = torch.nn.Conv2d(out_channels, out_channels, 1, 1, groups=groups)  # Keeping groups in pointwise_conv allows for further efficiency
        
    def forward(self, x):
        if self.upsample:
            x = torch.nn.functional.interpolate(x, scale_factor=self.upsample, mode='nearest', align_corners=None)
        
        x = self.padding(x)
        x = self.group_conv(x)
        x = self.pointwise_conv(x)
        return x


class OptimizedResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(OptimizedResidualBlock, self).__init__()
        self.conv1 = DepthwiseSeparableConv(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = DepthwiseSeparableConv(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        return self.relu(out + residual)

class OptimizedUpsampleConvLayer(torch.nn.Module):
    """
    An optimized upsampling layer that uses Pixel Shuffle for efficient upsampling,
    combined with a convolutional layer to refine the upsampled output. Reflection padding
    is applied before convolution to maintain spatial dimensions without introducing border artifacts.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, upsample=None):
        super(OptimizedUpsampleConvLayer, self).__init__()
        self.upsample = upsample

        if upsample:
            # Correct calculation for the channels needed for pixel shuffle
            shuffle_in_channels = in_channels * (upsample ** 2)

            # Expanding channels to prepare for pixel shuffle
            self.expand_conv = torch.nn.Conv2d(in_channels, shuffle_in_channels, kernel_size=1, stride=1)

            # Pixel shuffle for actual upsampling
            self.pixel_shuffle = torch.nn.PixelShuffle(upsample)

            # Ensuring the kernel size and padding are set to maintain dimensions after pixel shuffle
            # Note: If using a kernel size that doesn't maintain dimensions by default, adjust padding accordingly.
            self.refinement_conv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size, stride)
        else:
            # Direct depthwise separable convolution without upsampling
            self.refinement_conv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = self.expand_conv(x)
            x = self.pixel_shuffle(x)
            x = self.refinement_conv(x)
        else:
            x = self.refinement_conv(x)

        return x



import torch.nn.init as init

# This is an alternative to the inbuilt super resolution on the transformer network to try to speed it up
class SuperResolutionNet(torch.nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(SuperResolutionNet, self).__init__()

        self.relu = torch.nn.ReLU(inplace=inplace)
        self.conv1 = torch.nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = torch.nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = torch.nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = torch.nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = torch.nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)
