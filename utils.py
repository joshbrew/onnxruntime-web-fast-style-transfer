import torch
from PIL import Image


def load_image(filename, size=None, scale=None):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def gram_matrix(y):
    (b, ch, h, w) = y.size()  # Unpack the dimensions of the input feature map 'y'.
    # 'b' is the batch size, 'ch' is the number of channels, 'h' and 'w' are the height and width of the feature map.
    
    features = y.view(b, ch, w * h)  # Reshape 'y' to flatten the spatial dimensions.
    # This results in a matrix where each row represents a channel's features as a long vector.

    features_t = features.transpose(1, 2)  # Transpose the feature matrix to prepare for matrix multiplication.
    # This switches the channel and the flattened spatial dimensions, making the matrix multiplication feasible for computing the gram matrix.
    
    gram = features.bmm(features_t) / (ch * h * w)  # Perform batch matrix multiplication and normalize.
    # 'bmm' is batch matrix multiplication, which computes the gram matrix for each item in the batch.
    # The normalization factor 'ch * h * w' scales the gram matrix values, typically to reduce dependency on the feature map's size.
    
    return gram  # Return the computed gram matrix for each item in the batch


def normalize_batch(batch):
    # Normalize using imagenet mean and std
    # These specific mean and standard deviation values are from the ImageNet dataset,
    # which is a common practice when working with models pre-trained on ImageNet.
    
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)  # Create a tensor for the mean.
    # The '.view(-1, 1, 1)' reshapes the tensor to be compatible with the batch's dimensions, enabling broadcast during subtraction.
    
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)  # Similarly, create a tensor for the standard deviation.
    
    batch = batch.div_(255.0)  # Divide the batch by 255 to scale pixel values to the [0, 1] range.
    
    return (batch - mean) / std  # Normalize the batch by subtracting the mean and dividing by the standard deviation.