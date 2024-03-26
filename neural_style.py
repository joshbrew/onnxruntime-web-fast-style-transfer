import argparse
import os
import sys
import time
import re

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import torch.onnx

import utils
from transformer_net import TransformerNet
from vgg import Vgg16


def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def train(args):
    # Set the device (GPU or CPU) based on the user's configuration and availability.
    if args.cuda:
        device = torch.device("cuda")
    # Uncomment the following line if your hardware supports Metal Performance Shaders (MPS) for Apple devices.
    # elif args.mps:
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Seed the random number generators for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Define transformations for the input images, including resizing, cropping, conversion to tensor, and scaling pixel values.
    transform = transforms.Compose([
        transforms.Resize(args.image_size),  # Resize images to a specified size.
        transforms.CenterCrop(args.image_size),  # Crop images to a specified size from the center.
        transforms.ToTensor(),  # Convert images to PyTorch tensors.
        transforms.Lambda(lambda x: x.mul(255))  # Scale the tensor values to [0, 255].
    ])

    # Load the dataset from a folder, applying the specified transformations.
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    # Prepare a data loader for the training dataset, specifying batch size.
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    # Initialize the transformer network and move it to the specified device.
    transformer = TransformerNet().to(device)
    # Define the optimizer for training the transformer network, specifying learning rate.
    optimizer = Adam(transformer.parameters(), args.lr)
    # Use mean squared error loss for training.
    mse_loss = torch.nn.MSELoss()

    # Load a pre-trained VGG16 model for feature extraction, without training it further.
    vgg = Vgg16(requires_grad=False).to(device)
    # Define transformations for the style image.
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    # Load and transform the style image, then repeat it to match the batch size.
    style = utils.load_image(args.style_image, size=args.style_size)
    style = style_transform(style)
    style = style.repeat(args.batch_size, 1, 1, 1).to(device)

    # Extract style features using VGG16 and compute the Gram matrix for each feature map.
    features_style = vgg(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(y) for y in features_style]

    # Training loop: iterate over epochs.
    for e in range(args.epochs):
        transformer.train()  # Set the model to training mode.
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        # Iterate over the training data loader.
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()  # Clear previous gradients.

            x = x.to(device)  # Move input images to the specified device.
            y = transformer(x)  # Forward pass through the transformer network.

            # Normalize the batch of images for both input and output.
            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)

            # Extract features from both the original and transformed images using VGG16.
            features_y = vgg(y)
            features_x = vgg(x)

            # Compute content loss as MSE between feature representations of input and output images.
            content_loss = args.content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            # Initialize style loss.
            style_loss = 0.
            # Compute style loss as MSE between Gram matrices of style and output images.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= args.style_weight

            # Compute total loss, backpropagate gradients, and update the model parameters.
            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            # Aggregate content and style losses for reporting.
            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            # Print training progress periodically.
            if (batch_id + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)

            # Save checkpoint models periodically.
        if args.checkpoint_model_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
            transformer.eval().cpu()  # Set the model to evaluation mode and move to CPU for saving.
            ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
            ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
            torch.save(transformer.state_dict(), ckpt_model_path)  # Save the current state of the model.
            transformer.to(device).train()  # Move the model back to the specified device and set to training mode.

    # After training, save the final model.
    transformer.eval().cpu()  # Set the model to evaluation mode and move to CPU for saving.
    save_model_filename = "model.pth"  # Filename for the saved model.
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)  # Path to save the model.
    torch.save(transformer.state_dict(), save_model_path)  # Save the model's state.


    # Save the model in ONNX format with dynamic input resolution
    #dummy_input = torch.randn(1, 3, args.image_size, args.image_size)  # Initial size, but will allow for dynamic sizes
    #onnx_model_filename = args.model or "model.onnx"
    #onnx_model_path = os.path.join(args.save_model_dir, onnx_model_filename)

    # Specify dynamic_axes where the 'height' and 'width' dimensions can vary
    # This indicates to ONNX that these dimensions can be dynamic
    #torch.onnx.export(transformer, dummy_input, onnx_model_path, 
    #                export_params=True, opset_version=12, 
    #                do_constant_folding=True, 
    #                input_names=['input'], output_names=['output'],
    #                dynamic_axes={'input' : {0 : 'batch_size', 2 : 'height', 3 : 'width'}, 
    #                                'output': {0 : 'batch_size', 2 : 'height', 3 : 'width'}})

    print("\nDone, trained model saved at", save_model_path)


def stylize(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    content_image = utils.load_image(args.content_image, scale=args.content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    if args.model.endswith(".onnx"):
        output = stylize_onnx(content_image, args)
    else:
        with torch.no_grad():
            style_model = TransformerNet()
            state_dict = torch.load(args.model)
            # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
            for k in list(state_dict.keys()):
                if re.search(r'in\d+\.running_(mean|var)$', k):
                    del state_dict[k]
            style_model.load_state_dict(state_dict)
            style_model.to(device)
            style_model.eval()
            if args.export_onnx:
                assert args.export_onnx.endswith(".onnx"), "Export model file should end with .onnx"
                output = torch.onnx._export(
                    style_model, content_image, args.export_onnx, opset_version=11,
                ).cpu()            
            else:
                output = style_model(content_image).cpu()
    utils.save_image(args.output_image, output[0])


def stylize_onnx(content_image, args):
    """
    Read ONNX model and run it using onnxruntime
    """

    assert not args.export_onnx

    import onnxruntime

    ort_session = onnxruntime.InferenceSession(args.model)

    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(content_image)}
    ort_outs = ort_session.run(None, ort_inputs)
    img_out_y = ort_outs[0]

    return torch.from_numpy(img_out_y)


def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=2,
                                  help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--batch-size", type=int, default=4,
                                  help="batch size for training, default is 4")
    train_arg_parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--style-image", type=str, default="images/style-images/mosaic.jpg",
                                  help="path to style-image")
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--checkpoint-model-dir", type=str, default=None,
                                  help="path to folder where checkpoints of trained models will be saved")
    train_arg_parser.add_argument("--image-size", type=int, default=256,
                                  help="size of training images, default is 256 X 256")
    train_arg_parser.add_argument("--style-size", type=int, default=None,
                                  help="size of style-image, default is the original size of style image")
    train_arg_parser.add_argument("--cuda", type=int, required=True,
                                  help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")
    train_arg_parser.add_argument("--content-weight", type=float, default=1e5,
                                  help="weight for content-loss, default is 1e5")
    train_arg_parser.add_argument("--style-weight", type=float, default=1e10,
                                  help="weight for style-loss, default is 1e10")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 1e-3")
    train_arg_parser.add_argument("--log-interval", type=int, default=500,
                                  help="number of images after which the training loss is logged, default is 500")
    train_arg_parser.add_argument("--checkpoint-interval", type=int, default=2000,
                                  help="number of batches after which a checkpoint of the trained model will be created")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--content-image", type=str, required=True,
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--output-image", type=str, required=True,
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image. If file ends in .pth - PyTorch path is used, if in .onnx - Caffe2 path")
    eval_arg_parser.add_argument("--cuda", type=int, default=False,
                                 help="set it to 1 for running on cuda, 0 for CPU")
    eval_arg_parser.add_argument("--export_onnx", type=str,
                                 help="export ONNX model to a given file")
    eval_arg_parser.add_argument('--mps', action='store_true', default=False, help='enable macOS GPU training')

    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)
    #if not args.mps and torch.backends.mps.is_available():
    #    print("WARNING: mps is available, run with --mps to enable macOS GPU")

    if args.subcommand == "train":
        check_paths(args)
        train(args)
    else:
        stylize(args)


if __name__ == "__main__":
    main()
 