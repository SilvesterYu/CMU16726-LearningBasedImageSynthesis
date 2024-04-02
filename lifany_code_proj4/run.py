import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
import copy
import sys
from utils import load_image, Normalization, device, imshow, get_image_optimizer
from style_and_content import ContentLoss, StyleLoss
import torchvision.transforms as T

torch.manual_seed(36)

"""A ``Sequential`` module contains an ordered list of child modules. For
instance, ``vgg19.features`` contains a sequence (Conv2d, ReLU, MaxPool2d,
Conv2d, ReLU…) aligned in the right order of depth. We need to add our
content loss and style loss layers immediately after the convolution
layer they are detecting. To do this we must create a new ``Sequential``
module that has content loss and style loss modules correctly inserted.
"""

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_3']
# style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
# style_layers_default = ['conv_6', 'conv_7', 'conv_8', 'conv_9', 'conv_10']
# style_layers_default = ['conv_11', 'conv_12', 'conv_13', 'conv_14', 'conv_15']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5', 'conv_6', 'conv_7', 'conv_8', 'conv_9', 'conv_10']


def get_model_and_losses(cnn, style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # build a sequential model consisting of a Normalization layer
    # then all the layers of the VGG feature network along with ContentLoss and StyleLoss
    # layers in the specified places

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially

    normalization = Normalization().to(device)
    model = nn.Sequential(normalization)
    print(model)

    i = 1
    for l in cnn.children():
        print(type(l))
        if type(l) == nn.Conv2d:
            model.add_module("conv_" + str(i), l)

            # add losses immediately after conv layer
            if "conv_" + str(i) in content_layers:
                content_loss = ContentLoss(model(content_img).detach())
                model.add_module("ContentLoss_" + str(i), content_loss)
                content_losses.append(content_loss)

            if "conv_" + str(i) in style_layers:
                style_loss = StyleLoss(model(style_img).detach())
                model.add_module("StyleLoss_" + str(i), style_loss)
                style_losses.append(style_loss)

            i += 1

        # here if you need a nn.ReLU layer, make sure to use inplace=False
        # as the in place version interferes with the loss layers
        elif type(l) == nn.ReLU:
            model.add_module("ReLU_" + str(i), nn.ReLU(inplace=False))
        elif type(l) == nn.BatchNorm2d:
            model.add_module("BatchNorm2d_" + str(i), l)
        elif type(l) == nn.MaxPool2d:
            model.add_module("MaxPool2d_" + str(i), l)

    print(model)

    # trim off the layers after the last content and style losses
    # as they are vestigial
    for i in range(len(model) - 1, -1, -1):
        if type(model[i]) == StyleLoss or type(model[i]) == ContentLoss:
            print("break", i)
            break

    model = model[:i+1]
    print(model)

    return model, style_losses, content_losses


"""Finally, we must define a function that performs the neural transfer. For
each iteration of the networks, it is fed an updated input and computes
new losses. We will run the ``backward`` methods of each loss module to
dynamicaly compute their gradients. The optimizer requires a “closure”
function, which reevaluates the module and returns the loss.

We still have one final constraint to address. The network may try to
optimize the input with values that exceed the 0 to 1 tensor range for
the image. We can address this by correcting the input values to be
between 0 to 1 each time the network is run.



"""


def run_optimization(cnn, content_img, style_img, input_img, use_content=True, use_style=True, num_steps=600,
                     style_weight=1000000, content_weight=1):
    """Run the image reconstruction, texture synthesis, or style transfer."""
    print('Building the style transfer model..')

    # get your model, style, and content losses
    model, style_losses, content_losses = get_model_and_losses(cnn, style_img, content_img)
    model.requires_grad_(False)

    # get the optimizer
    input_img.requires_grad_(True)
    optimizer = get_image_optimizer(input_img)

    # run model training, with one weird caveat
    # we recommend you use LBFGS, an algorithm which preconditions the gradient
    # with an approximate Hessian taken from only gradient evaluations of the function
    # this means that the optimizer might call your function multiple times per step, so as
    # to numerically approximate the derivative of the gradients (the Hessian)
    # so you need to define a function

    step = [0]

    def losses():
        # a  correction...
        with torch.no_grad():
            input_img.clamp_(0, 1)
        optimizer.zero_grad()
        model(input_img)

        total_loss = 0
        content_loss = 0
        style_loss = 0
        if use_content:
            for l in content_losses:
                content_loss += l.loss*content_weight
            total_loss += content_loss
        if use_style:
            for l in style_losses:
                style_loss += l.loss*style_weight
            total_loss += style_loss

        total_loss.backward()
        step[0] += 1
        if step[0] % 100 == 0:
            print(step[0], num_steps, "style loss, content loss", style_loss, content_loss)
        
        return total_loss

    while step[0] < num_steps:
        optimizer.step(losses)
    

    # one more hint: the images must be in the range [0, 1]
    # but the optimizer doesn't know that
    # so you will need to clamp the img values to be in that range after every step
    # make sure to clamp once you are done
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img


def main(style_img_path, content_img_path):
    # we've loaded the images for you
    style_img = load_image(style_img_path)
    content_img = load_image(content_img_path)
    # imshow(content_img, title='Reconstructed Image')

    # interative MPL
    plt.ion()

    # -- resize style image according to content image size

    SIZE = (content_img.shape[-2], content_img.shape[-1])
    style_size = style_img.shape
    print("content size ", SIZE)
    print("style size ", style_size)
    style_img = T.Resize(size=SIZE)(style_img)
    content_img = T.Resize(size=SIZE)(content_img)
    print(style_img.size(), content_img.size())
    # --

    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

    # plot the original input image:
    # plt.figure()
    # imshow(style_img, title='Style Image')

    # plt.figure()
    # imshow(content_img, title='Content Image')

    # we load a pretrained VGG19 model from the PyTorch models library
    # but only the feature extraction part (conv layers)
    # and configure it for evaluation
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    
    '''
    # -- Part 1
    # image reconstruction
    print("Performing Image Reconstruction from white noise initialization")
    # input_img = random noise of the size of content_img on the correct device
    # output = reconstruct the image from the noise
    
    input_img = None
    input_img = torch.rand_like(content_img, device=device)
    plt.figure()
    imshow(input_img, title='Input Image')
    output = run_optimization(cnn, content_img, style_img, input_img, use_content=True, use_style=False,
                              style_weight=10000, content_weight=1)
    
    plt.figure()
    imshow(output, title='Reconstructed Image')
    
    
    
    # -- Part 2
    # texture synthesis
    print("Performing Texture Synthesis from white noise initialization")
    # input_img = random noise of the size of content_img on the correct device
    # output = synthesize a texture like style_image
    imshow(style_img, title='Synthesized Texture')
    input_img = torch.rand_like(content_img, device=device)
    output = run_optimization(cnn, content_img, style_img, input_img, use_content=False, use_style=True,
                              style_weight=10000, content_weight=1)

    plt.figure()
    
    imshow(output, title='Synthesized Texture')
    '''
    
    

    
    # -- Part 3
    # style transfer
    # input_img = random noise of the size of content_img on the correct device
    # output = transfer the style from the style_img to the content image
    plt.figure()
    imshow(content_img, title='Content')
    plt.figure()
    imshow(style_img, title='Style')
    # input_img = torch.rand_like(content_img, device=device)
    # output = run_optimization(cnn, content_img, style_img, input_img, use_content=True, use_style=True,
    #                           style_weight=10000, content_weight=1)

    # plt.figure()
    # imshow(output, title='Output Image from noise')

    print("Performing Style Transfer from content image initialization")
    input_img = content_img.clone()
    # output = transfer the style from the style_img to the content image
    output = run_optimization(cnn, content_img, style_img, input_img, use_content=True, use_style=True,
                              style_weight=10000, content_weight=1)
    

    plt.figure()
    imshow(output, title='Output Image from content_img')
    

    plt.ioff()
    plt.show()
    


if __name__ == '__main__':
    args = sys.argv[1:3]
    main(*args)
    # main("images/style/frida_kahlo.jpeg", "images/content/fallingwater.png")
