# --------------------------------------------------------
# Written by Yufei Ye and modified by Sheng-Yu Wang (https://github.com/JudyYe)
# Convert from MATLAB code https://inst.eecs.berkeley.edu/~cs194-26/fa18/hw/proj3/gradient_starter.zip
# --------------------------------------------------------
from __future__ import print_function

import argparse
import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt

import scipy
import copy

# -- Part 1.1
# keep a matrix “im2var” that maps each pixel to a variable number, such as:
def image2var(image):
    try:
        imh, imw = image.shape
    except:
        imh, imw, c = image.shape
    im2var = np.arange(imh * imw).reshape((imh, imw)).astype(int)
    return im2var

# convert a solved result variable back to image shape
def var2image(var, h, w):
    img = var.reshape((h, w)).astype(int)
    return img

def toy_recon(image):
    # every pixel turned into a variable to solve
    # the location of a pixel in image is now its variable number v_n
    # to be solved by Av = b
    im2var = image2var(image)
    print("variables:", im2var)

    h, w = image.shape[0], image.shape[1]
    num_vars = h * w # total num of pixels, each treated as a variable
    print("image h, image w:", h, w)

    # (x diretion + y direction) + top left corner
    A = np.zeros((num_vars * 2 + 1, num_vars))
    b = np.zeros((num_vars * 2 + 1))
    
    num_eqs = 0
    for i in range(h):
        # row-wise x-gradients
        for j in range(w - 1):
            v_n = im2var[i][j]
            v_next = im2var[i][j + 1]
            b[num_eqs] = image[i][j + 1] - image[i][j]
            A[num_eqs][v_n], A[num_eqs][v_next] = -1, 1
            num_eqs += 1

    for j in range(w):
        # column-wise y-gradients
        for i in range(h - 1):
            v_n = im2var[i][j]
            v_next = im2var[i + 1][j]
            b[num_eqs] = image[i + 1][j] - image[i][j]
            A[num_eqs][v_n], A[num_eqs][v_next] = -1, 1
            num_eqs += 1

    # top left pixel values should be the same
    v_0 = im2var[0][0]
    A[-1][v_0] = 1
    b[-1] = image[0][0]

    # lsq = np.linalg.lstsq(A, b, rcond=None) # Hmmmm too slow
    # print(lsq)

    # scipy is much faster
    sA = scipy.sparse.csr_matrix(A)
    lsq = scipy.sparse.linalg.lsqr(sA, b)
    print("least squares soolution", lsq)
    lsq = lsq[0] * 255
    res = var2image(lsq, h, w)
    return res

# -- Part 1.2
def poisson_blend(fg, mask, bg):
    """
    Poisson Blending.
    :param fg: (H, W, C) source texture / foreground object
    :param mask: (H, W, 1)
    :param bg: (H, W, C) target image / background
    :return: (H, W, C)
    """
    bg = bg[:, :, :3]
    im2var = image2var(fg)
    print("variables:", im2var)

    h, w, c = fg.shape[0], fg.shape[1], fg.shape[2]
    num_vars = h * w # total num of pixels, each treated as a variable
    print("image h, image w: image channels:", h, w, c)

    # for every pixel in the target image, we at most need to solve one equation
    # therefore we have num_vars many rows and columns in A
    # not using numpy because a numpy matrix takes up too much memory
    # A = np.zeros((num_vars, num_vars))
    # because we are using the same mask for all 3 channels
    # A is the same,just need to solve Av = b for 3 different b's
    A = scipy.sparse.lil_matrix((num_vars, num_vars))
    res = np.zeros((h, w, 3), dtype=int)

    ymask, xmask, _ = np.where(mask == 1)
    print("array of y and x indices in the mask", ymask, xmask)

    # solve Ax = b_ci for each channel
    for ci in range(c):
        b = np.zeros((num_vars, 1))
        for i in range(len(ymask)):   
            y, x = ymask[i], xmask[i]
            num_eq = (y - 1) * w + x
            center = im2var[y][x]

            ### fill matrix A and vector b
            # when we take the derivative, the center is added 4 times
            A[num_eq, center] = 4
            b[num_eq] += 4 * fg[y][x][ci]

            # top
            if y - 1 >= 0:
                yn, xn = y - 1, x
                # print(y, x, yn, xn)
                b[num_eq] -= fg[yn][xn][ci]
                if mask[yn][xn]:
                    nb = im2var[yn][xn]
                    if ci == 0:
                        A[num_eq, nb] = -1
                else:
                    b[num_eq] += bg[yn][xn][ci]

            # bottom
            if y + 1 <= h - 1:
                yn, xn = y + 1, x
                # print(y, x, yn, xn)
                b[num_eq] -= fg[yn][xn][ci]
                if mask[yn][xn]:
                    nb = im2var[yn][xn]
                    if ci == 0:
                        A[num_eq, nb] = -1
                else:
                    b[num_eq] += bg[yn][xn][ci]

            # left
            if x - 1 >= 0:
                yn, xn = y, x - 1
                # print(y, x, yn, xn)
                b[num_eq] -= fg[yn][xn][ci]
                if mask[yn][xn]:
                    nb = im2var[yn][xn]
                    if ci == 0:
                        A[num_eq, nb] = -1
                else:
                    b[num_eq] += bg[yn][xn][ci]

            # right
            if x + 1 <= w - 1: 
                yn, xn = y, x + 1
                # print(y, x, yn, xn)
                b[num_eq] -= fg[yn][xn][ci]
                if mask[yn][xn]:
                    nb = im2var[yn][xn]
                    if ci == 0:
                        A[num_eq, nb] = -1
                else:
                    b[num_eq] += bg[yn][xn][ci]
                
        lsq = scipy.sparse.linalg.lsqr(A.tocsr(), b)
        lsq = lsq[0] * 255
        res[:, :, ci] = var2image(lsq, h, w)
    print(res.shape, mask.shape, bg.shape)
    return (res / 255) * mask + bg * (1 - mask)
    # return fg * mask + bg * (1 - mask)

# -- Extra credit
def mixed_blend(fg, mask, bg):
    """EC: Mix gradient of source and target"""

    im2var = image2var(fg)
    h, w, c = fg.shape[0], fg.shape[1], fg.shape[2]
    num_vars = h * w # total num of pixels, each treated as a variable

    A = scipy.sparse.lil_matrix((num_vars, num_vars))
    res = np.zeros((h, w, 3), dtype=int)

    ymask, xmask, _ = np.where(mask == 1)
    print("array of y and x indices in the mask", ymask, xmask)

    # solve Ax = b_ci for each channel
    for ci in range(c):
        print("-"*100)
        b = np.zeros((num_vars, 1))
        for i in range(len(ymask)):   
            y, x = ymask[i], xmask[i]
            num_eq = (y - 1) * w + x
            center = im2var[y][x]

            ### fill matrix A and vector b
            # when we take the derivative, the center is added 4 times
            A[num_eq, center] = 4

            # top
            if y - 1 >= 0:
                yn, xn = y - 1, x
                si, sj = fg[y][x][ci], fg[yn][xn][ci]
                ti, tj = bg[y][x][ci], bg[yn][xn][ci]
                ds, dt = si - sj, ti - tj
                if abs(ds) > abs(dt):
                    dij = ds
                else:
                    dij = dt
                b[num_eq] += dij
                
                if mask[yn][xn]:
                    nb = im2var[yn][xn]
                    if ci == 0:
                        A[num_eq, nb] = -1
                else:
                    b[num_eq] += tj
            else:
                b[num_eq] += bg[y][x][ci]

            # bottom
            if y + 1 <= h - 1:
                yn, xn = y + 1, x
                si, sj = fg[y][x][ci], fg[yn][xn][ci]
                ti, tj = bg[y][x][ci], bg[yn][xn][ci]
                ds, dt = si - sj, ti - tj
                if abs(ds) > abs(dt):
                    dij = ds
                else:
                    dij = dt
                b[num_eq] += dij
                
                if mask[yn][xn]:
                    nb = im2var[yn][xn]
                    if ci == 0:
                        A[num_eq, nb] = -1
                else:
                    b[num_eq] += tj
            else:
                b[num_eq] += bg[y][x][ci]

            # left
            if x - 1 >= 0:
                yn, xn = y, x - 1
                si, sj = fg[y][x][ci], fg[yn][xn][ci]
                ti, tj = bg[y][x][ci], bg[yn][xn][ci]
                ds, dt = si - sj, ti - tj
                if abs(ds) > abs(dt):
                    dij = ds
                else:
                    dij = dt
                b[num_eq] += dij
                
                if mask[yn][xn]:
                    nb = im2var[yn][xn]
                    if ci == 0:
                        A[num_eq, nb] = -1
                else:
                    b[num_eq] += tj
            else:
                b[num_eq] += bg[y][x][ci]

            # right
            if x + 1 <= w - 1: 
                yn, xn = y, x + 1
                si, sj = fg[y][x][ci], fg[yn][xn][ci]
                ti, tj = bg[y][x][ci], bg[yn][xn][ci]
                ds, dt = si - sj, ti - tj
                if abs(ds) > abs(dt):
                    dij = ds
                else:
                    dij = dt
                b[num_eq] += dij
                
                if mask[yn][xn]:
                    nb = im2var[yn][xn]
                    if ci == 0:
                        A[num_eq, nb] = -1
                else:
                    b[num_eq] += tj
            else:
                b[num_eq] += bg[y][x][ci]
                
        lsq = scipy.sparse.linalg.lsqr(A.tocsr(), b)
        lsq = lsq[0] * 255
        res[:, :, ci] = var2image(lsq, h, w)
    bg = bg[:, :, :3]
    return (res / 255) * mask + bg * (1 - mask)

    # return fg * mask + bg * (1 - mask)

# -- Extra credit
def color2gray(rgb_image):
    """Naive conversion from an RGB image to a gray image."""
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

# -- Extra credit
def mixed_grad_color2gray(rgb_image):
    """EC: Convert an RGB image to gray image using mixed gradients."""
    img = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    fg = img[:, :, 1].reshape((img.shape[0], img.shape[1], 1)) / 255
    bg = img[:, :, 2].reshape((img.shape[0], img.shape[1], 1)) / 255
    mask = np.ones_like(img[:, :, 0]).reshape((img.shape[0], img.shape[1], 1))
    res = mixed_blend(fg, mask, bg)[:, :, 0]*255
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Poisson blending.")
    parser.add_argument("-q", "--question", required=True, choices=["toy", "blend", "mixed", "color2gray"])
    args, _ = parser.parse_known_args()

    # Example script: python proj2_starter.py -q toy
    if args.question == "toy":
        image = imageio.imread('./data/toy_problem.png') / 255.
        image_hat = toy_recon(image)

        plt.subplot(121)
        plt.imshow(image, cmap='gray')
        plt.title('Input')
        plt.subplot(122)
        plt.imshow(image_hat, cmap='gray')
        plt.title('Output')
        plt.show()

    # Example script: python proj2_starter.py -q blend -s data/source_01_newsource.png -t data/target_01.jpg -m data/target_01_mask.png
    if args.question == "blend":
        parser.add_argument("-s", "--source", required=True)
        parser.add_argument("-t", "--target", required=True)
        parser.add_argument("-m", "--mask", required=True)
        args = parser.parse_args()

        # after alignment (masking_code.py)
        ratio = 1
        fg = cv2.resize(imageio.imread(args.source), (0, 0), fx=ratio, fy=ratio)
        bg = cv2.resize(imageio.imread(args.target), (0, 0), fx=ratio, fy=ratio)
        mask = cv2.resize(imageio.imread(args.mask), (0, 0), fx=ratio, fy=ratio)

        fg = fg / 255.
        bg = bg / 255.
        mask = (mask.sum(axis=2, keepdims=True) > 0)

        blend_img = poisson_blend(fg, mask, bg)

        plt.subplot(121)
        bg = bg[:, :, :3]
        plt.imshow(fg * mask + bg * (1 - mask))
        plt.title('Naive Blend')
        plt.subplot(122)
        plt.imshow(blend_img)
        plt.title('Poisson Blend')
        plt.show()

    if args.question == "mixed":
        parser.add_argument("-s", "--source", required=True)
        parser.add_argument("-t", "--target", required=True)
        parser.add_argument("-m", "--mask", required=True)
        args = parser.parse_args()

        # after alignment (masking_code.py)
        ratio = 1
        fg = cv2.resize(imageio.imread(args.source), (0, 0), fx=ratio, fy=ratio)
        bg = cv2.resize(imageio.imread(args.target), (0, 0), fx=ratio, fy=ratio)
        mask = cv2.resize(imageio.imread(args.mask), (0, 0), fx=ratio, fy=ratio)

        fg = fg / 255.
        bg = bg / 255.
        mask = (mask.sum(axis=2, keepdims=True) > 0)

        blend_img = mixed_blend(fg, mask, bg)

        plt.subplot(121)
        bg = bg[:, :, :3]
        plt.imshow(fg * mask + bg * (1 - mask))
        plt.title('Naive Blend')
        plt.subplot(122)
        plt.imshow(blend_img)
        plt.title('Mixed Blend')
        plt.show()

    if args.question == "color2gray":
        parser.add_argument("-s", "--source", required=True)
        args = parser.parse_args()

        rgb_image = imageio.imread(args.source)
        gray_image = color2gray(rgb_image)
        mixed_grad_img = mixed_grad_color2gray(rgb_image)

        plt.subplot(121)
        plt.imshow(gray_image, cmap='gray')
        plt.title('rgb2gray')
        plt.subplot(122)
        plt.imshow(mixed_grad_img, cmap='gray', vmin=0, vmax=255)
        print(gray_image)
        print(mixed_grad_img)
        plt.title('mixed gradient')
        plt.show()

    plt.close()
