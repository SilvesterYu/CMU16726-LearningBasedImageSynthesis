# (16-726): Project 1 starter Python code
# credit to https://inst.eecs.berkeley.edu/~cs194-26/fa18/hw/proj1/data/colorize_skel.py
# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

# custom data download link: https://memory.loc.gov/master/pnp/prok/

import numpy as np
import skimage as sk
import skimage.io as skio
import os
import PIL
import torch

# name of the input file
folders = ["data"]
# folders = ["data", "custom"]
tifs = []
for folder in folders:
    tifs.extend([folder + "/" + item for item in os.listdir(folder)])
print("\nFiles to align: ", tifs)

def align(c1, c2):

    # in this case, channel2 (c2) is b channel
    window = [-15, 15] # a possible window of displacements
    shifting = [0, 0] # record how channel 1 (c1) needs to shifts [x, y]
    c = 2 # a constant to determin scaling
    total_shifting = [0, 0]

    total_scales = np.floor(np.log(c1.shape[0]/(100*c)) / np.log(c)).astype(int) + 1
    print("total_scales ", total_scales)

    # # image scale too small, directly align without pyramid
    # if total_scales < 3:
    #     total_scales = 0

    edge_size = 0.19 # how much of the image is considered as edge
    
    # pyramid method to speed up large image alignments
    for scale in range(total_scales, -1, -1):
        min_ssd = np.inf
        # crop some edges because they differ a lot
        c1_scaled = sk.transform.rescale(c1, 1 / c** scale, anti_aliasing=True)
        c2_scaled = sk.transform.rescale(c2, 1 / c** scale, anti_aliasing=True)
        h, w = c1_scaled.shape
        edge_h, edge_w = int(edge_size * h), int(edge_size * w)
        c1_crop = c1_scaled[edge_h : -edge_h, edge_w : -edge_w]
        c2_crop = c2_scaled[edge_h : -edge_h, edge_w : -edge_w]

        for x_shift in range(shifting[0] + window[0], shifting[0] + window[1] + 1):
            for y_shift in range(shifting[1] + window[0], shifting[1] + window[1] + 1):
                shifted_c1_crop = np.roll(c1_crop, (x_shift, y_shift), axis=(1, 0))

                # Default alignment metric: SSD
                ssd = np.sum((shifted_c1_crop - c2_crop) ** 2)
                if ssd < min_ssd:
                    min_ssd = ssd
                    shifting = [x_shift, y_shift] 

        shifting = [shifting[0]*c, shifting[1]*c]
        total_shifting = [total_shifting[0] + shifting[0], total_shifting[1] + shifting[1]]
        print("scale 1/", c** scale, " shifting ", shifting)   

    total_shifting = [total_shifting[0] - int(shifting[0]/c), total_shifting[1] - int(shifting[1]/c)]
    shifting = [int(shifting[0]/c), int(shifting[1]/c)]
    c1 = np.roll(c1, (shifting[0], shifting[1]), axis=(1, 0))
    return c1, total_shifting

def single_scale_align(c1, c2):
    # in this case, channel2 (c2) is b channel
    window = [-15, 15] # a possible window of displacements
    shifting = [0, 0] # record how channel 1 (c1) needs to shifts [x, y]
    min_ssd = np.Inf

    for x_shift in range(shifting[0] + window[0], shifting[0] + window[1] + 1):
        for y_shift in range(shifting[1] + window[0], shifting[1] + window[1] + 1):
            shifted_c1_crop = np.roll(c1, (x_shift, y_shift), axis=(1, 0))

            # Default alignment metric: SSD
            ssd = np.sum((shifted_c1_crop - c2) ** 2)
            if ssd < min_ssd:
                min_ssd = ssd
                shifting = [x_shift, y_shift] 
    c1 = np.roll(c1, (shifting[0], shifting[1]), axis=(1, 0))
    return c1, shifting

def contrast(c):
    contrast_c = 1.3
    if type(c) == torch.Tensor:
        mean_c = torch.mean(c)
    else:
        mean_c = np.mean(c)
    c[c < mean_c] /= contrast_c
    c[c > mean_c] *= contrast_c
    if type(c) == torch.Tensor:
        max_c, min_c = torch.max(c), torch.min(c)
    else:
        max_c, min_c = np.max(c), np.min(c)
    c = (c - min_c) / (max_c - min_c)
    return c

# The main loop for single scale
def main0():
    for imname in tifs:
        # read in the image
        im = skio.imread(imname)
        print("\nAligning file " + imname)

        # convert to double (might want to do this later on to save memory)    
        im = sk.img_as_float(im)
            
        # compute the height of each part (just 1/3 of total)
        height = np.floor(im.shape[0] / 3.0).astype(np.int)

        # separate color channels
        b = im[:height]
        g = im[height: 2*height]
        r = im[2*height: 3*height]

        ag, g_shifting = single_scale_align(g, b)
        ar, r_shifting = single_scale_align(r, b)
        im_out = np.dstack([ar, ag, b])

        im_out = im_out*255
        im_out = im_out.astype(np.uint8)

        jpgfname = imname.split("/")[-1].rstrip(".tif") + "_single_scale"
        # save the image
        fname = 'results/' + jpgfname + '.jpg'
        # print("imout", im_out)
        skio.imsave(fname, im_out)

        # display the image
        skio.imshow(im_out)
        skio.show()


# The main loop with numpy
def main1():
    for imname in tifs:
        # read in the image
        im = skio.imread(imname)
        print("\nAligning file " + imname)

        # convert to double (might want to do this later on to save memory)    
        im = sk.img_as_float(im)
            
        # compute the height of each part (just 1/3 of total)
        height = np.floor(im.shape[0] / 3.0).astype(np.int)

        # separate color channels
        b = im[:height]
        g = im[height: 2*height]
        r = im[2*height: 3*height]

        print("before crop", b.shape)

        row_len, col_len = b.shape[1], b.shape[0]
        crop_i1, crop_j1, crop_i2, crop_j2 = 0, 0, 0, 0
        for i in range(int(col_len * 0.2)):
            bi1, gi1, ri1, bi2, gi2, ri2 = 0, 0, 0, 0, 0, 0
            b_row = b[i, :]
            g_row = g[i, :]
            r_row = r[i, :]
            if len(b_row[b_row > crop_thresh]) > crop_proportion * row_len:
                bi1 = i
            if len(g_row[g_row > crop_thresh]) > crop_proportion * row_len:
                gi1 = i
            if len(r_row[r_row > crop_thresh]) > crop_proportion * row_len:
                ri1 = i
            if max([bi1, gi1, ri1]) > crop_i1:
                crop_i1 = max([bi1, gi1, ri1])
            i2 = -i - 1
            b_row = b[i2, :]
            g_row = g[i2, :]
            r_row = r[i2, :]
            if len(b_row[b_row > crop_thresh]) > crop_proportion * row_len:
                bi2 = i2
            if len(g_row[g_row > crop_thresh]) > crop_proportion * row_len:
                gi2 = i2
            if len(r_row[r_row > crop_thresh]) > crop_proportion * row_len:
                ri2 = i2
            if min([bi2, gi2, ri2]) < crop_i2:
                crop_i2 = min([bi2, gi2, ri2])

        for i in range(int(row_len * 0.2)):
            bi1, gi1, ri1, bi2, gi2, ri2 = 0, 0, 0, 0, 0, 0
            b_col = b[:, i]
            g_col = g[:, i]
            r_col = r[:, i]
            if len(b_col[b_col > crop_thresh]) > crop_proportion * col_len:
                bi1 = i
            if len(g_col[g_col > crop_thresh]) > crop_proportion * col_len:
                gi1 = i
            if len(r_col[r_col > crop_thresh]) > crop_proportion * col_len:
                ri1 = i
            if max([bi1, gi1, ri1]) > crop_j1:
                crop_j1 = max([bi1, gi1, ri1])
            i2 = -i - 1
            b_col = b[:, i2]
            b_col = g[:, i2]
            b_col = r[:, i2]
            if len(b_col[b_col > crop_thresh]) > crop_proportion * col_len:
                bi2 = i2
            if len(g_col[g_col > crop_thresh]) > crop_proportion * col_len:
                gi2 = i2
            if len(r_col[r_col > crop_thresh]) > crop_proportion * col_len:
                ri2 = i2
            if min([bi2, gi2, ri2]) < crop_j2:
                crop_j2 = min([bi2, gi2, ri2])
        
        if crop_i2 == 0:
            crop_i2 = col_len
        if crop_j2 == 0:
            crop_j2 = row_len

        b = b[crop_i1:crop_i2, crop_j1:crop_j2]
        g = g[crop_i1:crop_i2, crop_j1:crop_j2]
        r = r[crop_i1:crop_i2, crop_j1:crop_j2]
        print("after crop", b.shape)

        # align the images
        # functions that might be useful for aligning the images include:
        # np.roll, np.sum, sk.transform.rescale (for multiscale)
        if "emir" in imname:
            ab, g_shifting = align(b, g)
            ar, r_shifting = align(r, g)
            if CONTRAST:
                im_out = np.dstack([contrast(ar), contrast(g), contrast(ab)])
            else:
                im_out = np.dstack([ar, g, ab])
            if CROP:

                if g_shifting[0] >= 0 and r_shifting[0] >= 0:
                    i1 = max(g_shifting[0], r_shifting[0])
                    # i2 = im_out.shape[0]
                    i2 = -i1
                elif g_shifting[0] < 0 and r_shifting[0] < 0:
                    i2 = min(g_shifting[0], r_shifting[0])
                    # i1 = 0
                    i1 = -i2
                else:
                    i2 = min(g_shifting[0], r_shifting[0])
                    i1 = max(g_shifting[0], r_shifting[0])
                if g_shifting[1] >= 0 and r_shifting[1] >= 0:
                    j1 = max(g_shifting[1], r_shifting[1])
                    # j2 = im_out.shape[1]
                    j2 = -j1
                elif g_shifting[1] < 0 and r_shifting[1] < 0:
                    j2 = min(g_shifting[1], r_shifting[1])
                    # j1 = 0
                    j1 = -j2
                else:
                    j2 = min(g_shifting[1], r_shifting[1])
                    j1 = max(g_shifting[1], r_shifting[1])
                print(i1, i2, j1, j2)
                const = 1.5
                j1, j2 = int(j1*const), int(j2*const)
                im_out = im_out[i1 : i2, j1: j2, :]
        else:
            ag, g_shifting = align(g, b)
            ar, r_shifting = align(r, b)
            # create a color image
            if CONTRAST:
                im_out = np.dstack([contrast(ar), contrast(ag), contrast(b)])
            else:
                im_out = np.dstack([ar, ag, b])
            if CROP:
                if g_shifting[0] >= 0 and r_shifting[0] >= 0:
                    i1 = max(g_shifting[0], r_shifting[0])
                    # i2 = im_out.shape[0]
                    i2 = -i1
                elif g_shifting[0] < 0 and r_shifting[0] < 0:
                    i2 = min(g_shifting[0], r_shifting[0])
                    # i1 = 0
                    i1 = -i2
                else:
                    i2 = min(g_shifting[0], r_shifting[0])
                    i1 = max(g_shifting[0], r_shifting[0])
                if g_shifting[1] >= 0 and r_shifting[1] >= 0:
                    j1 = max(g_shifting[1], r_shifting[1])
                    # j2 = im_out.shape[1]
                    j2 = -j1
                elif g_shifting[1] < 0 and r_shifting[1] < 0:
                    j2 = min(g_shifting[1], r_shifting[1])
                    # j1 = 0
                    j1 = -j2
                else:
                    j2 = min(g_shifting[1], r_shifting[1])
                    j1 = max(g_shifting[1], r_shifting[1])
                print(i1, i2, j1, j2)
                const = 1.5
                i1, i2 = int(i1*const), int(i2*const)
                im_out = im_out[i1+30 : i2, j1: j2, :]

        im_out = im_out*255
        im_out = im_out.astype(np.uint8)

        jpgfname = imname.split("/")[-1].rstrip(".tif")

        if CONTRAST:
            jpgfname = jpgfname + "_contrast"
        if CROP:
            jpgfname = jpgfname + "_crop"
        # save the image
        fname = 'results/' + jpgfname + '.jpg'
        # print("imout", im_out)
        skio.imsave(fname, im_out)

        # display the image
        skio.imshow(im_out)
        skio.show()

# Extra credit: Pytorch Tensor
def align_tensor(c1, c2, device):

    # in this case, channel2 (c2) is b channel
    window = [-15, 15] # a possible window of displacements
    shifting = [0, 0] # record how channel 1 (c1) needs to shifts [x, y]
    c = 2 # a constant to determin scaling

    total_scales = np.floor(np.log(c1.shape[0]/(100*c)) / np.log(c)).astype(int) + 1
    print("total_scales ", total_scales)

    edge_size = 0.19 # how much of the image is considered as edge
    
    for scale in range(total_scales, -1, -1):
        min_ssd = np.inf
        # crop some edges because they differ a lot
        c1_scaled = torch.Tensor(sk.transform.rescale(c1.cpu().numpy(), 1 / c** scale, anti_aliasing=True)).to(device)
        c2_scaled = torch.Tensor(sk.transform.rescale(c2.cpu().numpy(), 1 / c** scale, anti_aliasing=True)).to(device)
        h, w = c1_scaled.shape
        edge_h, edge_w = int(edge_size * h), int(edge_size * w)
        c1_crop = c1_scaled[edge_h : -edge_h, edge_w : -edge_w]
        c2_crop = c2_scaled[edge_h : -edge_h, edge_w : -edge_w]

        for x_shift in range(shifting[0] + window[0], shifting[0] + window[1] + 1):
            for y_shift in range(shifting[1] + window[0], shifting[1] + window[1] + 1):
                shifted_c1_crop = torch.roll(c1_crop, shifts = (x_shift, y_shift), dims=(1, 0))

                # Default alignment metric: SSD
                # ssd = np.sum((shifted_c1_crop - c2_crop) ** 2)
                criterion = torch.nn.MSELoss()
                ssd = criterion(shifted_c1_crop, c2_crop)
                if ssd < min_ssd:
                    min_ssd = ssd
                    shifting = [x_shift, y_shift] 

        shifting = [shifting[0]*c, shifting[1]*c]
        print("scale 1/", c** scale, " shifting ", shifting)

    shifting = [int(shifting[0]/c), int(shifting[1]/c)]
    c1 = torch.roll(c1, shifts = (shifting[0], shifting[1]), dims = (1, 0))
    return c1

# The main loop with Pytorch
def main2():

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    for imname in tifs:
        # read in the image
        im = skio.imread(imname)
        print("\nAligning file " + imname)

        # convert to double (might want to do this later on to save memory)    
        im = sk.img_as_float(im)
        im = torch.from_numpy(im).to(device)

        # compute the height of each part (just 1/3 of total)
        height = np.floor(im.shape[0] / 3.0).astype(np.int)

        # separate color channels
        b = im[:height]
        g = im[height: 2*height]
        r = im[2*height: 3*height]

        print("b", b)

        # align the images
        # functions that might be useful for aligning the images include:
        # np.roll, np.sum, sk.transform.rescale (for multiscale)
        if "emir" in imname:
            ab = align_tensor(b, g, device)
            ar = align_tensor(r, g, device)
            if CONTRAST:
                im_out = torch.dstack([contrast(ar), contrast(g), contrast(ab)])
            else:
                im_out = torch.dstack([ar, g, ab])

        else:
            ag = align_tensor(g, b, device)
            ar = align_tensor(r, b, device)
            # create a color image
            if CONTRAST:
                im_out = torch.dstack([contrast(ar), contrast(ag), contrast(b)])
            else:
                im_out = torch.dstack([ar, ag, b])

        im_out = np.array(im_out.cpu())*255
        im_out = im_out.astype(np.uint8)

        jpgfname = imname.split("/")[-1].rstrip(".tif")

        if CONTRAST:
            jpgfname = imname.split("/")[-1].rstrip(".tif") + "_contrast"

        # save the image
        fname = 'results/' + jpgfname + '_torch.jpg'
        skio.imsave(fname, im_out)

        # display the image
        skio.imshow(im_out)
        skio.show()


if __name__ == "__main__":

    # Extra credit: Automatic cropping
    CROP = True
    crop_thresh = 0.96
    crop_proportion = 0.85

    # Extra credit: Automatic contrasting
    CONTRAST = True

    # The main loop using single scale
    # main0()

    # The main loop using numpy, pyramid, crop and adding contrast
    main1()

    # Extra credit: The main loop using Pytorch, pyramid
    # main2()

