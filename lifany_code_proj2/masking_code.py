# # Nikhil Uday Shinde: 7/23/18
# # https://github.com/nikhilushinde/cs194-26_proj3_2.2

# import cv2
# import numpy as np
# import skimage as sk
# import skimage.io as skio

# # global variables for drawing on mask
# from skimage.transform import SimilarityTransform, warp


# drawing = False
# polygon = False
# centerMode = False
# contours = []
# polygon_center = None
# img = None

# def create_mask(imname):
#     masks_to_ret = {"centers":[], "contours":[], "offsets":[]}

#     global drawing, polygon, contours, centerMode, polygon_center
#     pressed_key = 0
#     # mouse callback function
#     def draw_circle(event,x,y,flags,param):
#         global drawing, centerMode, polygon, pressed_key
#         if drawing == True and event == cv2.EVENT_MOUSEMOVE:
#             cv2.circle(img,(x,y),10,(255,255,255),-1)
#             cv2.circle(mask,(x,y),10,(255,255,255),-1)
#         if polygon == True and event == cv2.EVENT_LBUTTONDOWN:
#             contours.append([x,y])
#             cv2.circle(img,(x,y),2,(255,255,255),-1)
#         if centerMode == True and event == cv2.EVENT_LBUTTONDOWN:
#             polygon_center = (x,y)
#             print(polygon_center)
#             cv2.circle(img, polygon_center, 3, (255, 0, 0), -1)
#             centerMode = False

#             masks_to_ret["centers"].append(polygon_center)
#             masks_to_ret["contours"].append(contours)

#     # Create a black image, a window and bind the function to window
#     orig_img = cv2.imread(imname)
#     reset_orig_img = orig_img[:]
#     mask = np.zeros(orig_img.shape, np.uint8)
#     img = np.array(orig_img[:])
#     cv2.namedWindow('image')

#     cv2.setMouseCallback('image',draw_circle)

#     angle = 0
#     delta_angle = 5
#     resize_factor = 1.1
#     total_resize = 1
#     adjusted = False

#     while(1):
#         cv2.imshow('image',img)
#         pressed_key = cv2.waitKey(20) & 0xFF

#         """
#         Commands:
#         d: toggle drawing mode
#         p: toggle polygon mode
#         q: draw polygon once selected, and select center
#         """

#         if pressed_key == 27:
#             break
#         elif pressed_key == ord('d'):
#             drawing = not drawing
#             print("drawing status: ", drawing)
#         elif pressed_key == ord('p'):
#             polygon = not polygon
#             print("polygon status: ", polygon)
#         elif polygon == True and pressed_key == ord('q') and len(contours) > 2:
#             contours = np.array(contours)
#             cv2.fillPoly(img, pts=[contours], color = (255,255,255))
#             cv2.fillPoly(mask, pts=[contours], color = (255,255,255))

#             centerMode = True
#             polygon = False
#         elif pressed_key == ord('o'):
#             # loop over the rotation angles again, this time ensuring
#             # no part of the image is cut off
#             angle = (angle + delta_angle) % 360
#             adjusted = True
#             print("Rotate")

#         elif pressed_key == ord('i'):
#             # loop over the rotation angles again, this time ensuring
#             # no part of the image is cut off
#             angle = (angle - delta_angle) % 360  
#             adjusted = True
#             print("Rotate")
        
#         # Plus
#         elif pressed_key == ord('='):
#             total_resize = total_resize*resize_factor
#             adjusted = True
#             print("Resize up")

#         # Minus
#         elif pressed_key == ord('-'):
#             total_resize = total_resize*(1/resize_factor)
#             adjusted = True
#             print("Resize down")
        

#         elif pressed_key == ord('r'):
#             img = np.array(reset_orig_img)
#             contours = []
#             masks_to_ret["centers"] = []
#             masks_to_ret["contours"] = []

#             centerMode = False
#             polygon = False
#             angle = 0
#             total_resize = 1

#             print("polygon status: False")

#         # adjust
#         if adjusted:
#             rows,cols,_ = orig_img.shape
#             M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
#             img = cv2.resize(orig_img, dsize=(0,0), fx=total_resize, fy=total_resize)
#             img = cv2.warpAffine(img,M,(cols,rows))
#             cv2.imshow('image', img)
#             adjusted = False
            

#     cv2.destroyAllWindows()
#     name = imname.split('/')[-1]

#     # store offsets to allow recreation of masks in target image
#     for center_num in range(len(masks_to_ret["centers"])):
#         offset = []
#         center = masks_to_ret["centers"][center_num]
#         for point in masks_to_ret["contours"][center_num]:
#             xoffset = point[0] - center[0]
#             yoffset = point[1] - center[1]

#             offset.append([xoffset, yoffset])
#         masks_to_ret["offsets"].append(offset)

#     # adjust the output image
#     rows,cols,_ = orig_img.shape
#     M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
#     adj_orig_img = cv2.resize(reset_orig_img, dsize=(0,0), fx=total_resize, fy=total_resize)
#     adj_orig_img = cv2.warpAffine(adj_orig_img,M,(cols,rows))
    
#     return masks_to_ret, adj_orig_img

# def paste_mask(im2name, masks_to_ret, im2=None):
#     im2masks_to_ret = {"centers":[], "contours":[]}

#     # mouse callback function
#     def draw_circle(event,x,y,flags,param):
#         if event == cv2.EVENT_LBUTTONDOWN:
#             centernew = [x,y]
#             new_contour = []
#             for offsets in masks_to_ret["offsets"]:
#                 for point in offsets:
#                     xnew = point[0] + centernew[0]
#                     ynew = point[1] + centernew[1]
#                     new_contour.append([xnew, ynew])
#             new_contour= np.array(new_contour)
#             im2masks_to_ret["centers"].append(centernew)
#             im2masks_to_ret["contours"].append(new_contour)

#             cv2.fillPoly(img, pts=[new_contour], color = (255,255,255))

#     # Create a black image, a window and bind the function to window
#     if type(im2) == type(None):
#         orig_img = cv2.imread(im2name)#np.zeros((512,512,3), np.uint8)
#     else:
#         orig_img = np.array(im2)

#     img = np.array(orig_img[:])
#     cv2.namedWindow('image')
#     cv2.resizeWindow('image', 600,600)
#     cv2.setMouseCallback('image',draw_circle)

#     while(1):
#         cv2.imshow('image',img)
#         pressed_key = cv2.waitKey(20) & 0xFF

#         if pressed_key == 27:
#             break 
#         if pressed_key == ord('r'):
#             img = np.array(orig_img)
#             im2masks_to_ret["centers"] = []
#             im2masks_to_ret["contours"] = []

#     return im2masks_to_ret, orig_img

# # run with 2 image names to generate and save masks and new source image
# def save_masks(im1name, im2name):
#     masks_to_ret, source_im = create_mask(imname)
#     im2masks_to_ret, target_im = paste_mask(im2name=im2name, masks_to_ret=masks_to_ret)
#     # im1 is the source, im2 is the target
#     source_mask = np.zeros((source_im.shape[0], source_im.shape[1], 3))
#     target_mask = np.zeros((target_im.shape[0], target_im.shape[1], 3))
#     cv2.fillPoly(source_mask, np.array([masks_to_ret["contours"][0]]), (255,255,255))
#     cv2.fillPoly(target_mask, np.array([im2masks_to_ret["contours"][0]]), (255,255,255))

#     name1 = im1name.split('/')[-1]
#     name1 = name1[:-4]

#     name2 = im2name.split('/')[-1]
#     name2 = name2[:-4]

#     source_mask = np.clip(sk.img_as_float(source_mask), -1, 1)
#     target_mask = np.clip(sk.img_as_float(target_mask), -1, 1)
#     source_im = np.clip(sk.img_as_float(source_im), -1, 1)
#     source_im = np.dstack([source_im[:,:,2], source_im[:,:,1], source_im[:,:,0]])

#     offset =  np.array(-im2masks_to_ret['contours'][0][0]) + np.array(masks_to_ret['contours'][0][0])
#     tform = SimilarityTransform(translation=offset)
#     warped = warp(source_im, tform, output_shape=target_im.shape)

#     # skio.imsave(name1 + "_mask_1.png", source_mask)
#     # skio.imsave(name2 + "_mask_1.png",target_mask)
#     # skio.imsave(name1 + "_newsource_1.png", warped)
#     # print(name1 + "_mask.png")
#     # return source_mask, target_mask, source_im
#     source_mask.save(name1 + "_mask.png")
#     target_mask.save(name2 + "_mask.png")
#     warped.save(name1 + "_newsource.png")


# # Example usage
# imname = "./data/source_01.jpg"
# im2name = "./data/target_01.jpg"
# save_masks(imname, im2name)


# Nikhil Uday Shinde: 7/23/18
# https://github.com/nikhilushinde/cs194-26_proj3_2.2

import cv2
import numpy as np
import skimage as sk
import skimage.io as skio
from skimage import img_as_ubyte
# global variables for drawing on mask
from skimage.transform import SimilarityTransform, warp
from PIL import Image

drawing = False
polygon = False
centerMode = False
contours = []
polygon_center = None
img = None

def create_mask(imname):
    masks_to_ret = {"centers":[], "contours":[], "offsets":[]}

    global drawing, polygon, contours, centerMode, polygon_center
    pressed_key = 0
    # mouse callback function
    def draw_circle(event,x,y,flags,param):
        global drawing, centerMode, polygon, pressed_key
        if drawing == True and event == cv2.EVENT_MOUSEMOVE:
            cv2.circle(img,(x,y),10,(255,255,255),-1)
            cv2.circle(mask,(x,y),10,(255,255,255),-1)
        if polygon == True and event == cv2.EVENT_LBUTTONDOWN:
            contours.append([x,y])
            cv2.circle(img,(x,y),2,(255,255,255),-1)
        if centerMode == True and event == cv2.EVENT_LBUTTONDOWN:
            polygon_center = (x,y)
            print(polygon_center)
            cv2.circle(img, polygon_center, 3, (255, 0, 0), -1)
            centerMode = False

            masks_to_ret["centers"].append(polygon_center)
            masks_to_ret["contours"].append(contours)

    # Create a black image, a window and bind the function to window
    orig_img = cv2.imread(imname)
    reset_orig_img = orig_img[:]
    mask = np.zeros(orig_img.shape, np.uint8)
    img = np.array(orig_img[:])
    cv2.namedWindow('image')

    cv2.setMouseCallback('image',draw_circle)

    angle = 0
    delta_angle = 5
    resize_factor = 1.1
    total_resize = 1
    adjusted = False

    while(1):
        cv2.imshow('image',img)
        pressed_key = cv2.waitKey(20) & 0xFF

        """
        Commands:
        d: toggle drawing mode
        p: toggle polygon mode
        q: draw polygon once selected, and select center
        """

        if pressed_key == 27:
            break
        elif pressed_key == ord('d'):
            drawing = not drawing
            print("drawing status: ", drawing)
        elif pressed_key == ord('p'):
            polygon = not polygon
            print("polygon status: ", polygon)
        elif polygon == True and pressed_key == ord('q') and len(contours) > 2:
            contours = np.array(contours)
            cv2.fillPoly(img, pts=[contours], color = (255,255,255))
            cv2.fillPoly(mask, pts=[contours], color = (255,255,255))

            centerMode = True
            polygon = False
        elif pressed_key == ord('o'):
            # loop over the rotation angles again, this time ensuring
            # no part of the image is cut off
            angle = (angle + delta_angle) % 360
            adjusted = True
            print("Rotate")

        elif pressed_key == ord('i'):
            # loop over the rotation angles again, this time ensuring
            # no part of the image is cut off
            angle = (angle - delta_angle) % 360  
            adjusted = True
            print("Rotate")
        
        # Plus
        elif pressed_key == ord('='):
            total_resize = total_resize*resize_factor
            adjusted = True
            print("Resize up")

        # Minus
        elif pressed_key == ord('-'):
            total_resize = total_resize*(1/resize_factor)
            adjusted = True
            print("Resize down")
        

        elif pressed_key == ord('r'):
            img = np.array(reset_orig_img)
            contours = []
            masks_to_ret["centers"] = []
            masks_to_ret["contours"] = []

            centerMode = False
            polygon = False
            angle = 0
            total_resize = 1

            print("polygon status: False")

        # adjust
        if adjusted:
            rows,cols,_ = orig_img.shape
            M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
            img = cv2.resize(orig_img, dsize=(0,0), fx=total_resize, fy=total_resize)
            img = cv2.warpAffine(img,M,(cols,rows))
            cv2.imshow('image', img)
            adjusted = False
            

    cv2.destroyAllWindows()
    name = imname.split('/')[-1]

    # store offsets to allow recreation of masks in target image
    for center_num in range(len(masks_to_ret["centers"])):
        offset = []
        center = masks_to_ret["centers"][center_num]
        for point in masks_to_ret["contours"][center_num]:
            xoffset = point[0] - center[0]
            yoffset = point[1] - center[1]

            offset.append([xoffset, yoffset])
        masks_to_ret["offsets"].append(offset)

    # adjust the output image
    rows,cols,_ = orig_img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    adj_orig_img = cv2.resize(reset_orig_img, dsize=(0,0), fx=total_resize, fy=total_resize)
    adj_orig_img = cv2.warpAffine(adj_orig_img,M,(cols,rows))
    # print('masks_to_ret', masks_to_ret)
    # print('adj_orig_img', adj_orig_img.shape)
    return masks_to_ret, adj_orig_img

def paste_mask(im2name, masks_to_ret, im2=None):
    im2masks_to_ret = {"centers":[], "contours":[]}

    # mouse callback function
    def draw_circle(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            centernew = [x,y]
            new_contour = []
            for offsets in masks_to_ret["offsets"]:
                for point in offsets:
                    xnew = point[0] + centernew[0]
                    ynew = point[1] + centernew[1]
                    new_contour.append([xnew, ynew])
            new_contour= np.array(new_contour)
            print(new_contour)
            im2masks_to_ret["centers"].append(centernew)
            im2masks_to_ret["contours"].append(new_contour)

            cv2.fillPoly(img, pts=[new_contour], color = (255,255,255))

    # Create a black image, a window and bind the function to window
    if type(im2) == type(None):
        orig_img = cv2.imread(im2name)#np.zeros((512,512,3), np.uint8)
    else:
        orig_img = np.array(im2)

    img = np.array(orig_img[:])
    cv2.namedWindow('image')
    cv2.resizeWindow('image', 600,600)
    cv2.setMouseCallback('image',draw_circle)

    while(1):
        cv2.imshow('image',img)
        pressed_key = cv2.waitKey(20) & 0xFF

        if pressed_key == 27:
            break 
        if pressed_key == ord('r'):
            img = np.array(orig_img)
            im2masks_to_ret["centers"] = []
            im2masks_to_ret["contours"] = []
    print("im2masks_to_ret", im2masks_to_ret)
    print("orig_img", orig_img.shape)
    return im2masks_to_ret, orig_img

# run with 2 image names to generate and save masks and new source image
def save_masks(im1name, im2name):
    masks_to_ret, source_im = create_mask(imname)
    im2masks_to_ret, target_im = paste_mask(im2name=im2name, masks_to_ret=masks_to_ret)
    # im1 is the source, im2 is the target
    source_mask = np.zeros((source_im.shape[0], source_im.shape[1], 3))
    target_mask = np.zeros((target_im.shape[0], target_im.shape[1], 3))
    cv2.fillPoly(source_mask, np.array([masks_to_ret["contours"][0]]), (255,255,255))
    cv2.fillPoly(target_mask, np.array([im2masks_to_ret["contours"][0]]), (255,255,255))

    name1 = im1name.split('/')[-1]
    name1 = name1[:-4]

    name2 = im2name.split('/')[-1]
    name2 = name2[:-4]

    source_mask = np.clip(sk.img_as_float(source_mask), -1, 1)
    target_mask = np.clip(sk.img_as_float(target_mask), -1, 1)
    source_im = np.clip(sk.img_as_float(source_im), -1, 1)
    source_im = np.dstack([source_im[:,:,2], source_im[:,:,1], source_im[:,:,0]])

    offset =  np.array(-im2masks_to_ret['contours'][0][0]) + np.array(masks_to_ret['contours'][0][0])
    tform = SimilarityTransform(translation=offset)
    warped = warp(source_im, tform, output_shape=target_im.shape)
    # print('source_mask', source_mask.shape)
    # print('target_mask', target_mask.shape)
    # print('warped', warped.shape)
    # skio.imsave(name1 + "_mask.png", source_mask)
    # skio.imsave(name2 + "_mask.png",target_mask)
    # skio.imsave(name1 + "_newsource.png", warped)
    # print(name1 + "_mask.png")
    
    source_mask = (source_mask - source_mask.min()) / (source_mask.max() - source_mask.min()) * 255
    target_mask = (target_mask - target_mask.min()) / (target_mask.max() - target_mask.min()) * 255
    warped = (warped - warped.min()) / (warped.max() - warped.min()) * 255
    source_mask = source_mask.astype(np.uint8)
    target_mask = target_mask.astype(np.uint8)
    warped = warped.astype(np.uint8)
    source_mask = Image.fromarray(source_mask)
    target_mask = Image.fromarray(target_mask)
    warped = Image.fromarray(warped)
    
    source_mask.save("custom_data/" + name1 + "_mask.png")
    target_mask.save("custom_data/" + name2 + "_mask.png")
    warped.save("custom_data/" + name1 + "_newsource.png")

    
    return source_mask, target_mask, source_im

# Example usage
imname = "./custom_data/source_02.jpg"
im2name = "./custom_data/target_02_new.jpg"
save_masks(imname, im2name)

# custom 
# src image2 link: data:image/jpeg;base64,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
# target image2 link: data:image/jpeg;base64,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