# code from: https://stackoverflow.com/questions/56183201/detect-and-visualize-differences-between-two-images-with-opencv-python
import cv2

# load images
image1 = cv2.imread("/home/lifan/Documents/GitHub/ImageSynthesis/lifany_code_proj4/results/style.png")
image2 = cv2.imread("/home/lifan/Documents/GitHub/ImageSynthesis/lifany_code_proj4/results/randnoise36.png")

# compute difference
difference = cv2.subtract(image1, image2)

# color the mask red
Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
difference[mask != 255] = [255, 0, 255]

# add the red mask to the images to make the differences obvious
image1[mask != 255] = [255, 0, 255]
image2[mask != 255] = [255, 0, 255]

# store images
cv2.imwrite('results/diffOverOriginal36.png', image1)
cv2.imwrite('results/diffOverImage36.png', image2)
cv2.imwrite('results/diff36.png', difference)