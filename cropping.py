from PIL import Image 
import cv2
import numpy as np



# Read reference image
refImage = "PICS_11.05/thresholded/thresholded11.jpg"
img = cv2.imread(refImage, cv2.IMREAD_COLOR)

height, width = img.shape[:2]

start_row, start_col = 540, 200
end_row, end_col = 617, 1607

cropped = img[start_row:end_row, start_col:end_col]


cropped_image = "PICS_11.05/cropped/cropped2.jpg"
cv2.imwrite(cropped_image, cropped)


# cv2.namedWindow('original', cv2.WINDOW_NORMAL)
# cv2.imshow('original',cropped)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print(height, width)