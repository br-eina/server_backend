import cv2
import numpy as np
#import base64
#import io
#from PIL import Image

MAX_FEATURES = 1000
GOOD_MATCH_PERCENT = 0.15

def align_images(im1, im2):

    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key = lambda x: x.distance, reverse = False)

    # Remove worst matches
    num_good_matches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:num_good_matches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("PICS_11.05/matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype = np.float32)
    points2 = np.zeros((len(matches), 2), dtype = np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h, imMatches

if __name__ == '__main__':

    # Read first reference image
    refImage = "PICS_11.05/form.jpg"
    imReference = cv2.imread(refImage, cv2.IMREAD_COLOR)

    # Read image to be aligned
    phImage = "PICS_11.05/photo/photo6.jpg"
    im = cv2.imread(phImage, cv2.IMREAD_COLOR)

    # Align images
    imReg, h, imMathces = align_images(im, imReference)

    # Save aligned image to the disc
    aligned_image = "PICS_11.05/first_align/aligned6.jpg"
    cv2.imwrite(aligned_image, imReg)

    # Read second reference image
    refImage2 = "PICS_11.05/thresholded/new_form.jpg"
    imReference2 = cv2.imread(refImage2, cv2.IMREAD_COLOR)

    # Read second image to be aligned
    # phImage = "PICS_11.05/thresholded/thresholded6.jpg"
    im2 = cv2.imread(aligned_image, cv2.IMREAD_COLOR)

    # Second alignment
    imReg2, h, imMathces = align_images(im2, imReference2)

    # Second saving of aligned image to the disc
    aligned_image2 = "PICS_11.05/second_align/aligned6.jpg"
    cv2.imwrite(aligned_image2, imReg2)

    # Read third image to be aligned
    # phImage = "PICS_11.05/thresholded/thresholded6.jpg"
    im3 = cv2.imread(aligned_image2, cv2.IMREAD_COLOR)

    # Third alignment
    imReg3, h, imMathces = align_images(im3, imReference2)

    # Third saving of aligned image to the disc
    aligned_image3 = "PICS_11.05/second_align/aligned6.jpg"
    cv2.imwrite(aligned_image3, imReg3)

    img = cv2.imread(aligned_image3, 0)
    th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 91, 12)

    outFileName = "PICS_11.05/thresholded/thresholded6.jpg"
    cv2.imwrite(outFileName, th)

    
