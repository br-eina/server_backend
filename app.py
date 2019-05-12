from flask import Flask, jsonify, request
import cv2
import numpy as np
import base64
import io
import csv
import pytesseract
from PIL import Image 
import json

app = Flask(__name__)

mass = []
mass1 = []
recognized_text = [
        {
                "title" : "Name",
                "value" : "Vvsvsvsv"
        },
        {
                "title" : "Make",
                "value" : "2122121"
        }
]

recognized_json = []
new_json = []
text_mas = []

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
    #cv2.imwrite("pics/matches_123.jpg", imMatches)

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

def stringToRGB(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    image = Image.open(io.BytesIO(imgdata))
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)


def processing():
    # Read reference image
    path_reference_image = "pics/forms/form.jpg"
    reference_image = cv2.imread(path_reference_image, cv2.IMREAD_COLOR)

    # Base64 string to image
    f = open("strings/image_client.txt", "r")
    if f.mode == "r":
        imgstring = f.read()

    image_client = stringToRGB(imgstring)

    # Saving image from client
    path_image_client = "pics/image_client.jpg"
    cv2.imwrite(path_image_client, image_client)

    # Read image to be aligned
    path_toAlignImage = "pics/image_client.jpg"
    toAlignImage = cv2.imread(path_toAlignImage, cv2.IMREAD_COLOR)

    # Align images
    aligned_image, h, imMathces = align_images(toAlignImage, reference_image)

    # Saving aligned image 
    path_aligned_image = "pics/aligned_image.jpg"
    cv2.imwrite(path_aligned_image, aligned_image)

    img = cv2.imread(path_aligned_image, 0)
    th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 91, 12)

    outFileName = "PICS_11.05/first_align/thresholded6.jpg"
    cv2.imwrite(outFileName, th)

    # Read form1 reference image
    path_form1 = "pics/forms/form1.jpg"
    form1 = cv2.imread(path_form1, cv2.IMREAD_COLOR)

    # Read image to be aligned
    im1 = cv2.imread(outFileName, cv2.IMREAD_COLOR)

    # Alignment with form1
    imReg1, h, imMathces = align_images(im1, form1)

    # Saving of aligned image to the disc
    aligned_image1 = "PICS_11.05/second_align/aligned6.jpg"
    cv2.imwrite(aligned_image1, imReg1)

    # Read form2 reference image
    path_form2 = "pics/forms/form2.jpg"
    form2 = cv2.imread(path_form2, cv2.IMREAD_COLOR)

    # Read image to be aligned
    im2 = cv2.imread(aligned_image1, cv2.IMREAD_COLOR)

    # Alignment with form2
    imReg2, h, imMathces = align_images(im2, form2)

    # Saving of aligned image to the disc
    aligned_image2 = "PICS_12.05/align_form2/aligned1.jpg"
    cv2.imwrite(aligned_image2, imReg2)

    # Read form3 reference image
    path_form3 = "pics/forms/form3.jpg"
    form3 = cv2.imread(path_form3, cv2.IMREAD_COLOR)

    # Read image to be aligned
    im3 = cv2.imread(aligned_image2, cv2.IMREAD_COLOR)

    # Alignment with form3
    imReg3, h, imMathces = align_images(im3, form3)

    # Saving of aligned image to the disc
    aligned_image3 = "pics/thresholded_image.jpg"
    cv2.imwrite(aligned_image3, imReg3)

    # Opening thresholded image as base64 string
    with open("pics/thresholded_image.jpg", 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    # Saving thresholded image as base64 string
    f = open("strings/image_server.txt", "w+")
    for i in encoded_string:
        f.write(i)


def editor(val, imgstring):
	val = imgstring
	return val

def change_data(imgstring):
	for el in mass:
		val = el["value"]
		el["value"] = editor(val, imgstring)

              
def crop_im():
        coordinates = "pics/cropped/coordinates.csv"
        with open(coordinates, 'r') as csv_file:
                csv_reader = csv.DictReader(csv_file, delimiter=";")
        
                y1, x1, y2, x2 = [], [], [], []

                for line in csv_reader:
                        y1.append(line['y1'])
                        x1.append(line['x1'])
                        y2.append(line['y2'])
                        x2.append(line['x2'])
        
        y1 = [int(i) for i in y1]
        x1 = [int(i) for i in x1]
        y2 = [int(i) for i in y2]
        x2 = [int(i) for i in x2]

        image_to_crop = "pics/thresholded_image.jpg"
        img = cv2.imread(image_to_crop, cv2.IMREAD_COLOR)
        
        names = ["Name", "Make", "Model", "Color", "VIN", "State", "Location", "Plate"]

        for i in range(len(y1)):
                start_row, start_col = y1[i], x1[i]
                end_row, end_col = y2[i], x2[i]
                cropped_image = "pics/cropped/cropped_{0}.jpg".format(i)
                cv2.imwrite(cropped_image, img[start_row:end_row, start_col:end_col])
                text = pytesseract.image_to_string(Image.open(cropped_image))
                text_mas.append(text)
                recognized_json.append({names[i] : text})

        for i in range(len(names)):
                new_json.append({"title" : names[i], "value" : text_mas[i]})

        path = './pics/cropped'
        file_name = 'recognized_data'

        write_json_to_file(path, file_name, new_json)

        
        

def open_save():
        refImage = "pics/thresholded_image.jpg"
        im = cv2.imread(refImage, cv2.IMREAD_COLOR)

        immage = "pics/cropped/immage.jpg"
        cv2.imwrite(immage, im)

def write_json_to_file(path, file_name, data):
        file_name_extension = './' + path + '/' + file_name + '.txt'
        with open(file_name_extension, 'w') as file:
                json.dump(data, file)


@app.route('/', methods=['GET'])
def test():
	return jsonify({'message' : 'It works!'})



@app.route('/post_data', methods=['POST'])
def post_data():
    data = {'value' : request.json['value']}
    field = data['value']
    write_file(field)
    mass.append(data)
    return jsonify(mass)

@app.route('/show_data', methods=['GET'])
def show_data():
    imgstring = read_file()
    change_data(imgstring)

    return jsonify(mass)

@app.route('/show_text_data', methods=['GET', 'POST'])
def show_text_data():
        crop_im()
        return jsonify(new_json)

# @app.route('/show_text_data', methods=['GET', 'POST')
# def show_text_data():
#         recognizing_data()
#         return jsonify(recognized_json)


@app.route('/initiate_processing', methods=['POST'])
def initiate_processing():
        data1 = {'value' : request.json['value']}
        processing()
        mass1.append(data1)
        return jsonify(mass1)

def write_file(field):
    f = open("strings/image_client.txt", "w+")
    for i in field:
        f.write(i)

def read_file():
    f = open("strings/image_server.txt", "r")
    if f.mode == "r":
        imgstring = f.read()
        return imgstring



if __name__=='__main__':
	app.run(debug=True, port=5000)