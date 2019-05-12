from PIL import Image 
import cv2
import numpy as np
import pytesseract
import os
import csv
import json


number = input()


coordinates = "PICS_12.05/coordinates.csv"
with open(coordinates, 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=";")

    y1, x1, y2, x2 = [], [], [], []

    for line in csv_reader:
        y1.append(line['y1'])
        x1.append(line['x1'])
        y2.append(line['y2'])
        x2.append(line['x2'])
        # print(line['y1'])

y1 = [int(i) for i in y1]
x1 = [int(i) for i in x1]
y2 = [int(i) for i in y2]
x2 = [int(i) for i in x2]

refImage = "PICS_12.05/align_form3/aligned{0}.jpg".format(number)
img = cv2.imread(refImage, cv2.IMREAD_COLOR)

recognized_json = []
text_mas = []
new_json = []

names = ["Name", "Make", "Model", "Color", "VIN", "State", "Location", "Plate"]

for i in range(len(y1)):
    start_row, start_col = y1[i], x1[i]
    end_row, end_col = y2[i], x2[i]

    # cropped = img[start_row:end_row, start_col:end_col]


    cropped_image = "PICS_12.05/align_form3/cropped{1}/cropped_{0}.jpg".format(i, number)
    cv2.imwrite(cropped_image, img[start_row:end_row, start_col:end_col])
    text = pytesseract.image_to_string(Image.open(cropped_image))
    text_mas.append(text)
    recognized_json.append({names[i] : text})




    # text_mas.append(text)


for i in range(len(names)):
    new_json.append({"title" : names[i], "value" : text_mas[i]})



def writeToJSONFile(path, file_name, data):
    file_name_extension = './' + path + '/' + file_name + '.txt'
    with open(file_name_extension, 'w') as file:
        json.dump(data, file)


path = './PICS_12.05/align_form3/cropped{0}'.format(number)
file_name = 'recognized_data'

writeToJSONFile(path, file_name, new_json)



# print(new_json)


# print(recognized_json)
    




# filename = "PICS_11.05/cropped/cropped1.jpg"
# text = pytesseract.image_to_string(Image.open(filename))





# start_row, start_col = 650, 137
# end_row, end_col = 715, 433

# cropped = img[start_row:end_row, start_col:end_col]


# cropped_image = "PICS_11.05/cropped/cropped_33.jpg"
# cv2.imwrite(cropped_image, cropped)








# cropped = img[y1[0]:y2[0], x1[0]:x2[0]]
# cropped_filename = "PICS_11.05/cropped/cropped_image"
# cv2.imwrite(cropped_filename, cropped)

