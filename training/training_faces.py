# coding: utf-8
#import Required libraries
#  cv2: is _OpenCV_ module for Python which I use for face detection and face recognition.
#  os: I use this Python module to read training directories and file names.
#  numpy: I use this module to convert Python lists to numpy arrays as OpenCV face recognizers accept numpy arrays.
#  PIL : I use this python module to open , resize and convert Images as desired.
#  pickle : I use this python module to Serialize images to store as file .
import os
import numpy as np
import cv2
from PIL import Image
import pickle
#define training face images Directory
Runtime_DIR = os.path.dirname(os.path.abspath(__file__))
Training_faces_DIR =os.path.join(Runtime_DIR, "images")
#use Haar feature-based cascade classifier to detect faces.
#in this code i use "haarcascade_frontalface_alt2.xml" classifier and you can download that xml file from GitHub.
face_classifier = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
#This variable is set to learn the faces that have been detected by face_cascade
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
#in this section I define some variable  that you will see below.
counter=0   #This counter specifies the number of ids.
label_ids ={}  #This Library contain label ids .
y_labels=[]    #This Array contain label ids .
x_train =[]    #This Array contain Training images that i convert whit numpy array.
#in this section I Read Directory of Images and Initializing library and arrays that i define above.
for root, dirs, files in os.walk(Training_faces_DIR):   #walking on files :))))
    for file in files:                         #walk in all files in all file
        #select all files with .png/.jpg format .
        if file.endswith("png") or file.endswith("jpg"): 
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
            if not label in label_ids:
                label_ids[label] = counter
                counter += 1
            face_ID = label_ids[label]
            #translating color Image to Black&White / Convert to GrayScale
            pil_image = Image.open(path).convert("L")
            #Resizing images to Designated size. 
            size = (550,550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            #transfer images format to numpy array
            image_array = np.array(final_image, "uint8")
            #Detect faces
            faces = face_classifier.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
            #Dfine target (faces) position and put them in to list and append them whit labels to train faces.
            for (x, y, w, h) in faces:
                target = image_array[y:y+h, x:x+w]
                x_train.append(target)
                y_labels.append(face_ID)
#in this section i define all the Required values as you see above and use them to training faces with face_recognizer.
with open("lables.pickle", "wb") as f:
    pickle.dump(label_ids, f)
face_recognizer.train(x_train, np.array(y_labels))
face_recognizer.save("train_face.yml")
#at the end , this functions gives you train_face.ylm file to identify faces that you add in project
