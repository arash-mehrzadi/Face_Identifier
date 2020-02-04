# coding: utf-8
#import Required libraries
#  cv2: is _OpenCV_ module for Python which I use for face detection and face recognition.
#  numpy: I use this module to convert Python lists to numpy arrays as OpenCV face recognizers accept numpy arrays.
#  pickle : I use this python module to Serialize images to store as file .
import numpy as np
import cv2
import pickle
#you should import haarcascade_frontalface_alt2.xml file location in your runtime path. 
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
#This variable is set to learn the faces that have been detected by face_cascade
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
#Read train_face.yml file that be achieved from training phase. 
face_recognizer.read("train_face.yml")
#define persons label lib.
labels = {"person_name":1}
#open the lables lib in binary format ("rb") for Reading 
#in this section load input_lables items and use them them to create labels lib in key_value format.
with open("lables.pickle", "rb") as f:
    input_labels = pickle.load(f)
    labels = {v:k for k,v in input_labels.items()}
#set VideoCapture id=0 to use webcam captures.
cpture_in =cv2.VideoCapture(0)

while(True):
    #capture frame by frame.
    ret, frame =cpture_in.read()
    #translating color to gray ! thats the way its working. 
    cvt_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Detect faces
    faces = face_cascade.detectMultiScale(cvt_frame, scaleFactor=1.5, minNeighbors=5)
    #in this section I set target (face) position and predict target label via using train_face.yml file which trained in training_faces phase.
    for (x, y, w, h) in faces:
        print(x, y, w, h)
        target_gray = gray [y:y+h, x:x+w]
        target_color = frame [y:y+h, x:x+w]
        Targetid, Rate = face_recognizer.predict(target_gray)
        #after prediction I use the Rate variable to make a decision about choose labels.
        #I set my function to higher than 50% but you can change this parameter depend on the quality of the images that you use for learning .yml file in training phase.
        if Rate>= 50 :
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[Targetid]
            #BGR format
            color =(255,255,255)
            stroke = 2
            #put the chosen label on the target frame. 
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
        else :
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = "person"
            #BGR format
            color =(255,255,255)
            stroke = 2
            #put the chosen label on the target frame. 
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
        #BGR format
        color = (255, 0, 0)
        stroke = 2
        #set Rectangle around of target.
        target_x = x+w
        target_y = y+h
        cv2.rectangle(frame, (x,y) , (target_x, target_y), color, stroke)
        
    #display the resulting frame
    cv2.imshow('frame',frame)
    #set waitKey to exit from webcam window.
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
        
#when everything done, release the capture
cap.release()
cv2.destroyAllWindows()
