# Import Libraries
import face_recognition as fr
import numpy as np
import cv2
import pickle

# Import Saved_Encodings
with open('Year_11.dat', 'rb') as f:
    data = pickle.load(f)
names = list(data.keys())
encodings = np.array(list(data.values()))
print(names)

def show_face(name,bool):
    if bool == False:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
        cv2.rectangle(img, (x1, y2-35), (x2, y2), (0,0,255), cv2.FILLED)
        cv2.putText(img, "UNKNOWN", (x1+6, y2-6), (cv2.FONT_HERSHEY_SIMPLEX), 1, (255,255,255), 2)
    else:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.rectangle(img, (x1, y2-35), (x2, y2), (0,255,0), cv2.FILLED)
        cv2.putText(img, name, (x1+6, y2-6), (cv2.FONT_HERSHEY_SIMPLEX), 1, (0,0,0), 2)

# Find Faces
cap = cv2.VideoCapture(0)

while True:
    # Capturing Every Frame of Video
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Locating and Finding Encodings of Face in the Frame
    facesCurFrame = fr.face_locations(imgS)
    encodingsCurFrame = fr.face_encodings(imgS, facesCurFrame)
    
    # Comparing Faces Located with Faces Already Recognized
    for encodeFace, faceLoc in zip(encodingsCurFrame, facesCurFrame):
        matches_cv = fr.compare_faces(encodings, encodeFace, 0.45)
        matches_attendance = fr.compare_faces(encodings, encodeFace, 0.35)
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
        if True in matches_attendance:
            faceDistance = fr.face_distance(encodings, encodeFace)
            matchIndex = np.argmin(faceDistance)
            name = names[matchIndex].upper()
            show_face(name,True)
        elif True in matches_cv:
            faceDistance = fr.face_distance(encodings, encodeFace)
            matchIndex = np.argmin(faceDistance)
            name = names[matchIndex].upper()
            show_face(name,True)            
        else:
            show_face("UNKNOWN",False)

    cv2.imshow("Webcam",img)
    cv2.waitKey(1)
