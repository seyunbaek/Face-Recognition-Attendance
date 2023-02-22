# Import Libraries
import face_recognition as fr
import os
import cv2
import pickle
from tqdm import tqdm

# Declare Variables
path = "Images_Final"
images = []
names = []
image_list = os.listdir(path)
encodings = {}

# Bring in Images
for file in image_list:
    curImg = cv2.imread(f"{path}/{file}")
    images.append(curImg)
    names.append(os.path.splitext(file)[0])
print(names)

# Define Functions
def find_encoding(img,name):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encode = fr.face_encodings(img)[0]
    encodings[f"{name}"] = encode

# Find Encodings
for i in tqdm(range(len(names)), desc="Finding Encodings..."):
    img, name = images[i], names[i]
    find_encoding(img,name)
with open('Year_11.dat', 'wb') as f:
    pickle.dump(encodings, f)
print("Encoding Complete!")