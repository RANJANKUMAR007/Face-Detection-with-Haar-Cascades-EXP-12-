# Face Detection using Haar Cascades with OpenCV and Matplotlib
## Name Ranjan Kumar G
# Reg no:212223240138
## Aim

To write a Python program using OpenCV to perform the following image manipulations:  
i) Extract ROI from an image.  
ii) Perform face detection using Haar Cascades in static images.  
iii) Perform eye detection in images.  
iv) Perform face detection with label in real-time video from webcam.

## Software Required

- Anaconda - Python 3.7 or above  
- OpenCV library (`opencv-python`)  
- Matplotlib library (`matplotlib`)  
- Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)

## Algorithm

### I) Load and Display Images

- Step 1: Import necessary packages: `numpy`, `cv2`, `matplotlib.pyplot`  
- Step 2: Load grayscale images using `cv2.imread()` with flag `0`  
- Step 3: Display images using `plt.imshow()` with `cmap='gray'`

### II) Load Haar Cascade Classifiers

- Step 1: Load face and eye cascade XML files 
### III) Perform Face Detection in Images

- Step 1: Define a function `detect_face()` that copies the input image  
- Step 2: Use `face_cascade.detectMultiScale()` to detect faces  
- Step 3: Draw white rectangles around detected faces with thickness 10  
- Step 4: Return the processed image with rectangles  

### IV) Perform Eye Detection in Images

- Step 1: Define a function `detect_eyes()` that copies the input image  
- Step 2: Use `eye_cascade.detectMultiScale()` to detect eyes  
- Step 3: Draw white rectangles around detected eyes with thickness 10  
- Step 4: Return the processed image with rectangles  

### V) Display Detection Results on Images

- Step 1: Call `detect_face()` or `detect_eyes()` on loaded images  
- Step 2: Use `plt.imshow()` with `cmap='gray'` to display images with detected regions highlighted  

### VI) Perform Face Detection on Real-Time Webcam Video

- Step 1: Capture video from webcam using `cv2.VideoCapture(0)`  
- Step 2: Loop to continuously read frames from webcam  
- Step 3: Apply `detect_face()` function on each frame  
- Step 4: Display the video frame with rectangles around detected faces  
- Step 5: Exit loop and close windows when ESC key (key code 27) is pressed  
- Step 6: Release video capture and destroy all OpenCV windows  

# program
```
import cv2
import matplotlib.pyplot as plt
import numpy as np

w_glass = cv2.imread('ME.jpg', cv2.IMREAD_GRAYSCALE)
wo_glass = cv2.imread('ME.jpg', cv2.IMREAD_GRAYSCALE)
group = cv2.imread('MAX.jpg', cv2.IMREAD_GRAYSCALE)

w_glass1 = cv2.resize(w_glass, (1000, 1000))
wo_glass1 = cv2.resize(wo_glass, (1000, 1000)) 
group1 = cv2.resize(group, (1000, 1000))

plt.figure(figsize=(15,10))

plt.subplot(1,3,1)
plt.imshow(w_glass1, cmap='gray')
plt.title('With Glasses')
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(wo_glass1, cmap='gray')
plt.title('Without Glasses')
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(group1, cmap='gray')
plt.title('Group Image')
plt.axis('off')

plt.show()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_and_display(image):
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 10)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()


import cv2
from matplotlib import pyplot as plt

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if face_cascade.empty():
    print("Error: Cascade file not loaded properly!")
else:
    print("Cascade loaded successfully.")

w_glass1 = cv2.imread('image1.png')  # <-- replace with your image filename

if w_glass1 is None:
    print("Error: Image not found. Check the filename or path.")
else:
    print("Image loaded successfully.")

def detect_and_display(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
    return image

if w_glass1 is not None and not face_cascade.empty():
    result = detect_and_display(w_glass1)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

```
# output
<img width="1389" height="474" alt="image" src="https://github.com/user-attachments/assets/2947e2f2-833a-4344-9854-64e09bb36ad1" />
<img width="664" height="513" alt="image" src="https://github.com/user-attachments/assets/1ac83c37-c636-4384-a7bd-29b9d7a6787f" />
<img width="704" height="520" alt="image" src="https://github.com/user-attachments/assets/822a1a14-5ef0-4e09-ada7-0ba06e88150e" />
<img width="646" height="516" alt="image" src="https://github.com/user-attachments/assets/e161dcef-1c74-433c-9ddc-437ab7ad6a69" />
<img width="684" height="392" alt="image" src="https://github.com/user-attachments/assets/9fdbb3d3-a2d3-4824-a863-421e1ab7db0a" />
<img width="674" height="398" alt="image" src="https://github.com/user-attachments/assets/3cc4cbc2-943d-481c-bc00-107be4a1a84e" />
<img width="723" height="445" alt="image" src="https://github.com/user-attachments/assets/a1a13086-5d0d-407e-b8c3-152ff68196fb" />


# result
Thus the output for Face Detection using Haar Cascades with OpenCV and Matplotlib is successfully displayed 
