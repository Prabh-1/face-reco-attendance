from PIL import Image
import numpy as np
from numpy import asarray
from numpy import expand_dims
from keras_facenet import FaceNet
import pickle
import cv2
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

# Initialize FaceNet model
MyFaceNet = FaceNet()

# Load the database of embeddings
with open("data.pkl", "rb") as myfile:
    database = pickle.load(myfile)

img_path = 'group2.jpeg'
img = cv2.imread(img_path)

# Detect faces using Haar Cascade
HaarCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img_faces = HaarCascade.detectMultiScale(img, 1.1, 4)

# Convert BGR image to RGB for displaying with Matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Process detected faces and draw bounding boxes
for i, (x1, y1, width, height) in enumerate(img_faces):
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height

    face_img_rgb = img_rgb[y1:y2, x1:x2]  # Extract the face region
    face_pil = Image.fromarray(face_img_rgb)  # Convert to PIL Image
    face_resized = face_pil.resize((160, 160))  # Resize to match FaceNet input size
    face_array = asarray(face_resized)  # Convert back to numpy array

    # Expand dimensions for model input
    face_array_expanded = expand_dims(face_array, axis=0)

    # Compute the embedding for the face
    detected_embedding = MyFaceNet.embeddings(face_array_expanded)
    detected_embedding_flat = detected_embedding.flatten()

    # Compare with database embeddings
    min_distance = 100
    recognized_name = "Unknown"

    for name, database_embedding in database.items():
        database_embedding_flat = database_embedding.flatten()
        # Compute cosine similarity
        distance = cosine(database_embedding_flat, detected_embedding_flat)

        # Update recognized name if this is the closest match so far
        if distance < min_distance:
            min_distance = distance
            recognized_name = name

    # Draw bounding box
    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Adjust text position to prevent clustering
    text_x = x1
    text_y = y2 + 40 + i * 20  # Adjust the spacing between names

    # Draw the recognized name
    plt.text(text_x, text_y, recognized_name, color='b', backgroundcolor='w')

# Display the image with bounding boxes and recognized names
plt.imshow(img_rgb)
plt.axis('off')
plt.show()