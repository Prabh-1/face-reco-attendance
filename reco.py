from PIL import Image

import numpy as np
from numpy import asarray
from numpy import expand_dims
from keras_facenet import FaceNet
from test import test

import pickle
import cv2
from scipy.spatial.distance import cosine

# Iterate over database embeddings




HaarCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
MyFaceNet = FaceNet()

myfile = open("data.pkl", "rb")
database = pickle.load(myfile)
myfile.close()

# ... (previous code remains unchanged) ...

cap = cv2.VideoCapture(0)

while(1):
    _, img = cap.read()
    label = test(
        image=img,
        model_dir='resources/anti_spoof_models',
        device_id=0
    )
    if label == 1:
        img1 = HaarCascade.detectMultiScale(img, 1.1, 4)

        for (x1, y1, width, height) in img1:
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height

            img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img2 = Image.fromarray(img2)  # konversi dari OpenCV ke PIL
            img_array = asarray(img2)

            face = img_array[y1:y2, x1:x2]

            face = Image.fromarray(face)
            face = face.resize((160, 160))
            face = asarray(face)

            face = expand_dims(face, axis=0)
            signature = MyFaceNet.embeddings(face)
            signature_flat = signature.flatten()

            from sklearn.metrics.pairwise import cosine_similarity
            from sklearn.preprocessing import normalize
            identity = ' '

            for name, db_embedding in database.items():
                db_embedding_normalized = normalize(db_embedding.reshape(1, -1))
                signature_normalized = normalize(signature.reshape(1, -1))

                similarity = cosine_similarity(signature_normalized, db_embedding_normalized)

                threshold = 0.75

                if similarity > threshold:
                    print(f"Match found for {name} with similarity: {similarity}")
                    cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                                cv2.LINE_AA)
                else:
                    print("No match found.")

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    else:
        cv2.putText(img, 'unknown', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('res', img)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()