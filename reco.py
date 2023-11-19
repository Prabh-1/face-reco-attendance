from PIL import Image

import numpy as np
from numpy import asarray
from numpy import expand_dims
from keras_facenet import FaceNet
from test import test

import pickle
import cv2
from scipy.spatial.distance import cosine
import sys
# Iterate over database embeddings
from datetime import datetime,timedelta
import mysql.connector
mysql_connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Pr@bh123",
    database="face_att"
)

selected_course = sys.argv[1] if len(sys.argv) > 1 else None
print(selected_course)


HaarCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
MyFaceNet = FaceNet()

myfile = open("data.pkl", "rb")
database = pickle.load(myfile)
myfile.close()


cursor = mysql_connection.cursor()

cap = cv2.VideoCapture(0)


def Markattendance():
    # Get the current date and time
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    conn = mysql.connector.connect(host='localhost', username='root', password='Pr@bh123', database='face_att')
    my_cursor = conn.cursor()

    # Check if attendance is already marked for the course in the last half an hour
    my_cursor.execute('SELECT * FROM attendance_details WHERE urn=%s AND Course=%s AND Date=%s AND Time > %s',
                      (urn, selected_course, date, (now - timedelta(minutes=30)).strftime("%H:%M:%S")))
    result = my_cursor.fetchone()

    if result:
        print(f"Attendance already marked for {urn} and Course {selected_course} in the last half an hour.")
    else:
        # Fetch the result of the MAX(total_attendance) query
        my_cursor.execute('SELECT MAX(total_attendance) FROM attendance_details WHERE urn=%s AND Course=%s',
                  (urn, selected_course))
        max_attendance_result = my_cursor.fetchone()

        if max_attendance_result and max_attendance_result[0] is not None:
            max_attendance = max_attendance_result[0] + 1
        else:
            max_attendance = 1

        print(urn, student_name, student_year, section, selected_course, date, time, max_attendance)

# Insert the attendance record
        sql = "INSERT INTO attendance_details (URN, Name, Year, Section, Course, Date, Time, total_attendance) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
        values = (urn, student_name, student_year, section, selected_course, date, time, max_attendance)

        my_cursor.execute(sql, values)
        conn.commit()  # Commit changes to the database
        print("Attendance marked successfully.")

    my_cursor.close()





        
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

            for urn, db_embedding in database.items():
                db_embedding_normalized = normalize(db_embedding.reshape(1, -1))
                signature_normalized = normalize(signature.reshape(1, -1))

                similarity = cosine_similarity(signature_normalized, db_embedding_normalized)

                threshold = 0.75

                if similarity > threshold:
                    cursor.execute('SELECT name, year,section FROM student_details WHERE URN = %s', (urn,))
                    student_data = cursor.fetchone()
                    # insert_query = "INSERT INTO attendance_details (URN, Name, Year, Course, Date, Time, total_attendance) VALUES (%s, %s, %s, %s, %s, %s, %s)"
                    
                    if student_data:
                
                        student_name, student_year,section = student_data
                        
                    Markattendance()       
                 
                    print(f"Match found for {urn} with similarity: {similarity}")
                    cv2.putText(img, urn, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
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

cursor.close()
mysql_connection.close()