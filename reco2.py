from PIL import Image
import numpy as np
from numpy import asarray
from numpy import expand_dims
from keras_facenet import FaceNet
import pickle
import cv2
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import sys
# Initialize FaceNet model
MyFaceNet = FaceNet()
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

selected_course = sys.argv[2] if len(sys.argv) > 2 else None
cursor = mysql_connection.cursor()

def Markattendance(urn):
    # Get the current date and time
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    conn = mysql.connector.connect(host='localhost', username='root', password='Pr@bh123', database='face_att')
    my_cursor = conn.cursor()
    my_cursor.execute('SELECT MAX(total_attendance) FROM attendance_details WHERE urn=%s AND Course=%s ',
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

# Load the database of embeddings
with open("data.pkl", "rb") as myfile:
    database = pickle.load(myfile)
if len(sys.argv) > 1:
    img_path = sys.argv[1]
else:
    img_path = 'group5.jpeg'
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
    urn = "Unknown"

    for name, database_embedding in database.items():
        database_embedding_flat = database_embedding.flatten()
        # Compute cosine similarity
        distance = cosine(database_embedding_flat, detected_embedding_flat)

        # Update recognized name if this is the closest match so far
        if distance < min_distance:
            min_distance = distance
            urn = name
         
            

            
                
                

    # Draw bounding box
    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Adjust text position to prevent clustering
    text_x = x1
    text_y = y2 + 40 + i * 20  # Adjust the spacing between names

    # Draw the recognized name
    plt.text(text_x, text_y, urn, color='b', backgroundcolor='w')
    
    cursor.execute('SELECT name, year,section FROM student_details WHERE URN = %s', (urn,))
    student_data = cursor.fetchone()

    if student_data:
                student_name, student_year ,section= student_data
    Markattendance(urn)

# Display the image with bounding boxes and recognized names
plt.imshow(img_rgb)
plt.axis('off')
plt.show()