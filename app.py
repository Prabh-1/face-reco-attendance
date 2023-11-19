from flask import Flask, render_template, request, redirect, url_for, session,flash
from flask_mysqldb import MySQL
import subprocess
app = Flask(__name__)
import os

# Secret key for sessions. You should change this to a long, random string in a real application.
app.secret_key = 'your_secret_key'

# Configure MySQL
app.config['MYSQL_HOST'] = 'localhost'  # Your MySQL host
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Pr@bh123'
app.config['MYSQL_DB'] = 'face_att'  # The name of the MySQL database
mysql = MySQL(app)

@app.route('/', methods=['GET', 'POST'])
def index():
    msg = ''

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        cursor = mysql.connection.cursor()
        cursor.execute('SELECT * FROM users WHERE username = %s AND password = %s', (username, password))
        user = cursor.fetchone()

                # ...

        if user:
            session['username'] = user[0]
            cursor.close()
            cursor = mysql.connection.cursor()

            if username.startswith('1'):
                cursor.execute('SELECT name FROM Teachers_details WHERE Teacher_Id = %s', (username,))
            else:
                cursor.execute('SELECT name, year FROM student_details WHERE URN = %s', (username,))

            user_data = cursor.fetchone()

            if user_data:
                session['name'] = user_data[0]  # Access the first element of the tuple

                if not username.startswith('1'):
                    session['student_year'] = user_data[1]  # Assuming the year is in the second column of the student_details table

                cursor.close()

                if username.startswith('1'):
                    return redirect(url_for('year'))
                else:
                    return redirect(url_for('course'))
            else:
                msg = 'Error fetching user data'
                cursor.close()
        else:
            msg = 'Incorrect Username or Password'
            cursor.close()

        # ...


    return render_template('login.html', msg=msg)

@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    session.pop('access',None)
    # Redirect to login page
    return redirect(url_for('index'))

@app.route('/select_year/<selected_year>')
def select_year(selected_year):
    session['selected_year'] = selected_year
    return redirect(url_for('course'))

@app.route('/year')
def year():
    # Render the page for teachers
    return render_template('year.html', username=session.get('name'))


# ...

@app.route('/select_course/<selected_course>')
def select_course(selected_course):
    session['selected_course'] = selected_course
    return redirect(url_for('markncheck'))

@app.route('/select_course2/<selected_course>')
def select_course2(selected_course):
    session['selected_course'] = selected_course
    return redirect(url_for('Check2'))

@app.route('/courses')
def course():
    if 'student_year' in session:
        selected_year = session['student_year']
    elif 'selected_year' in session:
        selected_year = session['selected_year']
    else:
        # Handle the case where neither student_year nor selected_year is in the session
        return redirect(url_for('year'))

    cursor = mysql.connection.cursor()
    cursor.execute('SELECT CourseName FROM Courses WHERE Year = %s', (selected_year,))
    courses = cursor.fetchall()
    cursor.close()
    courses = list(courses)
    is_teacher = str(session.get('username', '')).startswith('1')

    return render_template('courses.html', username=session.get('name'), courses=courses, selected_year=selected_year,is_teacher=is_teacher)


@app.route('/markncheck')
def markncheck():
    selected_course = session.get('selected_course')
    if not selected_course:
        # Handle the case where selected_course is not in the session
        return redirect(url_for('courses'))

    # Add any additional logic you need for the markncheck route

    return render_template('markncheck.html', username=session.get('name'), selected_course=selected_course)

@app.route('/Options')
def Options():
    
    return render_template('Options.html',username=session.get('name'))

@app.route('/live_attendance')
def live_attendance():
    selected_course = session.get('selected_course', None)

    # Run reco.py as a separate process
    subprocess.Popen(['/usr/bin/python3', 'reco.py',selected_course])
    flash('Attendance marked successfully!')
    return render_template('Options.html',username=session.get('name'))

@app.route('/upload_photo', methods=['GET', 'POST'])
def upload_photo():
    selected_course = session.get('selected_course', None)
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file:
            # Save the uploaded file to a specific folder
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)

            # Add logic here to process the uploaded file as needed
            subprocess.Popen(['/usr/bin/python3', 'reco2.py', file_path,selected_course])
            flash('Attendance marked successfully!')
            return redirect(url_for('Options'))

        

    return render_template('upload_photo.html', username=session.get('name'))

# ... (previous code remains unchanged) ...

@app.route('/Check', methods=['GET', 'POST'])
def Check():
    if request.method == 'POST':
        course = session.get('selected_course')
    
        cursor = mysql.connection.cursor()
        date = request.form['date']
        username = str(session.get('username', ''))

        if username.startswith('1'):
            # Teacher view: Fetch all students for the selected year
            selected_year = session.get('selected_year')
            print(selected_year)
            cursor.execute('SELECT URN, Name FROM student_details WHERE year = %s', (selected_year,))
            student_details = cursor.fetchall()
            print(student_details)

            # Create a list to store attendance records
            records = []

            for student in student_details:
                urn, name = student
                # Check if attendance is marked for the student on the specified date and course
                cursor.execute('SELECT COUNT(*) FROM attendance_details WHERE URN = %s AND Course = %s AND Date = %s', (urn, course, date))
                attendance_count = cursor.fetchone()[0]
                print(attendance_count)
                
                # Determine attendance status (present or absent)
                status = 'Present' if attendance_count > 0 else 'Absent'
                cursor.execute('Select max(total_attendance) from attendance_details where urn=%s and course=%s',(urn,course))
                total_attendance=cursor.fetchone()[0]
                if total_attendance is not None:
                    total_attendance=total_attendance 
                else:
                    total_attendance=0
                
                # Append the record to the list
                records.append({'URN': urn, 'Name': name, 'Status': status , 'Total Attendance' : total_attendance})
                print(records)

            cursor.close()

            return render_template('attendance.html', username=session.get('name'), course=course, date=date, records=records)
        
    
            


    # Handle the GET request
    return render_template('attendance.html', username=session.get('name'))


@app.route('/Check2', methods=['GET'])
def Check2():
    if request.method == 'GET':
        course = session.get('selected_course')
        print(course)
        cursor = mysql.connection.cursor()
        urn = session.get('username', '')

        cursor.execute('SELECT date, time FROM attendance_details WHERE urn = %s AND course = %s', (urn, course))
        result = cursor.fetchall()
        print(result)

        # Create a list of dictionaries, each representing a record (date and time)
        records = [{'Date': record[0], 'Time': record[1]} for record in result]
        print(records)
        
        cursor.execute('Select max(total_attendance) from attendance_details where urn= %s AND course = %s', (urn, course))
        att_score=cursor.fetchone()[0]
        

        # Close the cursor
        cursor.close()

        return render_template('attendance2.html', username=session.get('name'), course=course, urn=urn, records=records,att_score=att_score)

    # Handle other HTTP methods
    else:
        return "Method Not Allowed"

# ... (remaining code remains unchanged) ...


if __name__ == '__main__':
    app.run()
