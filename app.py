from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
import hashlib

app = Flask(__name__)

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

        if user:
            session['username'] = user[1]
            cursor.close()
            return redirect(url_for('index'))
        else:
            msg = 'Incorrect Username or Password'
            cursor.close()


    return render_template('login.html', msg=msg)

if __name__ == '__main__':
    app.run()
