"""
This script handles the execution of the Flask Web Server(Web Application + JSON API)
"""

from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
#from flaskext.mysql import MySQL
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import tree 
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
# from googleplaces import GooglePlaces, types, lang 
from flask_socketio import SocketIO
import pandas as pd 
import numpy as np
import pickle
import re
import os
import random
import hashlib 
import bcrypt
import json
import requests
#import nltk
import pybase64
from datetime import date
from sklearn.preprocessing import normalize
import MySQLdb
from datetime import timedelta
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import seaborn as sns
import scipy.stats as stats
import sklearn
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing

from werkzeug.utils import secure_filename
import Predict as pred
#UPLOAD_FOLDER = './static/input'
app = Flask(__name__)

port = int(os.environ.get('PORT', 5000))


# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = 'canada$God7972#'
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Enter your database connection details below
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD	'] ="root"
app.config['MYSQL_DATABASE_DB'] = 'cardio'

# Intialize MySQL
# mysql = MySQL(autocommit=True)
# mysql.init_app(app)
mydb = MySQLdb.connect(host='localhost',user='root',passwd='root',db='cardio')
#app.permanent_session_lifetime = timedelta(minutes=15)

#ecg = ECG()

#Homepage
@app.route('/')
def index():
    if 'loggedin' not in session:
        return render_template('index.html')
    else:
        return home()

#Dashboard
@app.route('/dashboard')
def home():
    # Check if user is loggedin
    print("session===22",session)
    if 'loggedin' in session:
        print("Inside If in dashbord")
        # User is loggedin show them the home page
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM doctors WHERE ID = %s', (session['id'],))
        account = cursor.fetchone()
        
        print("is doctor==",session['isdoctor'])
        return render_template('dashboard.html', account = account, isdoctor=session['isdoctor'])
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))




#Doctor Register
@app.route('/docregister', methods=['GET', 'POST'])
def docregister():
    if 'loggedin' not in session:
    # Output message if something goes wrong...
        msg = ''
        # Check if "username", "password" and "email" POST requests exist (user submitted form)
        if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
            # Create variables for easy access
            username = request.form['username']
            password = request.form['password']
            email = request.form['email']
            full_name = request.form['full_name']
            registration_number = request.form['registration_number']
            contact_number = request.form['contact_number']
            spec = request.form['specialization']
            address = request.form['address']
            if(username and password and email and full_name and registration_number and contact_number and spec and address):
            # Check if account exists using MySQL
                cursor = mydb.cursor()
                cursor.execute('SELECT * FROM doctors WHERE Username = %s', (username,))
                account = cursor.fetchone()
                # If account exists show error and validation checks
                if account:
                    msg = 'Account already exists!'
                    flash(msg)
                elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
                    msg = 'Invalid email address!'
                    flash(msg)
                elif not re.match(r'[A-Za-z0-9]+', username):
                    msg = 'Username must contain only characters and numbers!'
                    flash(msg)
                else:
                    # Account doesnt exists and the form data is valid, now insert new account into users table
                    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
                    cursor.execute('INSERT INTO doctors VALUES (NULL, %s, %s, %s, %s, %s, %s, %s, %s, %s)', ( username, hashed_password, email, full_name, registration_number, contact_number, "Default Hospital" , spec, address ))
                    mydb.commit()
                    msg = 'You have successfully registered!'
                    cursor.execute('SELECT * FROM doctors WHERE Username = %s', (username,))
                    # Fetch one record and return result
                    account = cursor.fetchone()
                    session['loggedin'] = True
                    session['id'] = account[0]
                    session['username'] = account[1]
                    session['isdoctor'] = 1
                    return home()
            else:
                msg = 'Please fill out the form!'
                flash(msg)
        elif request.method == 'POST':
            # Form is empty... (no POST data)
            msg = 'Please fill out the form!'
    else:
        return home()
    # Show registration form with message (if any)
    return render_template('doctorlogin.html', msg=msg)

#Doctor Login
@app.route('/doclogin', methods=['GET', 'POST'])
def doclogin():
    if 'loggedin' not in session:
    # Output message if something goes wrong...
        msg = ''
        # Check if "username" and "password" POST requests exist (user submitted form)
        if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
            # Create variables for easy access
            username = request.form['username']
            password = request.form['password']
            if(username and password):

                # Check if account exists using MySQL
                cursor = mydb.cursor()
                cursor.execute('SELECT * FROM doctors WHERE Username = %s', (username,))
                # Fetch one record and return result
                account = cursor.fetchone()
                # If account exists in accounts table in out database
                if account:
                    if bcrypt.checkpw(password.encode('utf-8'), account[2].encode('utf-8')):
                        # Create session data, we can access this data in other routes
                        session['loggedin'] = True
                        session['id'] = account[0]
                        session['username'] = account[1]
                        session['isdoctor'] = 1
                        # Redirect to home page
                        print("session==",session)
                        return home()
                    else:
                        # Account doesnt exist or username/password incorrect
                        msg = 'Incorrect username/password!'
                        flash(msg)
                else:
                    # Account doesnt exist or username/password incorrect
                    msg = 'Incorrect username/password!'
                    flash(msg)
            else:
                msg = 'Please provide both username and password!'
                flash(msg)
    else:
        return home()
    # Show the login form with message (if any)
    return render_template('doctorlogin.html', msg=msg)


# Diagnose Based on the Cardiovascular problems
@app.route('/diagnosecardio',methods=['GET','POST'])
def diagnosecardio():
    # Check if user is loggedin
    if 'loggedin' in session:
        cursor = mydb.cursor()
        if session["isdoctor"]:
            cursor.execute('SELECT * FROM doctors WHERE ID = %s', (session['id'],))
        account = cursor.fetchone()
        
        

        
        if(request.method == 'POST'):
            gender = request.form['Gender']
            height = request.form['Height']
            weight= request.form['weight'] # in kilograms
            systolicbloodpressure= request.form['Sys'] # Systolic blood pressure
            diastolicbloodpressure= request.form['Dys'] # Diastolic blood pressure
            cholesterol= request.form['Chol'] # 1: normal, 2: above normal, 3: well above normal
            gluc= request.form['Gluc'] # 1: normal, 2: above normal, 3: well above normal
            smoke= request.form['Smoke'] # 1 if you smoke, 0 if not
            alco= request.form['Alco'] # 1 if you drink alcohol, 0 if not
            active= request.form['Active'] # 1 if you do physical activity, 0 if not
            esb=request.form['esb']
            bmi=request.form['BMI']
            hr=request.form['hr']
            gul=request.form['gul']


            

            res,treat,test_accuracy=pred.process(float(gender), float(height), float(weight), float(systolicbloodpressure), float(diastolicbloodpressure), float(cholesterol), float(gluc), float(smoke), float(alco),float(active),float(esb),float(bmi),float(hr),float(gul))

           

            
            

            return render_template('cardioanswer.html',ans=res,treat=treat,account=account,test_accuracy=test_accuracy)
        else:
            return render_template('cardiodetails.html',account=account)
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))


# Account information visible inside dashboard
@app.route('/myaccount')
def myaccount():
    if 'loggedin' in session:
        cursor = mydb.cursor()
        if session["isdoctor"]:
            cursor.execute('SELECT * FROM doctors WHERE ID = %s', (session['id'],))
        else:
            cursor.execute('SELECT * FROM users WHERE ID = %s', (session['id'],))
        account = cursor.fetchone()
        return render_template('myaccount.html', account=account, isDoctor = session["isdoctor"])
    else:
        return redirect(url_for('login'))
@app.route('/eprofile')
def eprofile():
    if 'loggedin' in session:
        cursor = mydb.cursor()
        if session["isdoctor"]:
            cursor.execute('SELECT * FROM doctors WHERE ID = %s', (session['id'],))
        else:
            cursor.execute('SELECT * FROM users WHERE ID = %s', (session['id'],))
        account = cursor.fetchone()
        hashed_password=bcrypt.checkpw(account[2].encode('utf-8'),bcrypt.gensalt())
        return render_template('editaccount.html', account=account, isDoctor = session["isdoctor"],hashed_password=hashed_password)
    else:
        return redirect(url_for('login'))
@app.route('/editprofile',methods=['GET','POST'])
def editprofile():
    if 'loggedin' in session:
    # Output message if something goes wrong...
        msg = ''
        # Check if "username", "password" and "email" POST requests exist (user submitted form)
        if request.method == 'POST' and 'uid' in request.form and 'username' in request.form and 'password' in request.form and 'email' in request.form:
            # Create variables for easy access
            uid=request.form['uid']
            username = request.form['username']
            password = request.form['password']
            email = request.form['email']
            full_name = request.form['full_name']
            registration_number = request.form['registration_number']
            contact_number = request.form['contact_number']
            spec = request.form['specialization']
            address = request.form['address']
            if(uid and username and password and email and full_name and registration_number and contact_number and spec and address):
            # Check if account exists using MySQL
                cursor = mydb.cursor()
                cursor.execute('SELECT * FROM doctors WHERE ID = %s', (uid,))
                account = cursor.fetchone()
                # If account exists show error and validation checks
                if account:
                    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
                    cursor.execute('update doctors set Username=%s,Password=%s,Email=%s,Full_Name=%s,Registration_Number=%s,Contact_Number=%s,Hospital_Name=%s,Specialization=%s,Address=%s where Id=%s', ( username, hashed_password, email, full_name, registration_number, contact_number, "Default Hospital" , spec, address,int(uid)))
                    mydb.commit()
                    msg = 'You have Edited successfully!'
                    cursor.execute('SELECT * FROM doctors WHERE Username = %s', (username,))
                    # Fetch one record and return result
                    account = cursor.fetchone()
                    session['loggedin'] = True
                    session['id'] = account[0]
                    session['username'] = account[1]
                    session['isdoctor'] = 1
                    return home()
                    
                elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
                    msg = 'Invalid email address!'
                    flash(msg)
                elif not re.match(r'[A-Za-z0-9]+', username):
                    msg = 'Username must contain only characters and numbers!'
                    flash(msg)
            else:
                msg = 'Please fill out the form!'
                flash(msg)
        elif request.method == 'POST':
            # Form is empty... (no POST data)
            msg = 'Please fill out the form!'
    else:
        return home()
    # Show registration form with message (if any)
    return render_template('doctorlogin.html', msg=msg)
@app.route("/graph")
def graph():
    if 'loggedin' in session:
        cursor = mydb.cursor()
        if session["isdoctor"]:
            cursor.execute('SELECT * FROM doctors WHERE ID = %s', (session['id'],))
        account = cursor.fetchone()
        return render_template("Graph.html",account=account, isDoctor = session["isdoctor"])
    





socketio = SocketIO(app)


# http://localhost:5000/logout - this will be the logout page
@app.route('/logout')
def logout():
   # Remove session data, this will log the user out
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   # Redirect to login page
   return redirect(url_for('index'))

#run the Flask Server
if __name__ == '__main__':
	socketio.run(app, debug=True)
    
"""-------------------------------End of Web Application-------------------------------"""
