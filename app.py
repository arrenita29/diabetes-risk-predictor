from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import tensorflow as tf
import pandas as pd
import psycopg2
import os
from datetime import datetime

app = Flask(__name__)

# ---------- Lazy-loaded models ----------
_readmission_model = None
_readmission_scaler = None
_readmission_columns = None

_screening_model = None
_screening_scaler = None
_screening_columns = None

def get_readmission_model():
    global _readmission_model, _readmission_scaler, _readmission_columns
    if _readmission_model is None:
        _readmission_model = tf.keras.models.load_model('diabetes_model.keras')
        _readmission_scaler = joblib.load('scaler.pkl')
        _readmission_columns = joblib.load('model_columns.pkl')
    return _readmission_model, _readmission_scaler, _readmission_columns

def get_screening_model():
    global _screening_model, _screening_scaler, _screening_columns
    if _screening_model is None:
        _screening_model = tf.keras.models.load_model('diabetes_screening_model.keras')
        _screening_scaler = joblib.load('scaler_screening.pkl')
        _screening_columns = joblib.load('screening_columns.pkl')
    return _screening_model, _screening_scaler, _screening_columns

# ---------- Database (Supabase / Postgres) ----------
DB_HOST = os.environ.get('SUPABASE_HOST', 'db.spcbhcujcsldcryqlanr.supabase.co')
DB_PORT = os.environ.get('SUPABASE_PORT', '5432')
DB_NAME = os.environ.get('SUPABASE_DB', 'postgres')
DB_USER = os.environ.get('SUPABASE_USER', 'postgres')
DB_PASSWORD = os.environ.get('SUPABASE_PASSWORD')  # set this in Render's environment variables, never hardcoded

def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASSWORD
    )

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS records (
            id SERIAL PRIMARY KEY,
            prediction_type TEXT,
            patient_name TEXT,
            summary TEXT,
            result TEXT,
            probability REAL,
            timestamp TEXT
        )
    ''')
    conn.commit()
    c.close()
    conn.close()

def save_record(prediction_type, patient_name, summary, result, probability):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        'INSERT INTO records (prediction_type, patient_name, summary, result, probability, timestamp) VALUES (%s, %s, %s, %s, %s, %s)',
        (prediction_type, patient_name, summary, result, probability, datetime.now().strftime('%Y-%m-%d %H:%M'))
    )
    conn.commit()
    c.close()
    conn.close()

def get_stats():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM records')
    total = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM records WHERE result LIKE %s", ('%High%',))
    high_risk = c.fetchone()[0]
    c.close()
    conn.close()
    return total, high_risk

init_db()

@app.route('/')
def home():
    return redirect(url_for('readmission'))

@app.route('/readmission', methods=['GET', 'POST'])
def readmission():
    readmission_model, readmission_scaler, readmission_columns = get_readmission_model()

    prediction = None
    probability = None

    if request.method == 'POST':
        patient_name = request.form.get('patient_name', 'Unknown').strip() or 'Unknown'
        time_in_hospital = int(request.form['time_in_hospital'])
        num_lab_procedures = int(request.form['num_lab_procedures'])
        num_medications = int(request.form['num_medications'])
        number_diagnoses = int(request.form['number_diagnoses'])
        number_inpatient = int(request.form['number_inpatient'])

        input_data = pd.DataFrame(np.zeros((1, len(readmission_columns))), columns=readmission_columns)
        input_data['time_in_hospital'] = time_in_hospital
        input_data['num_lab_procedures'] = num_lab_procedures
        input_data['num_medications'] = num_medications
        input_data['number_diagnoses'] = number_diagnoses
        input_data['number_inpatient'] = number_inpatient

        input_scaled = readmission_scaler.transform(input_data)
        prediction_proba = readmission_model.predict(input_scaled)[0][0]
        prediction = "High Risk of Readmission" if prediction_proba > 0.5 else "Low Risk of Readmission"
        probability = round(float(prediction_proba) * 100, 1)

        summary = f"Hospital days: {time_in_hospital}, Lab procedures: {num_lab_procedures}, Medications: {num_medications}"
        save_record('Readmission', patient_name, summary, prediction, probability)

    total, high_risk = get_stats()
    return render_template('readmission.html', prediction=prediction, probability=probability, total=total, high_risk=high_risk)

@app.route('/screening', methods=['GET', 'POST'])
def screening():
    screening_model, screening_scaler, screening_columns = get_screening_model()

    prediction = None
    probability = None

    if request.method == 'POST':
        patient_name = request.form.get('patient_name', 'Unknown').strip() or 'Unknown'
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        pedigree = float(request.form['pedigree'])
        age = float(request.form['age'])

        input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                                     insulin, bmi, pedigree, age]], columns=screening_columns)

        input_scaled = screening_scaler.transform(input_data)
        prediction_proba = screening_model.predict(input_scaled)[0][0]
        prediction = "High Risk of Diabetes" if prediction_proba > 0.5 else "Low Risk of Diabetes"
        probability = round(float(prediction_proba) * 100, 1)

        summary = f"Glucose: {glucose}, BMI: {bmi}, Age: {age}"
        save_record('Screening', patient_name, summary, prediction, probability)

    total, high_risk = get_stats()
    return render_template('screening.html', prediction=prediction, probability=probability, total=total, high_risk=high_risk)

@app.route('/records')
def records():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT patient_name, prediction_type, summary, result, probability, timestamp FROM records ORDER BY id DESC')
    all_records = c.fetchall()
    c.close()
    conn.close()

    total, high_risk = get_stats()
    return render_template('records.html', records=all_records, total=total, high_risk=high_risk)

if __name__ == '__main__':
    app.run(debug=True)