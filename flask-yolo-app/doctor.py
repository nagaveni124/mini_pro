import pandas as pd
import numpy as np
import os
from flask import Flask, render_template, request

app = Flask(__name__)

# Define the paths to your CSV files
# medicine_file = "C:\\Users\\nagav\\Desktop\\mini_pro\\flask-yolo-app\\data\\medicine.csv"
# teeth_score_file = "C:\\Users\\nagav\\Desktop\\mini_pro\\flask-yolo-app\\data\\teeth_score.csv"

# Load dataframes
# df_medicine = pd.read_csv(medicine_file)
# df_teeth_score = pd.read_csv(teeth_score_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/medicine')
def medicine():
    detected_classes = request.args.get('detected_classes').split(',')
    suggested_medicines = suggest_medicine(detected_classes)
    return render_template('medicine.html', detected_classes=detected_classes, medicines=suggested_medicines)

@app.route('/doctors')
def doctors():
    health_type = request.args.get('health_type')
    relevant_doctors = get_relevant_doctors(health_type)
    return render_template('doctor.html', doctors=relevant_doctors)

def get_relevant_doctors(health_type):
    # Replace with your logic to fetch doctors based on health_type
    doctors_data = [
        {'name': 'Dr. Smith', 'specialty': 'Dentist'},
        {'name': 'Dr. Johnson', 'specialty': 'Cardiologist'},
        {'name': 'Dr. Brown', 'specialty': 'Dermatologist'}
    ]
    relevant_doctors = [doctor for doctor in doctors_data if doctor['specialty'].lower() == health_type.lower()]
    return relevant_doctors

def suggest_medicine(detected_classes):
    medicine_df = pd.read_csv('C:\\Users\\nagav\\Desktop\\mini_pro\\flask-yolo-app\\data\\medicine.csv')
    suggested_medicines = []

    for condition in detected_classes:
        medicines = medicine_df[medicine_df['disease'].str.lower() == condition.lower()]
        for _, medicine in medicines.iterrows():
            suggested_medicines.append({
                'disease': condition,
                'medicine': medicine['medicine'],
                'image': medicine['image']
            })

    return suggested_medicines

# def dentist(scores):
#     code_counts = {}
#     for code in scores:
#         if code in code_counts:
#             code_counts[code] += 1
#         else:
#             code_counts[code] = 1

#     output_parts = []
#     for code, count in code_counts.items():
#         output_parts.append(f"{count} {code} teeth")

#     output_sentence = ", ".join(output_parts[:-1]) + ", and " + output_parts[-1]
#     output_sentence = "You have " + output_sentence

#     level = [int(s[-1]) for s in scores]
#     max_level = max(level)

#     severity = df_teeth_score[df_teeth_score.level == max_level]
#     treatment = severity.treatment.values[0]

#     if severity.need_dentist.values[0]:
#         need_dentist = "Book consultation with Dentist"
#     else:
#         need_dentist = "Keep your teeth healthy :)"

#     return output_sentence, treatment, need_dentist

def diagnosis(detected_classes):
    disease = np.unique(detected_classes)
    output_sentence = "You have indication of "
    if len(disease) == 1:
        output_sentence += disease[0]
    else:
        for item in disease:
            output_sentence += item + ", "
        output_sentence = output_sentence[:-2]
    return output_sentence

if __name__ == '__main__':
    app.run(debug=True)
