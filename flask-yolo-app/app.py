import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
import cv2
from doctor import suggest_medicine, diagnosis

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load YOLO models
# teeth_model = YOLO(r'C:\Users\nagav\Desktop\dr_martha-main\flask-yolo-app\weights\yolo_teeth.pt')
tongue_model = YOLO(r'C:\Users\nagav\Desktop\dr_martha-main\flask-yolo-app\weights\yolo_tongue.pt')
eye_model = YOLO(r'C:\Users\nagav\Desktop\dr_martha-main\flask-yolo-app\weights\yolo_eye.pt')
face_model = YOLO(r'C:\Users\nagav\Desktop\dr_martha-main\flask-yolo-app\weights\yolo_face.pt')

# Load the medicine and doctor data
medicine_file = os.path.join(os.path.dirname(__file__), 'data', 'medicine.csv')
doctor_file = os.path.join(os.path.dirname(__file__), 'data', 'doctor.csv')

if os.path.exists(medicine_file) and os.path.exists(doctor_file):
    medicine_df = pd.read_csv(medicine_file)
    doctor_df = pd.read_csv(doctor_file)
    print(doctor_df.columns)  # Debugging line to print column names
else:
    raise FileNotFoundError("The CSV files for medicine and/or doctor data were not found.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files or 'health_type' not in request.form:
        return redirect(request.url)

    file = request.files['file']
    health_type = request.form['health_type']

    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # if health_type == 'teeth':
        #     results = teeth_model.predict(filename)
        #     result_image_path = filename.replace('.jpg', '_result.jpg')
        #     result_image = results[0].plot()
        #     cv2.imwrite(result_image_path, result_image)
        #     detected_classes = [results[0].names[int(cls)] for cls in results[0].boxes.cls]
        #     output_sentence, treatment, need_dentist = dentist(detected_classes)
        #     return render_template('index.html', image=os.path.basename(result_image_path),
        #                            output_sentence=output_sentence, treatment=treatment, need_dentist=need_dentist,
        #                            detected_classes=detected_classes, health_type=health_type)

        if health_type == 'tongue':
            results = tongue_model.predict(filename)
            result_image_path = filename.replace('.jpg', '_result.jpg')
            result_image = results[0].plot()
            cv2.imwrite(result_image_path, result_image)
            detected_classes = [results[0].names[int(cls)] for cls in results[0].boxes.cls]
            output_sentence = diagnosis(detected_classes)
            return render_template('index.html', image=os.path.basename(result_image_path),
                                   output_sentence=output_sentence,
                                   detected_classes=detected_classes, health_type=health_type)

        elif health_type == 'eye':
            results = eye_model.predict(filename)
            result_image_path = filename.replace('.jpg', '_result.jpg')
            result_image = results[0].plot()
            cv2.imwrite(result_image_path, result_image)
            detected_classes = [results[0].names[int(cls)] for cls in results[0].boxes.cls]
            output_sentence = diagnosis(detected_classes)
            return render_template('index.html', image=os.path.basename(result_image_path),
                                   output_sentence=output_sentence,
                                   detected_classes=detected_classes, health_type=health_type)
        
        elif health_type == 'face':
            results = face_model.predict(filename)
            result_image_path = filename.replace('.jpg', '_result.jpg')
            result_image = results[0].plot()
            cv2.imwrite(result_image_path, result_image)
            detected_classes = [results[0].names[int(cls)] for cls in results[0].boxes.cls]
            output_sentence = diagnosis(detected_classes)
            return render_template('index.html', image=os.path.basename(result_image_path),
                                   output_sentence=output_sentence,
                                   detected_classes=detected_classes, health_type=health_type)

@app.route('/medicine')
def medicine():
    detected_classes = request.args.get('detected_classes').split(',')
    suggested_medicines = suggest_medicine(detected_classes)
    return render_template('medicine.html', detected_classes=detected_classes, medicines=suggested_medicines)

@app.route('/doctors')
def doctors():
    health_type = request.args.get('health_type')
    # Adjust column name based on actual column names in your CSV
    relevant_doctors = doctor_df[doctor_df['Category'].str.lower() == health_type.lower()]
    return render_template('doctor.html', doctors=relevant_doctors.to_dict(orient='records'))

@app.route('/results/<image>')
def results(image):
    return render_template('results.html', image=image)

if __name__ == '__main__':
    app.run(debug=True)
