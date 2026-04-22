
import sqlite3
from flask import Flask, render_template, request
import pytesseract
import cv2
import os
import pickle
from docx import Document
import PyPDF2

def save_to_db(filename, text, result):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO results (filename, extracted_text, ai_result)
        VALUES (?, ?, ?)
    ''', (filename, text, result))

    conn.commit()
    conn.close()     

app = Flask(__name__)

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load AI model
model = pickle.load(open("model/ai_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))


# 🏠 Home Page
@app.route('/')
def home():
    return render_template('index.html')


# 📤 Upload Page
@app.route('/upload')
def upload_page():
    return render_template('upload.html')


# 📥 Handle Upload
@app.route('/upload', methods=['POST'])
def upload_file():

    file = request.files['file']

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        filename = file.filename.lower()

        # 🖼 IMAGE FILES
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(filepath)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

            text = pytesseract.image_to_string(thresh, config='--psm 6')

        # 📄 PDF FILES
        elif filename.endswith('.pdf'):
            pdf = PyPDF2.PdfReader(filepath)
            text = ""

            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text()

        # 📄 DOCX FILES
        elif filename.endswith('.docx'):
            doc = Document(filepath)
            text = ""

            for para in doc.paragraphs:
                text += para.text + "\n"

        else:
            return "Unsupported file type"

        # 🤖 AI DETECTION
        X_input = vectorizer.transform([text])
        prediction = model.predict(X_input)[0]

        if prediction == 1:
            result = "⚠ AI Generated"
        else:
            result = "✅ Human Written"
            
            save_to_db(file.filename, text, result)

        return render_template('result.html',
                               extracted_text=text,
                               ai_result=result)

    return "No file uploaded"

@app.route('/records')
def view_records():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM results ORDER BY date DESC")
    records = cursor.fetchall()

    conn.close()

    return render_template('records.html', records=records)

# 🚀 Run App
if __name__ == '__main__':
    app.run(debug=True)
