import os
import json
import time
import re
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Optional

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

# --- IMPORT SCHEDULER ---
import scheduler

# --- 1. CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(BASE_DIR, 'templates') if os.path.exists(os.path.join(BASE_DIR, 'templates')) else None
static_dir = os.path.join(BASE_DIR, 'static') if os.path.exists(os.path.join(BASE_DIR, 'static')) else None

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
CORS(app)

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
DATA_FOLDER = os.path.join(BASE_DIR, 'data')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

CSV_PATH = os.path.join(BASE_DIR, 'survey.csv')
TRAINING_PKL = os.path.join(DATA_FOLDER, "training_data.pkl")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# --- 2. THIRD-PARTY & MODELS ---
try:
    from sklearn.linear_model import ElasticNet
except ImportError:
    sys.exit(1)

try:
    from google import genai
    from google.genai import types
    from pydantic import BaseModel, Field
except ImportError:
    genai = None

if genai:
    class AssignmentItem(BaseModel):
        date: str = Field(description="YYYY-MM-DD")
        time: Optional[str] = Field(description="Deadline or null")
        assignment_name: str = Field(description="Name")
        category: str = Field(description="Category")
        description: str = Field(description="Details")

    class SyllabusResponse(BaseModel):
        metadata: dict = Field(description="Metadata")
        assignments: List[AssignmentItem]

# --- 3. HELPER FUNCTIONS ---
def standardize_time(time_str):
    if not time_str: return None
    clean = re.split(r'\s*[-–]\s*|\s+to\s+', str(time_str))[0].strip()
    for fmt in ["%I:%M %p", "%I %p", "%H:%M", "%I:%M%p"]:
        try: return datetime.strptime(clean, fmt).strftime("%I:%M %p")
        except ValueError: continue
    if clean.isdigit():
        val = int(clean)
        if 8 <= val <= 11: return f"{val:02d}:00 AM"
        if 1 <= val <= 6:  return f"{val:02d}:00 PM"
        if val == 12:      return "12:00 PM"
    return clean

def map_pdf_category(cat):
    c = str(cat).lower()
    if 'reading' in c: return 'readings'
    if 'writing' in c: return 'essay'
    if 'exam' in c: return 'p_set'
    if 'project' in c: return 'research_paper'
    return 'p_set'

def parse_syllabus(file_path):
    if not genai or not GEMINI_API_KEY: return None
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        file_upload = client.files.upload(file=file_path)
        while file_upload.state.name == "PROCESSING":
            time.sleep(1)
            file_upload = client.files.get(name=file_upload.name)
        if file_upload.state.name != "ACTIVE": return None

        prompt = "Extract assignments (Dates YYYY-MM-DD), readings, and deliverables."
        response = client.models.generate_content(
            model='gemini-2.0-flash', 
            contents=[file_upload, prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=SyllabusResponse)
        )
        data = response.parsed
        rows = []
        c_name = getattr(data.metadata, 'course_name', 'Parsed Course')
        for item in data.assignments:
            rows.append({"Course": c_name, "Date": item.date, "Time": item.time, "Category": item.category, "Assignment": item.assignment_name, "Description": item.description})
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"Parsing error: {e}")
        return None

# --- 4. ML MODEL ---
model = None
model_columns = []
def initialize_model():
    global model, model_columns
    if not os.path.exists(CSV_PATH): return
    try:
        df = pd.read_csv(CSV_PATH)
        df = df.rename(columns={'What year are you? ': 'year', 'What is your major/concentration?': 'major', 'What type of assignment was it?': 'assignment_type', 'Approximately how long did it take (in hours)': 'time_spent_hours'})
        for col in ['year', 'assignment_type', 'external_resources', 'work_location', 'worked_in_group', 'submitted_in_person']:
            if col in df.columns: df = pd.get_dummies(df, columns=[col], prefix=col, dtype=int, drop_first=True)
        df = df.select_dtypes(include=[np.number])
        if 'time_spent_hours' in df.columns:
            X = df.drop('time_spent_hours', axis=1)
            y = df['time_spent_hours']
            clf = ElasticNet(alpha=0.078, l1_ratio=0.95, max_iter=5000)
            clf.fit(X, y)
            model = clf
            model_columns = list(X.columns)
            print("✅ Model Trained.")
    except Exception as e: print(f"Training Failed: {e}")

initialize_model()

# --- 5. ROUTES ---
@app.route('/', methods=['GET'])
def home():
    if template_dir: return render_template('mains.html')
    return "Server Online"

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

@app.route('/api/generate-schedule', methods=['POST'])
def generate_schedule():
    try:
        data_json = request.form.get('data')
        if not data_json: return jsonify({'error': 'No data'}), 400
        
        data = json.loads(data_json)
        survey = data.get('survey', {})
        manual_courses = data.get('courses', [])
        
        # 1. Get Preferences (Work Windows)
        preferences = data.get('preferences', {}) 
        
        pdf_courses = []
        user_ics_path = None

        # 2. Handle PDFs
        if 'pdfs' in request.files:
            for f in request.files.getlist('pdfs'):
                if f.filename:
                    fname = secure_filename(f.filename)
                    path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
                    f.save(path)
                    if GEMINI_API_KEY:
                        df = parse_syllabus(path)
                        if df is not None:
                            for _, r in df.iterrows():
                                pdf_courses.append({'name': f"{r['Course']} - {r['Assignment']}", 'type': map_pdf_category(r['Category']), 'subject': survey.get('major'), 'resources': 'Google/internet', 'date': r['Date'], 'time': r['Time'], 'description': r['Description'], 'source': 'pdf_parser'})

        # 3. Handle User ICS
        if 'ics' in request.files:
            f = request.files['ics']
            if f.filename:
                fname = secure_filename(f.filename)
                user_ics_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
                f.save(user_ics_path)

        # 4. Predict Hours (ML Model)
        all_courses = manual_courses + pdf_courses
        for course in all_courses:
            predicted = 0.0
            if model:
                try:
                    input_row = pd.DataFrame(0, index=[0], columns=model_columns)
                    yr = f"year_{survey.get('year')}"
                    if yr in input_row.columns: input_row[yr] = 1
                    rt = course.get('type', '').lower()
                    mt = 'p_set'
                    if 'essay' in rt: mt = 'essay'
                    elif 'coding' in rt: mt = 'coding'
                    elif 'read' in rt: mt = 'readings'
                    ty = f"assignment_type_{mt}"
                    if ty in input_row.columns: input_row[ty] = 1
                    predicted = float(model.predict(input_row)[0])
                except: pass
            course['predicted_hours'] = max(0.5, round(predicted, 2))

        # 5. RUN SCHEDULER
        output_ics = f"study_plan_{int(time.time())}.ics"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_ics)
        
        generated_file = scheduler.create_schedule(
            courses=all_courses,
            preferences=preferences,
            user_ics_path=user_ics_path,
            output_path=output_path
        )

        response = {
            "status": "success",
            "courses": all_courses,
            "parsed_count": len(pdf_courses)
        }
        
        if generated_file:
            response["ics_url"] = f"/download/{output_ics}"
        else:
            response["warning"] = "Could not generate schedule (no free time or parsing error)"

        return jsonify(response)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))

if __name__ == '__main__':
    # Local Dev
    app.run(debug=True, port=5000)
