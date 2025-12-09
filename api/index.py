import os
import json
import time
import re
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Optional

from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

# From previous I changed to import sender_template, replaced under the API router the def home from json to html and directory to static and template 
# --- THIRD PARTY LIBRARIES ---
# We wrap these in try/except blocks just to give clear errors if installation fails
try:
    from sklearn.linear_model import ElasticNet
except ImportError:
    print("CRITICAL: scikit-learn not installed.")
    sys.exit(1)

try:
    from google import genai
    from google.genai import types
    from pydantic import BaseModel, Field
except ImportError:
    print("CRITICAL: google-genai or pydantic not installed.")
    sys.exit(1)


# ============================================
# 1. CONFIGURATION
# ============================================

# Add this near the top
template_dir = os.path.join(BASE_DIR, 'templates')
static_dir = os.path.join(BASE_DIR, 'static')

# Update the app definition
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
CORS(app)

# Directory Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
DATA_FOLDER = os.path.join(BASE_DIR, 'data')

# Create necessary folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

# Configuration
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload

# Paths
CSV_PATH = os.path.join(BASE_DIR, 'survey.csv')  # RENAME YOUR CSV TO THIS
TRAINING_PKL = os.path.join(DATA_FOLDER, "training_data.pkl")
ABOUT_PKL = os.path.join(DATA_FOLDER, "about_you.pkl")

# Secrets
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


# ============================================
# 2. PDF PARSER (GEMINI)
# ============================================

class MeetingSchedule(BaseModel):
    days: List[str] = Field(description="List of days (e.g., ['Monday', 'Wednesday']).")
    start_time: str = Field(description="The START time of the class ONLY (e.g. '11:00 AM').")

class AssignmentItem(BaseModel):
    date: str = Field(description="The due date in 'YYYY-MM-DD' format.")
    time: Optional[str] = Field(description="Specific deadline if explicitly written (e.g. '11:59 PM'). Otherwise null.")
    assignment_name: str = Field(description="Concise name.")
    category: str = Field(description="Category: 'Reading', 'Writing', 'Exam', 'Project', 'Presentation', 'Other'.")
    description: str = Field(description="Details.")

class CourseMetadata(BaseModel):
    course_name: str = Field(description="Name of the course.")
    class_meetings: List[MeetingSchedule] = Field(description="The weekly schedule.")

class SyllabusResponse(BaseModel):
    metadata: CourseMetadata
    assignments: List[AssignmentItem]

def standardize_time(time_str):
    if not time_str: return None
    clean = re.split(r'\s*[-–]\s*|\s+to\s+', str(time_str))[0].strip()
    for fmt in ["%I:%M %p", "%I %p", "%H:%M", "%I:%M%p"]:
        try:
            return datetime.strptime(clean, fmt).strftime("%I:%M %p")
        except ValueError:
            continue
    # Handle plain integers (e.g. "11" -> "11:00 AM")
    if clean.isdigit():
        val = int(clean)
        if 8 <= val <= 11: return f"{val:02d}:00 AM"
        if 1 <= val <= 6:  return f"{val:02d}:00 PM"
        if val == 12:      return "12:00 PM"
    return clean

def resolve_time(row, schedule_map):
    # If explicit time exists, use it
    existing = row.get('Time')
    if existing and any(c.isdigit() for c in str(existing)):
        return standardize_time(existing)
   
    # Otherwise, try to infer from class schedule
    try:
        day_name = pd.to_datetime(row.get('Date')).strftime('%A')
        return schedule_map.get(day_name, "11:59 PM")
    except:
        return "11:59 PM"

def parse_syllabus(file_path):
    """Uses Google Gemini to extract assignments from PDF."""
    if not GEMINI_API_KEY:
        print("⚠️ Skipped Parsing: GEMINI_API_KEY missing.")
        return None

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
       
        # Upload file
        print(f"Uploading to Gemini: {os.path.basename(file_path)}")
        file_upload = client.files.upload(file=file_path)
       
        # Wait for processing
        while file_upload.state.name == "PROCESSING":
            time.sleep(1)
            file_upload = client.files.get(name=file_upload.name)
       
        if file_upload.state.name != "ACTIVE":
            print("❌ Gemini Error: File processing failed.")
            return None

        prompt = """
        Analyze this syllabus. Extract:
        1. Course Name.
        2. Weekly Schedule (Days & Start Times).
        3. ALL Assignments (Dates YYYY-MM-DD, Deliverables, Readings).
        """

        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[file_upload, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SyllabusResponse
            )
        )
       
        data: SyllabusResponse = response.parsed
       
        # Build Class Schedule Map
        schedule_map = {}
        for meeting in data.metadata.class_meetings:
            t = standardize_time(meeting.start_time)
            for d in meeting.days:
                for full_day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
                    if full_day.lower() in d.lower():
                        schedule_map[full_day] = t

        # Build DataFrame
        rows = []
        for item in data.assignments:
            rows.append({
                "Course": data.metadata.course_name,
                "Date": item.date,
                "Time": item.time,
                "Category": item.category,
                "Assignment": item.assignment_name,
                "Description": item.description
            })
           
        df = pd.DataFrame(rows)
        if not df.empty:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
            df = df.dropna(subset=['Date'])
            df['Time'] = df.apply(lambda r: resolve_time(r, schedule_map), axis=1)

        return df

    except Exception as e:
        print(f"❌ PDF Parsing Failed: {e}")
        return None

def map_pdf_category(cat):
    """Maps syllabus categories to ML model categories."""
    c = str(cat).lower()
    if 'reading' in c: return 'readings'
    if 'writing' in c: return 'essay'
    if 'exam' in c: return 'p_set' # Exams take study time similar to p-sets
    if 'project' in c: return 'research_paper'
    if 'present' in c: return 'presentation'
    return 'p_set' # Default


# ============================================
# 3. ML MODEL (INITIALIZED ON STARTUP)
# ============================================

model = None
model_columns = []

def initialize_model():
    global model, model_columns
    print("🤖 Loading CSV and Training Model...")

    if not os.path.exists(CSV_PATH):
        print(f"❌ CRITICAL ERROR: CSV not found at {CSV_PATH}")
        print("👉 Please rename your file to 'survey.csv' and place it in the root folder.")
        return

    try:
        df = pd.read_csv(CSV_PATH)
       
        # MAPPINGS
        col_map = {
            'What year are you? ': 'year',
            'What is your major/concentration?': 'major',
            'Second concentration? (if none, select N/A)': 'second_concentration',
            'Minor? (if none select N/A)': 'minor',
            'What field of study was the assignment in?': 'field_of_study',
            'What type of assignment was it?': 'assignment_type',
            'Approximately how long did it take (in hours)': 'time_spent_hours',
            'What was the extent of your reliance on external resources? ': 'external_resources',
            'Where did you primarily work on the assignment?': 'work_location',
            'Did you work in a group?': 'worked_in_group',
            'Did you have to submit the assignment in person (physical copy)?': 'submitted_in_person'
        }
        df = df.rename(columns=col_map)
       
        # Simplified Categorical Mapping
        category_map = {
            'Computer Science': 'tech_data', 'Data Science': 'tech_data',
            'Mathematics': 'math', 'Physics': 'natural_sciences',
            'Business Administration': 'business', 'Finance': 'business',
            'English / Literature': 'arts_humanities', 'History': 'arts_humanities'
        }
        # Apply simplified map (expand this map if you need precise major mapping)
        if 'major' in df: df['major_category'] = df['major'].map(category_map).fillna('other')

        # One Hot Encoding
        encode_cols = ['year', 'assignment_type', 'external_resources', 'work_location', 'worked_in_group', 'submitted_in_person']
        for col in encode_cols:
            if col in df.columns:
                df = pd.get_dummies(df, columns=[col], prefix=col, dtype=int, drop_first=True)

        # Train
        df = df.select_dtypes(include=[np.number])
        if 'time_spent_hours' not in df.columns:
            print("❌ Target column 'time_spent_hours' missing after processing.")
            return

        X = df.drop('time_spent_hours', axis=1)
        y = df['time_spent_hours']
       
        clf = ElasticNet(alpha=0.078, l1_ratio=0.95, max_iter=5000)
        clf.fit(X, y)
       
        model = clf
        model_columns = list(X.columns)
        print("✅ Model Trained Successfully!")

    except Exception as e:
        print(f"❌ Model Training Failed: {e}")

# Run initialization immediately
initialize_model()


# ============================================
# 4. DATA STORE (PERSISTENCE)
# ============================================

def save_user_data(survey, courses, pdf_names, ics_names):
    """Saves user submission to a local pickle file."""
    timestamp = datetime.now().isoformat()
    row = {
        'timestamp': timestamp,
        'year': survey.get('year'),
        'major': survey.get('major'),
        'num_courses': len(courses),
        'courses_json': json.dumps(courses),
        'pdf_files': json.dumps(pdf_names),
        'ics_files': json.dumps(ics_names)
    }
   
    # Save to Training Log
    try:
        new_df = pd.DataFrame([row])
        if os.path.exists(TRAINING_PKL):
            existing_df = pd.read_pickle(TRAINING_PKL)
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
            updated_df.to_pickle(TRAINING_PKL)
        else:
            new_df.to_pickle(TRAINING_PKL)
    except Exception as e:
        print(f"Error saving data: {e}")


# ============================================
# 5. PREDICTION ENGINE
# ============================================

def predict_hours(survey, course_item):
    """Runs a single course against the trained model."""
    if not model:
        return 0.0

    try:
        # Create input row with all zeros
        input_row = pd.DataFrame(0, index=[0], columns=model_columns)
       
        # Map inputs to one-hot columns
        # (You must match the logic in initialize_model exactly)
       
        # Year
        year_col = f"year_{survey.get('year')}"
        if year_col in input_row.columns: input_row[year_col] = 1
       
        # Assignment Type
        type_col = f"assignment_type_{course_item.get('type')}"
        # Note: You might need a mapper here if frontend sends 'Problem Set' but model expects 'p_set'
        # For now, assuming frontend sends raw strings that match simplified keys or we map them:
        if 'Problem Set' in course_item.get('type', ''): type_col = 'assignment_type_p_set'
        if 'Essay' in course_item.get('type', ''): type_col = 'assignment_type_essay'
       
        if type_col in input_row.columns: input_row[type_col] = 1

        # Predict
        val = model.predict(input_row)[0]
        return max(0.0, round(float(val), 2)) # Ensure no negative time

    except Exception as e:
        print(f"Prediction Error: {e}")
        return 0.0


# ============================================
# 6. API ROUTES
# ============================================

@app.route('/')
def home():
    return render_template('mains.html')

@app.route('/api/generate-schedule', methods=['POST'])
def generate_schedule():
    try:
        data_json = request.form.get('data')
        if not data_json: return jsonify({'error': 'No JSON data'}), 400
       
        data = json.loads(data_json)
        survey = data.get('survey', {})
        manual_courses = data.get('courses', [])
       
        pdf_courses = []
        pdf_names = []
        ics_names = []

        # 1. Handle PDFs
        if 'pdfs' in request.files:
            files = request.files.getlist('pdfs')
            for f in files:
                if f.filename == '': continue
                filename = secure_filename(f.filename)
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                f.save(save_path)
                pdf_names.append(filename)
               
                # Trigger Parser
                if GEMINI_API_KEY:
                    df = parse_syllabus(save_path)
                    if df is not None:
                        for _, row in df.iterrows():
                            pdf_courses.append({
                                'name': f"{row['Course']} - {row['Assignment']}",
                                'type': map_pdf_category(row['Category']), # Map for ML
                                'subject': survey.get('major'), # Fallback subject
                                'resources': 'Google/internet',
                                'date': row['Date'],
                                'time': row['Time'],
                                'description': row['Description'],
                                'source': 'pdf_parser'
                            })

        # 2. Handle ICS (Store only)
        if 'ics' in request.files:
            f = request.files['ics']
            if f.filename != '':
                fname = secure_filename(f.filename)
                f.save(os.path.join(app.config['UPLOAD_FOLDER'], fname))
                ics_names.append(fname)

        # 3. Combine & Predict
        all_courses = manual_courses + pdf_courses
       
        for course in all_courses:
            course['predicted_hours'] = predict_hours(survey, course)
           
        # 4. Save History
        save_user_data(survey, all_courses, pdf_names, ics_names)

        return jsonify({
            "status": "success",
            "courses": all_courses,
            "debug": {
                "pdfs_processed": len(pdf_names),
                "assignments_extracted": len(pdf_courses)
            }
        })

    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/view-data')
def view_data():
    if os.path.exists(TRAINING_PKL):
        df = pd.read_pickle(TRAINING_PKL)
        return jsonify(df.to_dict(orient='records'))
    return jsonify([])

if __name__ == '__main__':
    # Local Dev
    app.run(debug=True, port=5000)
