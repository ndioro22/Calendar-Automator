import os
import time
import pandas as pd
import re
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# --- CONFIGURATION ---
API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_ACTUAL_API_KEY_HERE")

# Define your directories here
BASE_DIR = os.getcwd()
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = BASE_DIR  # Main folder

# --- DATA MODELS ---
class MeetingSchedule(BaseModel):
    days: List[str] = Field(description="List of days (e.g., ['Monday', 'Wednesday']).")
    start_time: str = Field(description="The START time of the class ONLY (e.g. '11:00 AM').")

class AssignmentItem(BaseModel):
    date: str = Field(description="The due date in 'YYYY-MM-DD' format.")
    time: Optional[str] = Field(description="Specific deadline if explicitly written (e.g. '11:59 PM'). Otherwise null.")
    assignment_name: str = Field(description="Concise name.")
    category: str = Field(description="Category: 'Reading', 'Writing', 'Exam', 'Project', 'Presentation', 'Other'.")
    description: str = Field(description="Details. If exam, include duration here. If readings, bullet points.")

class CourseMetadata(BaseModel):
    course_name: str = Field(description="Name of the course.")
    semester_year: str = Field(description="Semester and Year (e.g., 'Fall 2025').")
    class_meetings: List[MeetingSchedule] = Field(description="The weekly schedule.")

class SyllabusResponse(BaseModel):
    metadata: CourseMetadata
    assignments: List[AssignmentItem]

# --- HELPERS ---
def standardize_time(time_str):
    """Parses messy times into 'HH:MM AM/PM'."""
    if not time_str: return None
    clean = re.split(r'\s*[-–]\s*|\s+to\s+', str(time_str))[0].strip()
    for fmt in ["%I:%M %p", "%I %p", "%H:%M", "%I:%M%p"]:
        try:
            return datetime.strptime(clean, fmt).strftime("%I:%M %p")
        except ValueError:
            continue
    if clean.isdigit():
        val = int(clean)
        if 8 <= val <= 11: return f"{val:02d}:00 AM"
        if 1 <= val <= 6:  return f"{val:02d}:00 PM"
        if val == 12:      return "12:00 PM"
    return clean

def resolve_time(row, schedule_map):
    """Prioritizes explicit deadlines -> class time -> 11:59 PM."""
    existing_time = row['Time']
    if existing_time and any(char.isdigit() for char in str(existing_time)):
        return standardize_time(existing_time)
    try:
        day_name = pd.to_datetime(row['Date']).strftime('%A') 
    except:
        return "11:59 PM"
    return schedule_map.get(day_name, "11:59 PM")

# --- CORE PARSER ---
def parse_syllabus(file_path):
    """Takes a FULL path to a PDF, uploads to Gemini, returns DataFrame."""
    client = genai.Client(api_key=API_KEY)
    filename = os.path.basename(file_path)
    print(f"🤖 Processing: {filename}...")

    try:
        # Upload using full path
        file_upload = client.files.upload(file=file_path)
        while file_upload.state.name == "PROCESSING":
            time.sleep(1)
            file_upload = client.files.get(name=file_upload.name)
        
        if file_upload.state.name != "ACTIVE":
            print(f"❌ Error: File not active.")
            return None

        prompt = """
        Analyze this syllabus for Calendar Import.
        PHASE 1: METADATA
        - Extract Course Name.
        - **Class Schedule:** Extract Days and **START TIME ONLY**.
        PHASE 2: ASSIGNMENTS
        - Extract deliverables and readings.
        - **Dates:** YYYY-MM-DD.
        - **Times:** Leave time NULL unless a specific deadline is written.
        - **Exams:** If "In Class", leave time NULL.
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
        
        schedule_map = {}
        for meeting in data.metadata.class_meetings:
            std_time = standardize_time(meeting.start_time)
            for day in meeting.days:
                for full_day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
                    if full_day.lower() in day.lower():
                        schedule_map[full_day] = std_time

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
            df['Time'] = df.apply(lambda row: resolve_time(row, schedule_map), axis=1)

        return df

    except Exception as e:
        print(f"❌ Error on {filename}: {e}")
        return None

def consolidate_assignments(df):
    if df.empty: return df
    group_cols = ['Course', 'Date', 'Time', 'Category']
    
    def merge_text(series):
        items = [s for s in series if s]
        unique_items = sorted(list(set(items)))
        if not unique_items: return ""
        if len(unique_items) == 1: return unique_items[0]
        
        clean_items = []
        for x in unique_items:
            x = x.strip()
            if x.startswith("•") or x.startswith("-"):
                clean_items.append(x)
            else:
                clean_items.append(f"• {x}")
        return "\n".join(clean_items)

    def merge_names(series):
        return " / ".join(sorted(set(series)))

    return df.groupby(group_cols).agg({
        'Assignment': merge_names,
        'Description': merge_text
    }).reset_index().sort_values(by=['Date', 'Time'])

# --- MAIN EXECUTION (BACKEND ENTRY POINT) ---
def process_uploads():
    """
    1. Looks in 'uploads' folder.
    2. Processes all PDFs.
    3. Saves master pickle/excel to main folder.
    """
    
    # 1. Create uploads folder if it doesn't exist (safety check)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
        print(f"📁 Created folder: {UPLOAD_FOLDER}")
        print("Please put your PDFs there and run again.")
        return

    # 2. Find PDFs in the uploads folder
    pdf_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"⚠️  No PDFs found in {UPLOAD_FOLDER}")
        return

    print(f"\n🚀 Found {len(pdf_files)} syllabi in /uploads. Processing...")
    
    all_dfs = []
    
    for filename in pdf_files:
        # Construct FULL PATH for the parser
        full_path = os.path.join(UPLOAD_FOLDER, filename)
        df = parse_syllabus(full_path)
        if df is not None:
            all_dfs.append(df)

    if all_dfs:
        print("\n🔄 Consolidating Master Schedule...")
        master_df = pd.concat(all_dfs, ignore_index=True)
        final_df = consolidate_assignments(master_df)
        
        # Save to OUTPUT_FOLDER (Main Directory)
        pkl_path = os.path.join(OUTPUT_FOLDER, "MASTER_Schedule.pkl")
        xlsx_path = os.path.join(OUTPUT_FOLDER, "MASTER_Schedule.xlsx")
        
        final_df.to_pickle(pkl_path)
        final_df.to_excel(xlsx_path, index=False)
        
        print(f"\n✅ SUCCESS!")
        print(f"   Saved Pickle: {pkl_path}")
        print(f"   Saved Excel:  {xlsx_path}")
        return final_df
    else:
        print("❌ No data could be extracted.")

if __name__ == "__main__":
    process_uploads()