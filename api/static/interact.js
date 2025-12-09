// --- 1. CONFIGURATION CONSTANTS (Matched to ML Model) ---
const MAJORS = [
    "Business", 
    "Tech & Data Science", 
    "Engineering", 
    "Math", 
    "Natural Sciences", 
    "Social Sciences", 
    "Arts & Humanities", 
    "Health & Education"
];

const ASSIGNMENT_TYPES = [
    "Problem Set", 
    "Coding Assignment", 
    "Research Paper", 
    "Creative Writing/Essay", 
    "Presentation", 
    "Modeling", 
    "Discussion Post", 
    "Readings", 
    "Case Study"
];

// --- 2. INITIALIZATION ---
document.addEventListener('DOMContentLoaded', () => {
    // Populate Majors
    const majorSelect = document.getElementById('major');
    majorSelect.innerHTML = '<option value="">-- Select Field --</option>';
    MAJORS.forEach(m => {
        majorSelect.add(new Option(m, m));
    });

    // Initialize Retro Time Pickers
    initTimePickers();

    // Add first empty assignment row
    addAssignmentRow();
});

// --- 3. RETRO TIME PICKER LOGIC ---
function initTimePickers() {
    document.querySelectorAll('.time-picker').forEach(container => {
        container.innerHTML = ''; // Clear
        const hourSel = document.createElement('select');
        const minSel = document.createElement('select');
        
        // Populate Hours (00-23)
        for(let i=0; i<24; i++) {
            let val = i < 10 ? '0'+i : i;
            hourSel.add(new Option(val, val));
        }
        
        // Populate Minutes (00-55)
        for(let i=0; i<60; i+=15) { 
            let val = i < 10 ? '0'+i : i;
            minSel.add(new Option(val, val));
        }
        
        // Set Defaults from HTML data attribute
        const defTime = container.dataset.default.split(':');
        hourSel.value = defTime[0]; 
        minSel.value = defTime[1]; 

        container.appendChild(hourSel);
        container.innerHTML += '<span style="margin:0 5px; font-weight:bold;">:</span>';
        container.appendChild(minSel);
    });
}

// Helper to extract "HH:MM" string from picker
function getPickerValue(id) {
    const container = document.getElementById(id);
    const selects = container.querySelectorAll('select');
    return `${selects[0].value}:${selects[1].value}`;
}

// --- 4. ASSIGNMENT ROW LOGIC ---
const assignmentsContainer = document.getElementById('assignmentsContainer');

function addAssignmentRow() {
    const div = document.createElement('div');
    div.className = 'syllabus-row'; // Re-using this class for styling
    
    // Type Options
    let typeOpts = `<option value="">Type...</option>`;
    ASSIGNMENT_TYPES.forEach(t => typeOpts += `<option value="${t}">${t}</option>`);

    div.innerHTML = `
        <div style="flex:2;">
            <label style="font-size:0.7rem; display:block; margin-bottom:2px;">Name</label>
            <input type="text" class="assign-name" placeholder="Task Name" style="width:100%;">
        </div>
        <div style="flex:1.5;">
            <label style="font-size:0.7rem; display:block; margin-bottom:2px;">Type</label>
            <select class="assign-type" style="width:100%;">${typeOpts}</select>
        </div>
        <div style="flex:1;">
            <label style="font-size:0.7rem; display:block; margin-bottom:2px;">Due Date</label>
            <input type="date" class="assign-date" style="width:100%;">
        </div>
        <button class="btn-delete" onclick="this.parentElement.remove()">×</button>
        <div class="prediction-tag" style="display:none; margin-left:10px; font-size:0.8rem; color:var(--accent-secondary); font-weight:bold;"></div>
    `;
    assignmentsContainer.appendChild(div);
}

document.getElementById('addAssignmentBtn').addEventListener('click', addAssignmentRow);

// --- 5. MAIN SUBMIT LOGIC ---
document.getElementById('submitBtn').addEventListener('click', async () => {
    const btn = document.getElementById('submitBtn');
    const originalText = btn.textContent;
    btn.textContent = "PROCESSING...";
    btn.disabled = true;

    try {
        // A. Collect Survey
        const surveyData = {
            year: document.getElementById('studentYear').value,
            major: document.getElementById('major').value
        };

        // B. Collect Work Windows (From Retro Pickers)
        const preferences = {
            weekdayStart: getPickerValue('weekdayStart'),
            weekdayEnd: getPickerValue('weekdayEnd'),
            weekendStart: getPickerValue('weekendStart'),
            weekendEnd: getPickerValue('weekendEnd')
        };

        // C. Collect Assignments
        const courses = [];
        const rows = document.querySelectorAll('.syllabus-row'); // reusing class name for rows
        rows.forEach(row => {
            const name = row.querySelector('.assign-name').value;
            const type = row.querySelector('.assign-type').value;
            const date = row.querySelector('.assign-date').value;
            if (name && type && date) {
                courses.push({ name, type, date });
            }
        });

        // D. Build Payload
        const fullJson = { survey: surveyData, courses: courses, preferences: preferences };
        const formData = new FormData();
        formData.append('data', JSON.stringify(fullJson));

        // E. Files
        const pdfInput = document.getElementById('pdfUpload');
        if(pdfInput.files.length > 0) {
            for(let i=0; i<pdfInput.files.length; i++) {
                formData.append('pdfs', pdfInput.files[i]);
            }
        }

        const icsInput = document.getElementById('icsUpload');
        if(icsInput.files.length > 0) {
            formData.append('ics', icsInput.files[0]);
        }

        // F. Send to Backend
        const response = await fetch('/api/generate-schedule', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();

        if (response.ok) {
            btn.textContent = "DONE";
            
            // Show Download
            if (result.ics_url) {
                const resultArea = document.getElementById('resultArea');
                const downloadLink = document.getElementById('downloadLink');
                
                resultArea.style.display = 'block';
                downloadLink.href = result.ics_url;
                downloadLink.download = "My_Study_Schedule.ics";
                
                // Scroll to result
                resultArea.scrollIntoView({ behavior: 'smooth' });
            }

            // Show predictions on UI (Retro Flair)
            if (result.courses) {
                // This matches result courses back to UI rows (only works for manual entry part)
                // Note: PDF extracted courses won't show up in the input list, but are in the download.
                rows.forEach((row, index) => {
                    if (result.courses[index]) {
                        const tag = row.querySelector('.prediction-tag');
                        if (tag) {
                            tag.style.display = 'block';
                            tag.textContent = `${result.courses[index].predicted_hours}h`;
                        }
                    }
                });
            }

        } else {
            alert("Error: " + (result.error || "Unknown error"));
        }

    } catch (e) {
        console.error(e);
        alert("Request failed. See console.");
    } finally {
        setTimeout(() => {
            btn.disabled = false;
            btn.textContent = originalText;
        }, 2000);
    }
});
