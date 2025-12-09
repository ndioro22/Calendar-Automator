// Data Constants
const MAJORS = [
    "Business", "Tech & Data Science", "Engineering", "Math", 
    "Natural Sciences", "Social Sciences", "Arts & Humanities", "Health & Education"
];

const ASSIGNMENT_TYPES = [
    "Problem Set", "Coding", "Research Paper", "Creative Writing", 
    "Presentation", "Modeling - Finance", "Modeling - Stats", "Modeling - Data", 
    "Discussion Post", "Readings", "Case Study"
];

document.addEventListener('DOMContentLoaded', () => {
    // Populate Survey Major/Minor Selects
    const majorSelects = document.querySelectorAll('.major-select');
    majorSelects.forEach(sel => {
        MAJORS.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m;
            opt.textContent = m;
            sel.appendChild(opt);
        });
    });

    // Init Defaults
    addAssignment();
    addSyllabusRow();
});

/* --- Assignments Logic --- */
const assignmentsContainer = document.getElementById('assignmentsContainer');

function addAssignment() {
    const div = document.createElement('div');
    div.className = 'dynamic-card';
    // Add a unique ID to help locate this card later if needed, though we use order currently
    const cardId = `card-${Date.now()}`;
    div.dataset.id = cardId;
    
    let majorOpts = `<option value="">Select Field...</option>`;
    MAJORS.forEach(m => majorOpts += `<option value="${m}">${m}</option>`);

    let typeOpts = `<option value="">Select Type...</option>`;
    ASSIGNMENT_TYPES.forEach(t => typeOpts += `<option value="${t}">${t}</option>`);

    div.innerHTML = `
        <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
            <label style="color:#a78bfa; font-weight:bold;">Task Node</label>
            <button type="button" class="btn-delete" onclick="this.parentElement.parentElement.remove()">Remove</button>
        </div>
        
        <div class="prediction-box" style="display:none; margin-bottom:15px; padding:10px; background:#dcfce7; color:#166534; border-radius:6px; font-weight:bold;">
            </div>

        <div class="grid-2">
            <div class="field">
                <label>Field of Study</label>
                <select class="a-field">${majorOpts}</select>
            </div>
            <div class="field">
                <label>Type</label>
                <select class="a-type">${typeOpts}</select>
            </div>
        </div>
        
        <div class="grid-3">
            <div class="field">
                <label>Resources</label>
                <select class="a-resources">
                    <option value="Internet">Internet</option>
                    <option value="AI">AI</option>
                    <option value="Class Materials">Class Materials</option>
                </select>
            </div>
            <div class="field">
                <label>Est. Sessions</label>
                <input type="number" class="a-sessions" min="1" value="1">
            </div>
            <div class="field">
                <label>Location</label>
                <select class="a-location">
                    <option value="Home">Home</option>
                    <option value="School">School</option>
                    <option value="Public">Public Area</option>
                </select>
            </div>
        </div>

        <div class="field">
            <label style="color:#f472b6;">Deadline</label>
            <input type="datetime-local" class="a-deadline">
        </div>

        <div class="grid-2">
            <div class="field">
                <label>In Person Submit?</label>
                <select class="a-inperson">
                    <option value="No">No</option>
                    <option value="Yes">Yes</option>
                </select>
            </div>
            <div class="field">
                <label>Group Work?</label>
                <select class="a-group">
                    <option value="No">No</option>
                    <option value="Yes">Yes</option>
                </select>
            </div>
        </div>
    `;
    assignmentsContainer.appendChild(div);
}

document.getElementById('addAssignmentBtn').addEventListener('click', addAssignment);


/* --- Syllabus Logic --- */
const syllabusContainer = document.getElementById('syllabusContainer');

function addSyllabusRow() {
    const div = document.createElement('div');
    div.className = 'syllabus-row';
    div.innerHTML = `
        <div style="flex:1">
            <input type="text" class="s-name" placeholder="Class Name (e.g. CS 101)">
        </div>
        <div style="flex:2">
            <input type="file" class="s-file" accept=".pdf">
        </div>
        <button type="button" class="btn-delete" onclick="this.parentElement.remove()">×</button>
    `;
    syllabusContainer.appendChild(div);
}

document.getElementById('addSyllabusBtn').addEventListener('click', addSyllabusRow);


/* --- Submission Logic --- */
document.getElementById('submitBtn').addEventListener('click', async () => {
    const btn = document.getElementById('submitBtn');
    const originalText = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Processing...";

    // 1. Gather Survey Data
    const surveyData = {
        year: document.getElementById('studentYear').value,
        major: document.getElementById('major').value,
        secondMajor: document.getElementById('secondMajor').value,
        minor: document.getElementById('minor').value,
        schedule: {
            weekday: {
                wake: document.getElementById('wakeWeekday').value,
                sleep: document.getElementById('sleepWeekday').value
            },
            weekend: {
                wake: document.getElementById('wakeWeekend').value,
                sleep: document.getElementById('sleepWeekend').value
            }
        }
    };

    // 2. Gather Assignments
    const assignmentCards = document.querySelectorAll('.dynamic-card');
    const assignments = [];
    
    // We iterate to maintain order so index 0 in array matches index 0 in DOM
    assignmentCards.forEach(card => {
        // Reset previous predictions
        const predBox = card.querySelector('.prediction-box');
        predBox.style.display = 'none';
        predBox.textContent = '';

        assignments.push({
            fieldOfStudy: card.querySelector('.a-field').value,
            type: card.querySelector('.a-type').value,
            resources: card.querySelector('.a-resources').value,
            sessions: parseInt(card.querySelector('.a-sessions').value) || 1,
            location: card.querySelector('.a-location').value,
            deadline: card.querySelector('.a-deadline').value, // Gather new deadline
            submitInPerson: card.querySelector('.a-inperson').value,
            groupWork: card.querySelector('.a-group').value
        });
    });

    const syllabusMeta = [];
    const formData = new FormData();
    
    // 3. Files
    const icsFile = document.getElementById('icsUpload').files[0];
    if (icsFile) formData.append('calendar_file', icsFile);

    document.querySelectorAll('.syllabus-row').forEach((row, index) => {
        const name = row.querySelector('.s-name').value;
        const fileInput = row.querySelector('.s-file');
        if (fileInput.files[0]) {
            const fileKey = `syllabus_${index}`;
            formData.append(fileKey, fileInput.files[0]);
            syllabusMeta.push({ className: name, fileKey: fileKey });
        }
    });

    // 4. Construct Payload
    const fullJson = {
        survey: surveyData,
        assignments: assignments,
        syllabusMeta: syllabusMeta
    };
    formData.append('data_json', JSON.stringify(fullJson));

    try {
        const response = await fetch('/upload_data', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();

        if (response.ok) {
            // Handle Predictions
            if (result.predictions && result.predictions.length > 0) {
                result.predictions.forEach(pred => {
                    // Match result index to card index
                    if (assignmentCards[pred.index]) {
                        const card = assignmentCards[pred.index];
                        const predBox = card.querySelector('.prediction-box');
                        predBox.style.display = 'block';
                        predBox.innerHTML = `AI Estimate: ${pred.predicted_hours} Hours`;
                    }
                });
                btn.textContent = "Optimization Complete";
                setTimeout(() => { btn.textContent = originalText; btn.disabled = false; }, 3000);
            } else {
                alert("Data uploaded, but no predictions returned.");
                btn.disabled = false;
                btn.textContent = originalText;
            }
        } else {
            throw new Error(result.error || "Unknown server error.");
        }

    } catch (e) {
        console.error("Ingestion failed:", e);
        alert(`Error: ${e.message}`);
        btn.disabled = false;
        btn.textContent = originalText;
    }
});