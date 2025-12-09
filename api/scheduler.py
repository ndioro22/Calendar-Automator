import os
import math
import pandas as pd
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo
from icalendar import Calendar as ICalCalendar
from ics import Calendar, Event
import recurring_ical_events

# Configuration
LOCAL_TZ = ZoneInfo("America/New_York")
CHUNK_SIZE = 60 # Minutes per study session

def parse_user_ics(ics_path, start_date, end_date):
    """Parses user's uploaded ICS to find BUSY times."""
    busy_blocks = []
    
    if not ics_path or not os.path.exists(ics_path):
        return busy_blocks

    try:
        with open(ics_path, 'rb') as f:
            cal = ICalCalendar.from_ical(f.read())
        
        # Expand recurring events
        events = recurring_ical_events.of(cal).between(start_date, end_date)
        
        for event in events:
            dtstart = event.get('DTSTART').dt
            dtend = event.get('DTEND').dt
            
            # Normalize to datetime
            if not isinstance(dtstart, datetime):
                dtstart = datetime.combine(dtstart, time.min).replace(tzinfo=LOCAL_TZ)
            if not isinstance(dtend, datetime):
                dtend = datetime.combine(dtend, time.max).replace(tzinfo=LOCAL_TZ)
            
            # Normalize Timezone
            if dtstart.tzinfo is None: dtstart = dtstart.replace(tzinfo=LOCAL_TZ)
            else: dtstart = dtstart.astimezone(LOCAL_TZ)
            
            if dtend.tzinfo is None: dtend = dtend.replace(tzinfo=LOCAL_TZ)
            else: dtend = dtend.astimezone(LOCAL_TZ)

            busy_blocks.append((dtstart, dtend))
            
    except Exception as e:
        print(f"Error parsing ICS: {e}")
    
    return busy_blocks

def generate_free_blocks(start_date, end_date, preferences, busy_blocks):
    """Generates available study slots."""
    free_blocks = []
    current_day = start_date.date()
    end_date_date = end_date.date()

    busy_blocks.sort(key=lambda x: x[0])

    while current_day <= end_date_date:
        is_weekend = current_day.weekday() >= 5
        
        if is_weekend:
            start_str = preferences.get('weekendStart', '10:00')
            end_str = preferences.get('weekendEnd', '20:00')
        else:
            start_str = preferences.get('weekdayStart', '09:00')
            end_str = preferences.get('weekdayEnd', '22:00')

        try:
            s_h, s_m = map(int, start_str.split(':'))
            e_h, e_m = map(int, end_str.split(':'))
        except:
            s_h, s_m, e_h, e_m = 9, 0, 21, 0
        
        day_start = datetime.combine(current_day, time(s_h, s_m)).replace(tzinfo=LOCAL_TZ)
        day_end = datetime.combine(current_day, time(e_h, e_m)).replace(tzinfo=LOCAL_TZ)

        current_pointer = day_start
        day_busy = [b for b in busy_blocks if b[1] > day_start and b[0] < day_end]

        for b_start, b_end in day_busy:
            b_start = max(b_start, day_start)
            b_end = min(b_end, day_end)

            if b_start > current_pointer:
                duration = (b_start - current_pointer).total_seconds() / 60
                if duration >= 30: 
                    free_blocks.append({'start': current_pointer, 'end': b_start, 'duration': duration})
            current_pointer = max(current_pointer, b_end)

        if current_pointer < day_end:
            duration = (day_end - current_pointer).total_seconds() / 60
            if duration >= 30:
                free_blocks.append({'start': current_pointer, 'end': day_end, 'duration': duration})

        current_day += timedelta(days=1)

    return pd.DataFrame(free_blocks)

def create_schedule(courses, preferences, user_ics_path, output_path):
    now = datetime.now(LOCAL_TZ)
    end_horizon = now + timedelta(days=90)
    
    busy_blocks = parse_user_ics(user_ics_path, now, end_horizon)
    free_df = generate_free_blocks(now, end_horizon, preferences, busy_blocks)
    
    if free_df.empty: return None

    sessions = []
    for c in courses:
        try:
            d_str = c.get('date')
            if not d_str: continue
            due_date = datetime.strptime(d_str, '%Y-%m-%d').replace(tzinfo=LOCAL_TZ)
            due_date = due_date.replace(hour=23, minute=59)
            
            hours = float(c.get('predicted_hours', 1))
            total_minutes = int(hours * 60)
            num_chunks = math.ceil(total_minutes / CHUNK_SIZE)
            
            for i in range(num_chunks):
                duration = min(CHUNK_SIZE, total_minutes - (i * CHUNK_SIZE))
                sessions.append({
                    'name': c['name'],
                    'due_date': due_date,
                    'duration': duration
                })
        except: pass

    sessions.sort(key=lambda x: x['due_date'])
    scheduled_events = []
    
    for session in sessions:
        valid_blocks = free_df[
            (free_df['start'] >= now) & 
            (free_df['end'] <= session['due_date']) &
            (free_df['duration'] >= session['duration'])
        ]
        
        if not valid_blocks.empty:
            idx = valid_blocks.index[0]
            block = free_df.loc[idx]
            
            start_time = block['start']
            end_time = start_time + timedelta(minutes=session['duration'])
            
            scheduled_events.append({'name': f"Study: {session['name']}", 'start': start_time, 'end': end_time})
            
            new_start = end_time
            new_duration = (block['end'] - new_start).total_seconds() / 60
            
            if new_duration >= 30:
                free_df.at[idx, 'start'] = new_start
                free_df.at[idx, 'duration'] = new_duration
            else:
                free_df.drop(idx, inplace=True)
    
    c = Calendar()
    for ev in scheduled_events:
        e = Event()
        e.name = ev['name']
        e.begin = ev['start']
        e.end = ev['end']
        c.events.add(e)
        
    with open(output_path, 'w') as f:
        f.writelines(c.serialize_iter())
        
    return output_path
