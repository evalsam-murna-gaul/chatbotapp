import sqlite3
import json

# Load JSON data
with open("auninfo.json", "r") as file:
    json_data = json.load(file)

# Connect to SQLite
conn = sqlite3.connect("chatbotapp.db")
cursor = conn.cursor()

# Create the main university table
cursor.execute("""
CREATE TABLE IF NOT EXISTS university (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    abbreviation TEXT,
    city TEXT,
    state TEXT,
    country TEXT,
    established INTEGER,
    motto TEXT,
    founder TEXT,
    type TEXT,
    website TEXT
)
""")

# Create the academic programs table
cursor.execute("""
CREATE TABLE IF NOT EXISTS academic_programs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    university_id INTEGER,
    school TEXT,
    department TEXT,
    FOREIGN KEY (university_id) REFERENCES university (id)
)
""")

# Create the scholarships table
cursor.execute("""
CREATE TABLE IF NOT EXISTS scholarships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    university_id INTEGER,
    type TEXT,
    requirement TEXT,
    coverage TEXT,
    FOREIGN KEY (university_id) REFERENCES university (id)
)
""")

# Insert main university data
cursor.execute("""
INSERT INTO university (name, abbreviation, city, state, country, established, motto, founder, type, website)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", (
    json_data["name"],
    json_data["abbreviation"],
    json_data["location"]["city"],
    json_data["location"]["state"],
    json_data["location"]["country"],
    json_data["established"],
    json_data["motto"],
    json_data["founder"],
    json_data["type"],
    json_data["website"]
))

# Get the university ID for foreign keys
university_id = cursor.lastrowid

# Insert academic programs
for school in json_data["academic_programs"]:
    for department in school.get("departments", []):
        cursor.execute("""
        INSERT INTO academic_programs (university_id, school, department)
        VALUES (?, ?, ?)
        """, (university_id, school["school"], department))

# Insert scholarships
for scholarship_type, details in json_data["scholarships"].items():
    cursor.execute("""
    INSERT INTO scholarships (university_id, type, requirement, coverage)
    VALUES (?, ?, ?, ?)
    """, (university_id, scholarship_type, details["requirement"], details["coverage"]))

# Commit and close
conn.commit()
conn.close()

print("Data stored in structured SQLite tables successfully!")
conn = sqlite3.connect("chatbotapp.db")
cursor = conn.cursor()

cursor.execute("SELECT school, department FROM academic_programs")
programs = cursor.fetchall()

print("Academic Programs:")
for program in programs:
    print(program)

conn.close()
