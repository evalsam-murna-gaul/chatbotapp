'''from sqlalchemy import create_engine, MetaData, Table, Column, Integer, Text
from sqlalchemy.sql import text

# Step 1: Connect to your SQLite database
engine = create_engine("sqlite:///chatbotapp.db")  # Replace with your DB path
metadata = MetaData()

# Step 2: Define the new table structure
new_table = Table(
    "university_documents_temp", metadata,
    Column("id", Integer, primary_key=True),
    Column("topic", Text),
    Column("file_path", Text)
)

# Step 3: Create the new temporary table
metadata.create_all(engine)

# Step 4: Run raw SQL safely using `text()`
with engine.connect() as conn:
    # Copy data
    conn.execute(text("""
        INSERT INTO university_documents_temp (id, file_path)
        SELECT id, file_path FROM university_documents;
    """))

    # Drop old table
    conn.execute(text("DROP TABLE university_documents;"))

    # Rename new table
    conn.execute(text("ALTER TABLE university_documents_temp RENAME TO university_documents;"))

print("✅ Table successfully recreated with the new 'topic' column.")

from sqlalchemy import create_engine, inspect

engine = create_engine("sqlite:///chatbotapp.db")
inspector = inspect(engine)

print("Tables:", inspector.get_table_names())

from sqlalchemy import create_engine
from sqlalchemy.sql import text

engine = create_engine("sqlite:///chatbotapp.db")

with engine.connect() as conn:
    conn.execute(text("DROP TABLE IF EXISTS university_documents;"))
    conn.execute(text("ALTER TABLE university_documents_temp RENAME TO university_documents;"))

print("✅ Table renamed successfully.")

from sqlalchemy import create_engine, MetaData, Table, Column, Integer, Text, DateTime
from sqlalchemy.sql import func, text

# Connect to the SQLite database
engine = create_engine("sqlite:///chatbotapp.db")
metadata = MetaData()

# Define new structure with the timestamp column
new_table = Table(
    "university_documents_temp", metadata,
    Column("id", Integer, primary_key=True),
    Column("topic", Text),
    Column("file_path", Text),
    Column("timestamp", DateTime, server_default=func.now())
)

# Create the temporary table
metadata.create_all(engine)

# Copy data from old to new table (timestamp will auto-fill)
with engine.connect() as conn:
    conn.execute(text("""
        INSERT INTO university_documents_temp (id, topic, file_path)
        SELECT id, topic, file_path FROM university_documents;
    """))

    # Drop old table and rename the new one
    conn.execute(text("DROP TABLE university_documents;"))
    conn.execute(text("ALTER TABLE university_documents_temp RENAME TO university_documents;"))

print("✅ Timestamp column added successfully.")'''

import sqlite3
import os

# === CONFIG ===
DB_PATH = r"C:\Users\MURNA\Documents\chatbotapp\chatbotapp.db"
FILE_PATH = r"C:\Users\MURNA\Documents\chatbotapp\aun_university_info.txt"  # full path to your file
TOPIC = "University Info"  # You can customize this

# === Insert into database ===
def add_document(topic: str, file_path: str):
    if not os.path.exists(file_path):
        print("File does not exist.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check if already exists
    cursor.execute("SELECT id FROM university_documents WHERE topic = ?", (topic,))
    if cursor.fetchone():
        print(f"'{topic}' already exists in the database.")
    else:
        cursor.execute("INSERT INTO university_documents (topic, file_path) VALUES (?, ?)", (topic, file_path))
        conn.commit()
        print(f"✅ Inserted: {topic} → {file_path}")

    conn.close()

add_document(TOPIC, FILE_PATH)
print('okay! ✅ ')
























'''
def add_document(topic: str, filename: str):
    file_path = f"C:/Users/MURNA/Documents/chatbotapp/aun_university_info.txt{filename}"
    #file_path = f"docs/aun_university_info.txt"
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO university_documents (topic, file_path) VALUES (?, ?)", (topic, file_path))
    conn.commit()
    conn.close()
    print(f"Document added: {topic} → {file_path}")
    add_document("school rules", "aun_university_info.txt")


def fetch_document_info(user_input):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT file_path FROM university_documents WHERE topic LIKE ?", (f"%{user_input}%",))
    row = cursor.fetchone()
    conn.close()

    if row:
        file_path = row[0]
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                return content[:1000]  # Keep it concise for the AI
        except Exception as e:
            return f"Error reading file: {e}"
    return None


def store_conversation(student_id: str, question: str, answer: str):
    current_time = datetime.now() 
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chat_history2 (user_id, question, response, timestamp) VALUES (?, ?, ?, ?)", (student_id, question, answer, current_time))
    conn.commit()
    conn.close()

# Prompt template for multilingual and local context support
def build_prompt(user_input: str, lang_hint: str = "English"):
    return f"""
<|system|>
You are a helpful university assistant who can speak multiple languages including English, Hausa, Pidgin, and French. Respond to user queries clearly, politely, and concisely in {lang_hint}.
<|user|>
{user_input}
<|assistant|>
"""

# Smart Chat Endpoint
@app.get("/chat", tags=['USER'])
def chat(message: str = Query(...), student_id: str = Query(...), lang: str = Query(default="English")):
    message_lower = message.lower()

    doc_info = fetch_document_info(message)
    if doc_info:
        prompt = build_prompt(f"{doc_info}\n\nUser question: {message}", lang)
    else:
        prompt = build_prompt(message, lang)

    # Check if it's a schedule-related question
    #if "next class" in message_lower or "my schedule" in message_lower:
        schedule = query_schedule(student_id)
        if schedule:
            reply = "\n".join([f"{c} on {d} at {t}" for c, d, t in schedule])
        else:
            reply = "No schedule found for this student."
    else:
        # Build prompt with language context
        #prompt = build_prompt(message, lang)
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    with torch.no_grad():
        output = chat_model.generate(input_ids, max_new_tokens=100, do_sample=False)

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        reply = response.replace(prompt, "").strip()

    # Store conversation
    store_conversation(student_id, message, reply)

    return {"response": reply}
from fastapi import FastAPI, Query
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import sqlite3
import uvicorn
from huggingface_hub import HfFolder

HfFolder.save_token("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")
# Load LLaMA model from Hugging Face
model_id = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
llama_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# SQLite database path
db_path = "./university.db"

# Prompt template with multilingual support
multilingual_template = """
You are a helpful university assistant for African students.
Respond in the language the question was asked.
If it’s Pidgin, keep it casual. If it’s Hausa, respond in Hausa. Otherwise, use English.
Use the following database knowledge if relevant:
{db_result}

Question: {query}
Answer:
"""

# Function to get database info based on query keywords
def query_database(user_query: str) -> str:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    result = ""

    if "schedule" in user_query.lower():
        cursor.execute("SELECT * FROM schedule LIMIT 3")
        rows = cursor.fetchall()
        result = "\n".join([str(row) for row in rows])

    elif "exam" in user_query.lower():
        cursor.execute("SELECT * FROM exams LIMIT 3")
        rows = cursor.fetchall()
        result = "\n".join([str(row) for row in rows])

    # Add more keyword-based logic here

    conn.close()
    return result or "No matching data found in database."

# Build prompt dynamically
def build_prompt(query: str, db_result: str) -> str:
    return multilingual_template.format(query=query, db_result=db_result)

# Generate answer using LLaMA
def generate_answer(prompt: str) -> str:
    result = llama_pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)
    return result[0]["generated_text"].split("Answer:")[-1].strip()

# Request schema
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(payload: ChatRequest):
    user_query = payload.message
    db_data = query_database(user_query)
    prompt = build_prompt(user_query, db_data)
    response = generate_answer(prompt)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)'''
'''def query_schedule(student_id: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT course_name, day, time FROM schedules WHERE student_id = ?", (student_id,))
    results = cursor.fetchall()
    conn.close()
    return results'''