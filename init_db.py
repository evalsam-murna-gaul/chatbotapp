'''from database import engine, Base

# Create tables in the database
Base.metadata.create_all(bind=engine)

print("Database tables created successfully!")'''

import sqlite3

conn = sqlite3.connect(r"C:\Users\MURNA\Documents\chatbotapp\chatbotapp.db")
cursor = conn.cursor()

# Create chat_history table if it doesn't exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    question TEXT,
    response TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")


conn.commit()
conn.close()

print("✅ Tables created successfully.")

