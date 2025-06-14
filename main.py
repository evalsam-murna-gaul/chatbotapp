from fastapi import FastAPI, Depends, HTTPException, status, Request, Query
import model, schema, database, JWToken
from sqlalchemy.orm import Session
from database import engine, get_db, SessionLocal
from hashing import Hash
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from JWToken import blacklist_token, is_token_blacklisted, decode_token  # Custom JWT functions
from hashing import Hash
from utils import generate_otp, send_otp_email
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, json, re
from alembic import op
from fastapi import Header
from typing import Optional, Union


app = FastAPI()


# Connect to SQLite database
DB_PATH = r"C:\Users\MURNA\Documents\chatbotapp\chatbotapp.db"
JSON_DATA_PATH = "aun_faqs.json"
 

# Global variables for the model and tokenizer
'''model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
chat_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
chat_model.eval()  # Set model to eval mode for inference
# Move the model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"


#Load JSON Knowledge Base
with open("aun_faqs.json", "r", encoding="utf-8") as f:
    aun_data = json.load(f)


# Cache Documents
document_cache = {}

def preload_documents():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT topic, file_path FROM university_documents")
    for topic, path in cursor.fetchall():
        try:
            with open(path, "r", encoding="utf-8") as f:
                document_cache[topic.lower()] = f.read()
        except Exception as e:
            print(f"Failed to load {path}: {e}")
    conn.close()

preload_documents()'''


# Load model only once at startup
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Initialize with None and load lazily
tokenizer = None
chat_model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load JSON data with proper structure handling
def load_faq_data(file_path: str) -> list:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('faqs', []) if isinstance(data, dict) else data
    except Exception as e:
        print(f"Error loading FAQ data: {e}")
        return []

# Preprocess questions for faster lookup
FAQ_DATA = load_faq_data("aun_faqs.json")
QUESTION_MAP = {item['question'].lower(): item['answer'] for item in FAQ_DATA if 'question' in item}

def normalize_text(text: str) -> str:
    """Normalize text for consistent matching"""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return re.sub(r'\s+', ' ', text)  # Normalize whitespace

def format_answer(answer: Union[str, dict, list]) -> str:
    """Convert different answer formats to consistent string output"""
    if isinstance(answer, str):
        return answer
    elif isinstance(answer, dict):
        # Special handling for department chairs structure
        if all(isinstance(v, list) for v in answer.values()):
            result = []
            for school, departments in answer.items():
                result.append(f"{school}:")
                for dept in departments:
                    for dept_name, chair in dept.items():
                        result.append(f"  - {dept_name}: {chair}")
            return "\n".join(result)
        return json.dumps(answer, indent=2)
    elif isinstance(answer, list):
        if all(isinstance(item, dict) and len(item) == 1 for item in answer):
            # Handle list of single-key dictionaries
            return "\n".join(f"{list(item.keys())[0]}: {list(item.values())[0]}" 
                           for item in answer)
        else:
            return "\n".join(str(item) for item in answer)
    return str(answer)

def make_conversational(answer: str, question: str, lang: str = "English") -> str:
    """Transform raw answers into natural responses"""
    greetings = {
        "English": "Here's what I know about that",
        "Hausa": "Ga abin da na sani game da wannan",
        "Pidgin": "Na hear dis one well well",
        "French": "Voici ce que je sais à ce sujet"
    }
    prefix = greetings.get(lang, "Here's what I know")
    
    # Format based on content type
    if ":" in answer and "\n" in answer:  # Structured data
        return f"{prefix}:\n{answer}"
    elif len(answer.split()) > 20:  # Long text answer
        return f"{prefix}: {answer}"
    else:  # Short answer
        return answer  # Return as-is for simple answers

# Add this right after loading your JSON to verify its structure
'''print("=== JSON STRUCTURE VERIFICATION ===")
print(f"Total items: {len(AUN_DATA)}")
print("Sample items:")
for i, (key, value) in enumerate(AUN_DATA.items()):
    if i < 3:  # Print first 3 items
        print(f"Key: {key}")
        print(f"Question: {value.get('question')}")
        print(f"Answer type: {type(value.get('answer'))}")'''

# Cache Documents
DOCUMENT_CACHE = {}

def initialize_model():
    """Lazy load the model only when needed"""
    global tokenizer, chat_model
    if tokenizer is None or chat_model is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        chat_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        chat_model.eval()

def preload_documents():
    """Load documents into cache once at startup"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT topic, file_path FROM university_documents")
        for topic, path in cursor.fetchall():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    DOCUMENT_CACHE[topic.lower()] = f.read()
            except Exception as e:
                print(f"Failed to load {path}: {e}")
        conn.close()
    except Exception as e:
        print(f"Database error: {e}")

# Call this at startup
preload_documents()



origins = [
         "http://192.168.0.152:8080",
         "http://192.168.0.117:8080",
         'http://127.0.0.1:8000'
         
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,              # or use ["*"] to allow all
    allow_credentials=True,
    allow_methods=["*"],                
    allow_headers=["*"],                
)

model.Base.metadata.create_all(engine)

from sqlalchemy import create_engine, text

engine = create_engine("sqlite:///chatbotapp.db")
with engine.connect() as conn:
    pass
    #conn.execute(text("DROP TABLE IF EXISTS CHAT_HISTORY"))


#new addition- to allow endpoint without 0Auth2PasswordBearer
@app.get("/me", tags=['USER'])
def get_current_user(Authorization: str = Header(None), db: Session = Depends(get_db)):
    if Authorization is None or not Authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = Authorization.split("Bearer ")[1]
    payload = decode_token(token)

    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    student_id = payload.get("sub")
    user = db.query(model.Student).filter(model.Student.studentID == student_id).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {"studentID": user.studentID, "email": user.email, "user_id": user.user_id}       

@app.get("/")
def root():
    return{"message": "CORS is working!"}


#TO SIGN UP 
@app.post('/signup',status_code=status.HTTP_201_CREATED, 
          responses={201:{"model":schema.UserResponseModel}}, tags=['USER'])
def signup(request: schema.UserList, db:Session = Depends(get_db)):
   new_user = model.Student(studentID =request.studentID,
                         email=request.email,
                         password=Hash.bcrypt(request.password))
   db.add(new_user)
   db.commit() 
   db.refresh(new_user) 
   return new_user

                                                                                                                                                                                                                                          
#TO LOGIN
@app.post('/token', responses={200: {"model": schema.Token}}, tags=['USER'])
def login(request:schema.Login, db: Session = Depends(database.get_db)):
   user = db.query(model.Student).filter(model.Student.studentID == request.studentID).first()
   if not user:
      raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                          detail =f'Oops, we cannot find this student')
   if not Hash.verify(user.password, request.password):
       raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f'Incorrect password or studentID')
   
   access_token = JWToken.create_access_token(data = {'sub':user.studentID})
   refresh_token = JWToken.create_refresh_token(data={'sub': user.studentID})

   user.token = access_token
   db.commit()
   db.refresh(user)  # Refresh to update the instance
   
   return {'user_id': user.user_id,'access_token': access_token, 
           'token_type': 'bearer', 'refresh_token': refresh_token}


#TO LOGOUT
@app.post('/logout', status_code=status.HTTP_200_OK, tags=['USER'])
def logout(request: Request, db: Session = Depends(get_db)):
    """
    Logs out the user by extracting the token manually and blacklisting it.
    """
    # Extract Authorization header
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

    token = auth_header.split("Bearer ")[1]  # Extract the token

    # Check if token is blacklisted
    if is_token_blacklisted(token, db):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Token already blacklisted")

    # Decode the token to verify it's valid
    payload = decode_token(token)  # Ensure `decode_token` is implemented
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    # Blacklist the token
    blacklist_token(token, db)

    return {"message": "Successfully logged out"}


#PASSWORD RESET
'''@app.post("/reset-password", status_code=status.HTTP_200_OK, tags=['USER'])
def reset_password(request: schema.PasswordReset, db: Session = Depends(get_db)):
    # Find the student
    student = db.query(model.Student).filter(model.Student.studentID == request.studentID).first()
    if not student:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")

    # Verify old password
    if not Hash.verify(student.password, request.old_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Old password is incorrect")

    # Update with new hashed password
    student.password = Hash.bcrypt(request.new_password)
    db.commit()

    return {"message": "Password reset successful"}'''


@app.post("/request-password-reset", tags=['USER'])
def request_password_reset(data: schema.RequestResetPassword, db: Session = Depends(get_db)):
    user = db.query(model.Student).filter(model.Student.email == data.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="Email not registered")

    otp = generate_otp()
    expiry = datetime.utcnow() + timedelta(minutes=5)

    otp_entry = model.OTPRequest(email=data.email, otp=otp, expires_at=expiry)
    db.add(otp_entry)
    db.commit()

    send_otp_email(data.email, otp)
    return {"message": "OTP sent to email"}


@app.post("/reset-password", tags=['USER'])
def reset_password(data: schema.VerifyOTPReset, db: Session = Depends(get_db)):
    otp_entry = (
        db.query(model.OTPRequest)
        .filter(model.OTPRequest.email == data.email, model.OTPRequest.otp == data.otp)
        .first()
    )
    if not otp_entry or otp_entry.expires_at < datetime.utcnow():
        raise HTTPException(status_code=400, detail="Invalid or expired OTP")

    user = db.query(model.Student).filter(model.Student.email == data.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.password = Hash.bcrypt(data.new_password)
    db.commit()

    return {"message": "Password reset successful"}


#RETRIEVE CHAT HISTORY
@app.get("/chat-history", tags=['USER'])
def get_chat_history(user_id: int):
    # Connect to SQLite
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # Optional: Check if user exists in STUDENT table
    cur.execute("SELECT 1 FROM STUDENT WHERE user_id = ?", (user_id,))
    if not cur.fetchone():
        conn.close()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    # Query chat history
    cur.execute("""
        SELECT question, response, timestamp 
        FROM chat_history2 
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT 50
    """, (user_id,))
    
    rows = cur.fetchall()
    conn.close()
    
    chats = [{
        "question": row[0],
        "response": row[1],
        "timestamp": row[2]
    } for row in rows]
    
    return {"user_id": user_id, "chats": chats}

'''@app.get("/chat-history", response_model=schema.ChatHistoryResponse, tags=['USER'])
def get_chat_history(user_id: int, db: Session = Depends(get_db)):

    user = db.query(model.Student).filter(model.Student.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
                            detail="User not found")

    chat_history = db.query(model.ChatHistory).filter(model.ChatHistory.user_id == user_id).all()
    
    # Connect to SQLite
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Query chat history
    cursor.execute("""
    SELECT question, response, timestamp 
    FROM chat_history2 
    WHERE user_id = ?
    ORDER BY timestamp DESC
    LIMIT 50
    """, (user_id,))
    
    rows = cursor.fetchall()
    conn.close()
    
    # Format response
    chats = [{
        "question": row[0],
        "response": row[1],
        "timestamp": row[2]
    } for row in rows]
    
    return {"user_id": user_id, "chats": chat_history}'''


#CHATBOT ENDPOINT
'''# Load TinyLlama
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
chat_model = AutoModelForCausalLM.from_pretrained(model_id) #torch_dtype=torch.float32)
chat_model.eval()'''


'''def fetch_json_answer(message):
    for key, item in aun_data.items():
        q = item.get("question", "").lower()
        if q and q in message.lower():
            ans = item.get("answer")
            if isinstance(ans, dict):
                return json.dumps(ans, indent=2)
            elif isinstance(ans, list):
                return "\n".join(ans)
            return ans
    return None'''


'''def fetch_document_info(user_input):
    for topic, content in document_cache.items():
        if topic in user_input.lower():
            return content[:500]
    return None'''

'''def store_conversation(student_id: str, question: str, answer: str):
    current_time = datetime.now()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chat_history2 (user_id, question, response, timestamp) VALUES (?, ?, ?, ?)",
                   (student_id, question, answer, current_time))
    conn.commit()
    conn.close()

def build_prompt(user_input: str, lang_hint: str = "English"):
    return f"""
<|system|>
You are a helpful university assistant who can speak multiple languages including English, Hausa, Pidgin, and French. Respond to user queries clearly, politely, and concisely in {lang_hint}.
<|user|>
{user_input}
<|assistant|>
"""


#Chat Endpoint 
@app.get("/chat", tags=['USER'])
def chat(message: str = Query(...), student_id: str = Query(default="anonymous"), lang: str = Query(default="English")):
    message_lower = message.lower()
    
    json_answer = fetch_json_answer(message_lower)
    if json_answer:
        store_conversation(student_id, message, json_answer)
        return {"response": json_answer}

    fallback_reply = "I'm sorry, I don't have enough information to answer that based on what I know. Please ask another question."'''
    
'''doc_info = fetch_document_info(message_lower)

    if doc_info:
        prompt = build_prompt(f"{doc_info}\n\nUser question: {message}", lang)
    else:'''
'''prompt = build_prompt(message, lang)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(chat_model.device)

    try:
        with torch.no_grad():
            output = chat_model.generate(input_ids, max_new_tokens=80, do_sample=True, temperature=0.8)
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            reply = response.split("<|assistant|>")[-1].strip()
    except Exception as e:
        reply = f"Sorry, there was an error generating a response: {e}"'''

'''store_conversation(student_id, message, fallback_reply)

    return {"response": fallback_reply}'''

def find_faq_answer(user_query: str) -> Optional[str]:
    """Find the best matching FAQ answer with case-insensitive matching"""
    query = normalize_text(user_query)
    normalized_questions = {normalize_text(q): a for q, a in QUESTION_MAP.items()}
    
      # 1. Check for exact matches first
    if query in normalized_questions:
        return format_answer(normalized_questions[query])
    
    # 2. Check for partial matches (question contained in query)
    for question, answer in normalized_questions.items():
        if question in query:
            return format_answer(answer)
    
    # 3. Check for keyword matches (query contained in question)
    for question, answer in normalized_questions.items():
        if query in question:
            return format_answer(answer)
    
    return None
    '''# 1. Check for exact matches first
    if query in QUESTION_MAP:
        return format_answer(QUESTION_MAP[query])
    
    # 2. Check for partial matches (question contained in query)
    for question, answer in QUESTION_MAP.items():
        if question in query:
            return format_answer(answer)
    
    # 3. Check for keyword matches (query contained in question)
    for question, answer in QUESTION_MAP.items():
        if query in question:
            return format_answer(answer)
    
    return None'''


def fetch_json_answer(message: str) -> Optional[str]:
    message_lower = message.lower().strip()
    
    # Check for direct matches first
    if message_lower in QUESTION_MAP:
        return format_answer(QUESTION_MAP[message_lower])
    
    # Check for partial matches
    for question, answer in QUESTION_MAP.items():
        if question in message_lower:
            return format_answer(answer)
    
    return None

def format_answer(answer) -> str:
    """Handle different answer formats consistently"""
    if not answer:
        return "No answer available"
    if isinstance(answer, dict):
        return json.dumps(answer, indent=2)
    if isinstance(answer, list):
        return "\n".join(str(item) for item in answer)
    return str(answer)

def store_conversation(student_id: str, question: str, answer: str):
    """Store conversation in database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_history2 (user_id, question, response, timestamp) VALUES (?, ?, ?, ?)",
            (student_id, question, answer, datetime.now())
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Failed to store conversation: {e}")

def generate_model_response(prompt: str) -> str:
    """Generate response using the language model"""
    initialize_model()  # Ensure model is loaded
    
    try:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        with torch.no_grad():
            output = chat_model.generate(
                input_ids,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return response.split("<|assistant|>")[-1].strip()
    
    except Exception as e:
        print(f"Model generation error: {e}")
        return "Sorry, I encountered an error processing your request."

@app.get("/chat", tags=['USER'])
def chat(
    message: str = Query(...),
    student_id: str = Query(default="anonymous"),
    lang: str = Query(default="English")
):
    # First try to find an FAQ answer
    faq_answer = find_faq_answer(message)
    
    if faq_answer:
        # Enhance with conversational wrapper
        response = make_conversational(
            answer=faq_answer,
            question=message,
            lang=lang
        )
        store_conversation(student_id, message, response)
        return {"response": response}
    # If no FAQ match, use the language model
    prompt = f"""
    <|system|>
    You are a helpful university assistant who can speak multiple languages including English, Hausa, Pidgin, and French. 
    Respond to user queries clearly, politely, and concisely in {lang}.
    <|user|>
    {message}
    <|assistant|>
    """
    
    model_response = generate_model_response(prompt)
    store_conversation(student_id, message, model_response)
    
    return {"response": model_response}

@app.get("/faq-debug", tags=['DEBUG'])
def debug_faqs():
    """Endpoint to verify FAQ loading"""
    return {
        "total_questions": len(QUESTION_MAP),
        "sample_questions": list(QUESTION_MAP.keys())[:5],
        "sample_answers": [format_answer(list(QUESTION_MAP.values())[0])[:100] + "..." if QUESTION_MAP else None]
    }

'''@app.get("/chat", tags=['USER'])
def chat(
    message: str = Query(...),
    student_id: str = Query(default="anonymous"),
    lang: str = Query(default="English")
):
    # First try JSON lookup
    json_answer = fetch_json_answer(message)
    if json_answer:
        store_conversation(student_id, message, json_answer)
        return {"response": json_answer}
    
    # If no JSON answer, use model
    prompt = f"""
    <|system|>
    You are a helpful university assistant who can speak multiple languages including English, Hausa, Pidgin, and French. 
    Respond to user queries clearly, politely, and concisely in {lang}.
    <|user|>
    {message}
    <|assistant|>
    """
    
    model_response = generate_model_response(prompt)
    store_conversation(student_id, message, model_response)
    
    return {"response": model_response}'''


import uvicorn

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

