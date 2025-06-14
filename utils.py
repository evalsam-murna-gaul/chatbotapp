import random
import string

def generate_otp(length: int = 6) -> str:
    return ''.join(random.choices(string.digits, k=length))

import smtplib
from email.message import EmailMessage

def send_otp_email(to_email: str, otp: str):
    msg = EmailMessage()
    msg.set_content(f"Your OTP code is: {otp}")
    msg['Subject'] = 'Password Reset OTP'
    msg['From'] = 'your@email.com'
    msg['To'] = to_email

    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login('your@email.com', 'your_app_password')
        server.send_message(msg)
