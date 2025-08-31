# utils.py
import io
import base64
import smtplib
from email.mime.text import MIMEText


def get_table_download_link(df, filename='data.csv'):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Encode to base64
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href


def send_email(to_email, subject, body):
    sender = "finquantse@gmail.com"
    password = "xrhv blhb nznk fgly".replace(" ", "")  # Gmail App Password

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to_email

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
    except Exception as e:
        print(f"Error sending email: {e}")
