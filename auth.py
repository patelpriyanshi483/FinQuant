# auth.py
import uuid
import hashlib
import random, string
from datetime import datetime, timedelta
from utils import send_email 

class AuthManager:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.current_user = None
        
    def generate_otp(self):
        return ''.join(random.choices(string.digits, k=6))

    def login_with_otp(self, email, password):
        user = self.db_manager.get_user_by_email(email)
        if not user or user['password'] != self.hash_password(password):
            return None, "Invalid email or password"
    
    # Generate OTP
        otp = self.generate_otp()
        expiry = datetime.now() + timedelta(minutes=5)
        self.db_manager.set_user_otp(email, otp, expiry.isoformat())

    # Send email
        send_email(email, "Your Login OTP", f"Your OTP is: {otp}\nIt expires in 5 minutes.")
        return user, "OTP sent to your email"
    
    def set_logged_in(self, user):
        import streamlit as st
        st.session_state["user"] = user
        st.session_state["is_logged_in"] = True
        

    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()

    def register(self, company_name, email, password, role):
        user = {
            'id': str(uuid.uuid4()),
            'company_name': company_name,
            'email': email,
            'password': self.hash_password(password),
            'role': role
        }
        return self.db_manager.register_user(user)

    def login(self, email, password):
        user = self.db_manager.get_user_by_email(email)
        if user and user['password'] == self.hash_password(password):
            self.current_user = user
            return True
        return False

    def logout(self):
        import streamlit as st
        self.current_user = None
        if "user" in st.session_state:
            del st.session_state["user"]
        if "is_logged_in" in st.session_state:
            del st.session_state["is_logged_in"]
        st.rerun()  # Refresh the app to show the login page
    
    def is_authenticated(self):
        return self.current_user is not None

    def get_current_user(self):
        import streamlit as st
        return st.session_state.get("user")

