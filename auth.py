# auth.py
import uuid
import hashlib

class AuthManager:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.current_user = None

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
        self.current_user = None

    def is_authenticated(self):
        return self.current_user is not None

    def get_current_user(self):
        return self.current_user
