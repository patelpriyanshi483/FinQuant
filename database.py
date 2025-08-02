# # database.py - Fixed version
# import sqlite3
# import json
# import uuid
# from datetime import datetime

# class DatabaseManager:
#     def __init__(self, db_path='data.db'):
#         self.conn = sqlite3.connect(db_path, check_same_thread=False)
#         self.create_tables()

#     def create_tables(self):
#         c = self.conn.cursor()
        
#         # Users table
#         c.execute('''CREATE TABLE IF NOT EXISTS users (
#             id TEXT PRIMARY KEY,
#             company_name TEXT,
#             email TEXT UNIQUE,
#             password TEXT,
#             role TEXT,
#             created_at TEXT DEFAULT CURRENT_TIMESTAMP
#         )''')
        
#         # Portfolios table
#         c.execute('''CREATE TABLE IF NOT EXISTS portfolios (
#             id TEXT PRIMARY KEY,
#             user_id TEXT,
#             name TEXT,
#             description TEXT,
#             tags TEXT,
#             initial_value REAL,
#             risk_tolerance TEXT,
#             expected_return REAL DEFAULT 0,
#             volatility REAL DEFAULT 0,
#             sharpe_ratio REAL DEFAULT 0,
#             created_at TEXT,
#             FOREIGN KEY (user_id) REFERENCES users (id)
#         )''')
        
#         # Holdings table
#         c.execute('''CREATE TABLE IF NOT EXISTS holdings (
#             id TEXT PRIMARY KEY,
#             portfolio_id TEXT,
#             ticker TEXT,
#             investment REAL,
#             shares REAL DEFAULT 0,
#             created_at TEXT DEFAULT CURRENT_TIMESTAMP,
#             FOREIGN KEY (portfolio_id) REFERENCES portfolios (id)
#         )''')
        
#         self.conn.commit()

#     def register_user(self, user):
#         try:
#             c = self.conn.cursor()
#             created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#             c.execute("INSERT INTO users (id, company_name, email, password, role, created_at) VALUES (?, ?, ?, ?, ?, ?)",
#                       (user['id'], user['company_name'], user['email'], user['password'], user['role'], created_at))
#             self.conn.commit()
#             return True
#         except sqlite3.IntegrityError as e:
#             print(f"User registration error: {e}")
#             return False
#         except Exception as e:
#             print(f"Unexpected error during user registration: {e}")
#             return False

#     def get_user_by_email(self, email):
#         try:
#             c = self.conn.cursor()
#             c.execute("SELECT * FROM users WHERE email=?", (email,))
#             row = c.fetchone()
#             if row:
#                 return {
#                     'id': row[0],
#                     'company_name': row[1],
#                     'email': row[2],
#                     'password': row[3],
#                     'role': row[4],
#                     'created_at': row[5] if len(row) > 5 else None
#                 }
#             return None
#         except Exception as e:
#             print(f"Error getting user by email: {e}")
#             return None

#     def create_portfolio(self, user_id, portfolio_data):
#         try:
#             portfolio_id = str(uuid.uuid4())
#             created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
#             c = self.conn.cursor()
            
#             # Insert portfolio
#             c.execute('''INSERT INTO portfolios 
#                         (id, user_id, name, description, tags, initial_value, risk_tolerance, 
#                          expected_return, volatility, sharpe_ratio, created_at)
#                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
#                       (portfolio_id, user_id, portfolio_data['name'], portfolio_data['description'],
#                        portfolio_data['tags'], portfolio_data['initial_value'], portfolio_data['risk_tolerance'],
#                        0, 0, 0, created_at))
            
#             # Insert holdings
#             for holding in portfolio_data['holdings']:
#                 holding_id = str(uuid.uuid4())
#                 # Only use investment amount, shares will be calculated later if needed
#                 c.execute('''INSERT INTO holdings (id, portfolio_id, ticker, investment, shares)
#              VALUES (?, ?, ?, ?, ?)''',
#           (holding_id, portfolio_id, holding['ticker'], holding['investment'], holding['shares']))

            
#             self.conn.commit()
#             print(f"Portfolio created successfully: {portfolio_id} with {len(portfolio_data['holdings'])} holdings")
#             return True
            
#         except Exception as e:
#             print(f"Error creating portfolio: {e}")
#             self.conn.rollback()
#             return False

#     def get_user_portfolios(self, user_id):
#         try:
#             c = self.conn.cursor()
#             c.execute("SELECT * FROM portfolios WHERE user_id=? ORDER BY created_at DESC", (user_id,))
#             rows = c.fetchall()
#             portfolios = []
#             for row in rows:
#                 portfolios.append({
#                     'id': row[0],
#                     'user_id': row[1],
#                     'name': row[2],
#                     'description': row[3],
#                     'tags': row[4],
#                     'initial_value': row[5],
#                     'risk_tolerance': row[6],
#                     'expected_return': row[7],
#                     'volatility': row[8],
#                     'sharpe_ratio': row[9],
#                     'created_at': row[10]
#                 })
#             return portfolios
#         except Exception as e:
#             print(f"Error getting user portfolios: {e}")
#             return []

#     def get_portfolio_holdings(self, portfolio_id):
#         try:
#             c = self.conn.cursor()
#             c.execute("SELECT ticker, investment, shares FROM holdings WHERE portfolio_id=?", (portfolio_id,))
#             rows = c.fetchall()
#             holdings = []
#             for row in rows:
#                 holdings.append({
#                     'ticker': row[0],
#                     'investment': row[1],
#                     'shares': row[2]
#                 })
#             print(f"Retrieved {len(holdings)} holdings for portfolio {portfolio_id}")
#             return holdings
#         except Exception as e:
#             print(f"Error getting portfolio holdings: {e}")
#             return []

#     def update_portfolio_metrics(self, portfolio_id, expected_return, volatility, sharpe_ratio):
#         try:
#             c = self.conn.cursor()
#             c.execute('''UPDATE portfolios 
#                         SET expected_return=?, volatility=?, sharpe_ratio=?
#                         WHERE id=?''',
#                       (expected_return, volatility, sharpe_ratio, portfolio_id))
#             self.conn.commit()
#             return True
#         except Exception as e:
#             print(f"Error updating portfolio metrics: {e}")
#             return False

#     def delete_portfolio(self, portfolio_id):
#         try:
#             c = self.conn.cursor()
#             # Delete holdings first (foreign key constraint)
#             c.execute("DELETE FROM holdings WHERE portfolio_id=?", (portfolio_id,))
#             # Then delete portfolio
#             c.execute("DELETE FROM portfolios WHERE id=?", (portfolio_id,))
#             self.conn.commit()
#             print(f"Portfolio {portfolio_id} deleted successfully")
#             return True
#         except Exception as e:
#             print(f"Error deleting portfolio: {e}")
#             self.conn.rollback()
#             return False

#     def get_all_users(self):
#         """Get all users for admin panel"""
#         try:
#             c = self.conn.cursor()
#             c.execute("SELECT * FROM users ORDER BY created_at DESC")
#             rows = c.fetchall()
#             users = []
#             for row in rows:
#                 users.append({
#                     'id': row[0],
#                     'company_name': row[1],
#                     'email': row[2],
#                     'password': row[3],  # Don't expose this in real apps
#                     'role': row[4],
#                     'created_at': row[5] if len(row) > 5 else 'Unknown'
#                 })
#             return users
#         except Exception as e:
#             print(f"Error getting all users: {e}")
#             return []

#     def get_portfolio_by_id(self, portfolio_id):
#         """Get a specific portfolio by ID"""
#         try:
#             c = self.conn.cursor()
#             c.execute("SELECT * FROM portfolios WHERE id=?", (portfolio_id,))
#             row = c.fetchone()
#             if row:
#                 return {
#                     'id': row[0],
#                     'user_id': row[1],
#                     'name': row[2],
#                     'description': row[3],
#                     'tags': row[4],
#                     'initial_value': row[5],
#                     'risk_tolerance': row[6],
#                     'expected_return': row[7],
#                     'volatility': row[8],
#                     'sharpe_ratio': row[9],
#                     'created_at': row[10]
#                 }
#             return None
#         except Exception as e:
#             print(f"Error getting portfolio by ID: {e}")
#             return None

#     def close(self):
#         """Close database connection"""
#         if self.conn:
#             self.conn.close()

#     def __del__(self):
#         """Cleanup when object is destroyed"""
#         try:
#             if hasattr(self, 'conn') and self.conn:
#                 self.conn.close()
#         except:
#             pass

# database.py - Fixed version
import sqlite3
import json
import uuid
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path='data.db'):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        c = self.conn.cursor()
        
        # Users table
        c.execute('''CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            company_name TEXT,
            email TEXT UNIQUE,
            password TEXT,
            role TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )''')
        
        # Portfolios table
        c.execute('''CREATE TABLE IF NOT EXISTS portfolios (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            name TEXT,
            description TEXT,
            tags TEXT,
            initial_value REAL,
            risk_tolerance TEXT,
            expected_return REAL DEFAULT 0,
            volatility REAL DEFAULT 0,
            sharpe_ratio REAL DEFAULT 0,
            created_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )''')
        
        # Holdings table
        c.execute('''CREATE TABLE IF NOT EXISTS holdings (
            id TEXT PRIMARY KEY,
            portfolio_id TEXT,
            ticker TEXT,
            investment REAL,
            shares REAL DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (portfolio_id) REFERENCES portfolios (id)
        )''')
        
        self.conn.commit()

    def register_user(self, user):
        try:
            c = self.conn.cursor()
            created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            c.execute("INSERT INTO users (id, company_name, email, password, role, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                      (user['id'], user['company_name'], user['email'], user['password'], user['role'], created_at))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError as e:
            print(f"User registration error: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error during user registration: {e}")
            return False

    def get_user_by_email(self, email):
        try:
            c = self.conn.cursor()
            c.execute("SELECT * FROM users WHERE email=?", (email,))
            row = c.fetchone()
            if row:
                return {
                    'id': row[0],
                    'company_name': row[1],
                    'email': row[2],
                    'password': row[3],
                    'role': row[4],
                    'created_at': row[5] if len(row) > 5 else None
                }
            return None
        except Exception as e:
            print(f"Error getting user by email: {e}")
            return None

    def create_portfolio(self, user_id, portfolio_data):
        try:
            portfolio_id = str(uuid.uuid4())
            created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            c = self.conn.cursor()
            
            # Insert portfolio
            c.execute('''INSERT INTO portfolios 
                        (id, user_id, name, description, tags, initial_value, risk_tolerance, 
                         expected_return, volatility, sharpe_ratio, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                      (portfolio_id, user_id, portfolio_data['name'], portfolio_data['description'],
                       portfolio_data['tags'], portfolio_data['initial_value'], portfolio_data['risk_tolerance'],
                       0, 0, 0, created_at))
            
            # Insert holdings - FIXED: Only use ticker and investment, shares defaults to 0
            for holding in portfolio_data['holdings']:
                holding_id = str(uuid.uuid4())
                c.execute('''INSERT INTO holdings (id, portfolio_id, ticker, investment, shares)
                            VALUES (?, ?, ?, ?, ?)''',
                          (holding_id, portfolio_id, holding['ticker'], holding['investment'], 0.0))
            
            self.conn.commit()
            print(f"Portfolio created successfully: {portfolio_id} with {len(portfolio_data['holdings'])} holdings")
            return True
            
        except Exception as e:
            print(f"Error creating portfolio: {e}")
            self.conn.rollback()
            return False

    def get_user_portfolios(self, user_id):
        try:
            c = self.conn.cursor()
            c.execute("SELECT * FROM portfolios WHERE user_id=? ORDER BY created_at DESC", (user_id,))
            rows = c.fetchall()
            portfolios = []
            for row in rows:
                portfolios.append({
                    'id': row[0],
                    'user_id': row[1],
                    'name': row[2],
                    'description': row[3],
                    'tags': row[4],
                    'initial_value': row[5],
                    'risk_tolerance': row[6],
                    'expected_return': row[7],
                    'volatility': row[8],
                    'sharpe_ratio': row[9],
                    'created_at': row[10]
                })
            return portfolios
        except Exception as e:
            print(f"Error getting user portfolios: {e}")
            return []

    def get_portfolio_holdings(self, portfolio_id):
        try:
            c = self.conn.cursor()
            c.execute("SELECT ticker, investment, shares FROM holdings WHERE portfolio_id=?", (portfolio_id,))
            rows = c.fetchall()
            holdings = []
            for row in rows:
                holdings.append({
                    'ticker': row[0],
                    'investment': row[1],
                    'shares': row[2]
                })
            print(f"Retrieved {len(holdings)} holdings for portfolio {portfolio_id}")
            return holdings
        except Exception as e:
            print(f"Error getting portfolio holdings: {e}")
            return []

    def update_portfolio_metrics(self, portfolio_id, expected_return, volatility, sharpe_ratio):
        try:
            c = self.conn.cursor()
            c.execute('''UPDATE portfolios 
                        SET expected_return=?, volatility=?, sharpe_ratio=?
                        WHERE id=?''',
                      (expected_return, volatility, sharpe_ratio, portfolio_id))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error updating portfolio metrics: {e}")
            return False

    def delete_portfolio(self, portfolio_id):
        try:
            c = self.conn.cursor()
            # Delete holdings first (foreign key constraint)
            c.execute("DELETE FROM holdings WHERE portfolio_id=?", (portfolio_id,))
            # Then delete portfolio
            c.execute("DELETE FROM portfolios WHERE id=?", (portfolio_id,))
            self.conn.commit()
            print(f"Portfolio {portfolio_id} deleted successfully")
            return True
        except Exception as e:
            print(f"Error deleting portfolio: {e}")
            self.conn.rollback()
            return False

    def get_all_users(self):
        """Get all users for admin panel"""
        try:
            c = self.conn.cursor()
            c.execute("SELECT * FROM users ORDER BY created_at DESC")
            rows = c.fetchall()
            users = []
            for row in rows:
                users.append({
                    'id': row[0],
                    'company_name': row[1],
                    'email': row[2],
                    'password': row[3],  # Don't expose this in real apps
                    'role': row[4],
                    'created_at': row[5] if len(row) > 5 else 'Unknown'
                })
            return users
        except Exception as e:
            print(f"Error getting all users: {e}")
            return []

    def get_portfolio_by_id(self, portfolio_id):
        """Get a specific portfolio by ID"""
        try:
            c = self.conn.cursor()
            c.execute("SELECT * FROM portfolios WHERE id=?", (portfolio_id,))
            row = c.fetchone()
            if row:
                return {
                    'id': row[0],
                    'user_id': row[1],
                    'name': row[2],
                    'description': row[3],
                    'tags': row[4],
                    'initial_value': row[5],
                    'risk_tolerance': row[6],
                    'expected_return': row[7],
                    'volatility': row[8],
                    'sharpe_ratio': row[9],
                    'created_at': row[10]
                }
            return None
        except Exception as e:
            print(f"Error getting portfolio by ID: {e}")
            return None

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            if hasattr(self, 'conn') and self.conn:
                self.conn.close()
        except:
            pass