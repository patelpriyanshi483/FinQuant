import sqlite3
import json
import uuid
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path='data.db'):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()

    def set_user_otp(self, email, otp, expiry):
        c = self.conn.cursor()
        c.execute("UPDATE users SET otp=?, otp_expiry=? WHERE email=?", (otp, expiry, email))
        self.conn.commit()

    def verify_user_otp(self, email, otp):
        c = self.conn.cursor()
        c.execute("SELECT otp, otp_expiry FROM users WHERE email=?", (email,))
        row = c.fetchone()
        if not row:
            return False
    
        stored_otp, expiry = row
        from datetime import datetime
    
        if not stored_otp or not expiry:
            return False
        
        try:
            expiry_time = datetime.fromisoformat(expiry)
        except Exception:
            return False
    
        return stored_otp == otp and datetime.now() < expiry_time



    def create_tables(self):
        c = self.conn.cursor()
        
        # Users table
        c.execute('''CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            company_name TEXT,
            email TEXT UNIQUE,
            password TEXT,
            role TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            otp TEXT,
            otp_expiry TEXT
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
            type TEXT,  -- NEW
            investment_goal TEXT,  -- NEW
            target_amount REAL DEFAULT 0,  -- NEW
            created_at TEXT,
            updated_at TEXT,  -- NEW
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
            c.execute("SELECT id, company_name, email, password, role, created_at FROM users WHERE email=?", (email,))
            row = c.fetchone()
            # row = c.fetchone()
            if row:
                return {
                    'id': row[0],
                    'company_name': row[1],
                    'email': row[2],
                    'password': row[3],
                    'role': row[4],
                    'created_at': row[5] 
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
        
#     def update_portfolio(self, portfolio_id, name, description=None, portfolio_type=None, 
#                         risk_tolerance=None, investment_goal=None, target_amount=None):
        
#         print("DEBUG update_portfolio called with:", {
#     "portfolio_id": portfolio_id,
#     "name": name,
#     "description": description,
#     "portfolio_type": portfolio_type,
#     "risk_tolerance": risk_tolerance,
#     "investment_goal": investment_goal,
#     "target_amount": target_amount
# })
#         """Update portfolio details"""
#         try:
#             # Build update query dynamically based on provided parameters
#             update_fields = ["name = ?"]
#             values = [name]
            
#             if description is not None:
#                 update_fields.append("description = ?")
#                 values.append(description)
            
#             if portfolio_type is not None:
#                 update_fields.append("type = ?")
#                 values.append(portfolio_type)
            
#             if risk_tolerance is not None:
#                 update_fields.append("risk_tolerance = ?")
#                 values.append(risk_tolerance)
            
#             if investment_goal is not None:
#                 update_fields.append("investment_goal = ?")
#                 values.append(investment_goal)
            
#             if target_amount is not None:
#                 update_fields.append("target_amount = ?")
#                 values.append(target_amount)
            
#             # Add updated timestamp
#             update_fields.append("updated_at = CURRENT_TIMESTAMP")
#             values.append(portfolio_id)
            
#             query = f"""
#             UPDATE portfolios 
#             SET {', '.join(update_fields)}
#             WHERE id = ?
#             """
            
#             cursor = self.conn.cursor()
#             cursor.execute(query, values)
#             self.conn.commit()
            
#             return cursor.rowcount > 0
            
#         except sqlite3.Error as e:
#             print(f"Error updating portfolio: {e}")
#             return False
    
    # def get_portfolio_by_id(self, portfolio_id):
    #     """Get portfolio by ID - SINGLE CORRECT VERSION"""
    #     try:
    #         cursor = self.conn.cursor()
    #         cursor.execute("SELECT * FROM portfolios WHERE id = ?", (portfolio_id,))
    #         row = cursor.fetchone()
            
    #         if row:
    #             columns = [description[0] for description in cursor.description]
    #             portfolio = dict(zip(columns, row))
    #             print(f"DEBUG: Found portfolio {portfolio_id}: {portfolio}")
    #             return portfolio
    #         else:
    #             print(f"DEBUG: Portfolio {portfolio_id} not found")
    #             return None
                
    #     except Exception as e:
    #         print(f"Error getting portfolio: {e}")
    #         return None
        
    # def migrate_database(self):
    #     """Add missing columns to existing portfolios table"""
    #     db = DatabaseManager()
    #     cursor = db.conn.cursor()
        
    #     # Check if columns exist and add them if they don't
    #     try:
    #         cursor.execute("ALTER TABLE portfolios ADD COLUMN type TEXT DEFAULT 'Custom'")
    #         print("Added 'type' column")
    #     except:
    #         pass
        
    #     try:
    #         cursor.execute("ALTER TABLE portfolios ADD COLUMN investment_goal TEXT DEFAULT 'Wealth Building'")
    #         print("Added 'investment_goal' column")
    #     except:
    #         pass
    
    #     try:
    #         cursor.execute("ALTER TABLE portfolios ADD COLUMN target_amount REAL DEFAULT 0")
    #         print("Added 'target_amount' column")
    #     except:
    #         pass
    
    #     try:
    #         cursor.execute("ALTER TABLE portfolios ADD COLUMN updated_at TEXT DEFAULT CURRENT_TIMESTAMP")
    #         print("Added 'updated_at' column")
    #     except:
    #         pass
    
    #     db.conn.commit()
    #     print("Database migration completed")


    # def get_user_portfolios(self, user_id):
    #     try:
    #         c = self.conn.cursor()
    #         c.execute("SELECT * FROM portfolios WHERE user_id=? ORDER BY created_at DESC", (user_id,))
    #         rows = c.fetchall()
    #         portfolios = []
    #         for row in rows:
    #             portfolios.append({
    #                 'id': row[0],
    #                 'user_id': row[1],
    #                 'name': row[2],
    #                 'description': row[3],
    #                 'tags': row[4],
    #                 'initial_value': row[5],
    #                 'risk_tolerance': row[6],
    #                 'expected_return': row[7],
    #                 'volatility': row[8],
    #                 'sharpe_ratio': row[9],
    #                 'created_at': row[10]
    #             })
    #         return portfolios
    #     except Exception as e:
    #         print(f"Error getting user portfolios: {e}")
    #         return []

    def get_user_portfolios(self, user_id):
        """Fetch all portfolios for a user, sorted by creation date"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM portfolios WHERE user_id=? ORDER BY created_at DESC", (user_id,))
            rows = cursor.fetchall()
    
            portfolios = []
            if rows:
                columns = [desc[0] for desc in cursor.description]  # get column names dynamically
                for row in rows:
                    portfolios.append(dict(zip(columns, row)))  # map into dict
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