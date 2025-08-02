import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
import uuid
import io
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from database import DatabaseManager
from auth import AuthManager
from portfolio_analyzer import PortfolioAnalyzer
from ml_recommender import MLRecommender
from utils import get_table_download_link
import report_generator

# Configure Streamlit
st.set_page_config(
    page_title="FinQuant Pro - Portfolio Management",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize managers
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = DatabaseManager()

if 'auth_manager' not in st.session_state:
    st.session_state.auth_manager = AuthManager(st.session_state.db_manager)

if 'portfolio_analyzer' not in st.session_state:
    st.session_state.portfolio_analyzer = PortfolioAnalyzer()

if 'ml_recommender' not in st.session_state:
    st.session_state.ml_recommender = MLRecommender()

# Initialize navigation state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Dashboard'

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sidebar-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .portfolio-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f8f9fa;
    }
    .risk-card {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    .risk-low { border-left-color: #28a745; background-color: #d4edda; }
    .risk-medium { border-left-color: #ffc107; background-color: #fff3cd; }
    .risk-high { border-left-color: #dc3545; background-color: #f8d7da; }
</style>
""", unsafe_allow_html=True)

def calculate_weights_from_investments(holdings):
    """Calculate weights from investment amounts"""
    total_investment = sum([h['investment'] for h in holdings])
    if total_investment == 0:
        return holdings
    
    for holding in holdings:
        holding['weight'] = holding['investment'] / total_investment
    
    return holdings

def generate_portfolio_report(portfolio):
    holdings = st.session_state.db_manager.get_portfolio_holdings(portfolio['id'])
    holdings = calculate_weights_from_investments(holdings)
    tickers = [h['ticker'] for h in holdings]
    weights = [h['weight'] for h in holdings]
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()
    analysis = st.session_state.portfolio_analyzer.analyze_portfolio(tickers, weights, start_date, end_date, '^GSPC')
    pdf = report_generator.generate_pdf_report(portfolio, analysis)
    return pdf

def main():
    if not st.session_state.auth_manager.is_authenticated():
        show_auth_page()
        return

    user = st.session_state.auth_manager.get_current_user()
    st.sidebar.markdown(f"### Welcome, {user['company_name']}!")
    st.sidebar.markdown(f"**Role:** {user['role'].title()}")
    
    if user['role'] == 'admin':
        pages = ['Dashboard', 'My Portfolios', 'Analysis', 'ML Recommendations', 'Admin Panel', 'Settings']
    else:
        pages = ['Dashboard', 'My Portfolios', 'Analysis', 'ML Recommendations', 'Settings']
    
    selected_page = st.sidebar.radio("Navigation", pages, 
                                   index=pages.index(st.session_state.current_page) if st.session_state.current_page in pages else 0,
                                   key="nav_radio")
    
    if selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page
    
    if st.sidebar.button("Logout", key="logout_btn"):
        st.session_state.auth_manager.logout()
        st.rerun()
    
    if selected_page == 'Dashboard':
        show_dashboard()
    elif selected_page == 'My Portfolios':
        show_portfolios_page()
    elif selected_page == 'Analysis':
        show_analysis_page()
    elif selected_page == 'ML Recommendations':
        show_ml_recommendations()
    elif selected_page == 'Admin Panel' and user['role'] == 'admin':
        show_admin_panel()
    elif selected_page == 'Settings':
        show_settings_page()

def show_auth_page():
    st.markdown('<h1 class="main-header">🏦 FinQuant Pro</h1>', unsafe_allow_html=True)
    st.markdown("### Professional Portfolio Management Platform")
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    with tab1:
        st.subheader("Login to Your Account")
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                if st.session_state.auth_manager.login(email, password):
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
    with tab2:
        st.subheader("Create New Account")
        with st.form("signup_form"):
            company_name = st.text_input("Company Name")
            email = st.text_input("Email Address")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            role = st.selectbox("Role", ["user", "admin"])
            submitted = st.form_submit_button("Create Account")
            if submitted:
                if password != confirm_password:
                    st.error("Passwords do not match")
                elif st.session_state.auth_manager.register(company_name, email, password, role):
                    st.success("Account created successfully! Please login.")
                else:
                    st.error("Registration failed. Email may already exist.")

def show_dashboard():
    st.markdown('<h1 class="main-header">📊 Dashboard</h1>', unsafe_allow_html=True)
    user = st.session_state.auth_manager.get_current_user()
    portfolios = st.session_state.db_manager.get_user_portfolios(user['id'])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='metric-card'><h3>{len(portfolios)}</h3><p>Total Portfolios</p></div>", unsafe_allow_html=True)
    with col2:
        total_value = sum([p.get('initial_value', 0) for p in portfolios])
        st.markdown(f"<div class='metric-card'><h3>${total_value:,.0f}</h3><p>Total Portfolio Value</p></div>", unsafe_allow_html=True)
    with col3:
        if portfolios:
            avg_return = np.mean([p.get('expected_return', 0) for p in portfolios])
            st.markdown(f"<div class='metric-card'><h3>{avg_return:.2%}</h3><p>Avg Expected Return</p></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='metric-card'><h3>0.00%</h3><p>Avg Expected Return</p></div>", unsafe_allow_html=True)
    with col4:
        if portfolios:
            avg_sharpe = np.mean([p.get('sharpe_ratio', 0) for p in portfolios])
            st.markdown(f"<div class='metric-card'><h3>{avg_sharpe:.3f}</h3><p>Avg Sharpe Ratio</p></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='metric-card'><h3>0.000</h3><p>Avg Sharpe Ratio</p></div>", unsafe_allow_html=True)
    
    st.subheader("Recent Portfolios")
    if portfolios:
        recent_portfolios = sorted(portfolios, key=lambda x: x['created_at'], reverse=True)[:5]
        for portfolio in recent_portfolios:
            with st.expander(f"📊 {portfolio['name']} - {portfolio['description'][:50]}..."):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Created:** {portfolio['created_at']}")
                    st.write(f"**Tags:** {portfolio['tags']}")
                    st.write(f"**Total Investment:** ${portfolio.get('initial_value', 0):,.2f}")
                with col2:
                    st.write(f"**Expected Return:** {portfolio.get('expected_return', 0):.2%}")
                    st.write(f"**Volatility:** {portfolio.get('volatility', 0):.2%}")
                with col3:
                    st.write(f"**Sharpe Ratio:** {portfolio.get('sharpe_ratio', 0):.3f}")
                    if st.button(f"Analyze", key=f"analyze_{portfolio['id']}"):
                        st.session_state.selected_portfolio_id = portfolio['id']
                        st.session_state.current_page = 'Analysis'
                        st.rerun()
    else:
        st.info("No portfolios found. Create your first portfolio!")
        if st.button("Create Portfolio"):
            st.session_state.current_page = 'My Portfolios'
            st.rerun()

def show_portfolios_page():
    st.markdown('<h1 class="main-header">📁 My Portfolios</h1>', unsafe_allow_html=True)
    user = st.session_state.auth_manager.get_current_user()
    with st.expander("➕ Create New Portfolio", expanded=False):
        create_portfolio_form(user['id'])
    portfolios = st.session_state.db_manager.get_user_portfolios(user['id'])
    if portfolios:
        st.subheader(f"Your Portfolios ({len(portfolios)})")
        col1, col2 = st.columns([3, 1])
        with col1:
            search_term = st.text_input("🔍 Search portfolios...", key="portfolio_search")
        with col2:
            sort_by = st.selectbox("Sort by", ["Created Date", "Name", "Investment Value"], key="portfolio_sort")
        filtered_portfolios = portfolios
        if search_term:
            filtered_portfolios = [p for p in portfolios if search_term.lower() in p['name'].lower() or search_term.lower() in p['description'].lower() or search_term.lower() in p['tags'].lower()]
        if sort_by == "Name":
            filtered_portfolios = sorted(filtered_portfolios, key=lambda x: x['name'])
        elif sort_by == "Investment Value":
            filtered_portfolios = sorted(filtered_portfolios, key=lambda x: x.get('initial_value', 0), reverse=True)
        else:
            filtered_portfolios = sorted(filtered_portfolios, key=lambda x: x['created_at'], reverse=True)
        for portfolio in filtered_portfolios:
            display_portfolio_card(portfolio)
    else:
        st.info("No portfolios found. Create your first portfolio above!")

def generate_csv_template():
    """Generate a CSV template for portfolio upload"""
    sample_data = [
        ['ticker', 'investment'],
        ['AAPL', '25000'],
        ['GOOGL', '20000'],
        ['MSFT', '15000'],
        ['TSLA', '10000'],
        ['NVDA', '12000'],
        ['AMZN', '18000']
    ]
    csv_content = '\n'.join([','.join(row) for row in sample_data])
    return csv_content

def create_portfolio_form(user_id):
    with st.form("create_portfolio"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Portfolio Name*")
            description = st.text_area("Description")
            tags = st.text_input("Tags (comma-separated)")
        with col2:
            risk_tolerance = st.selectbox("Risk Tolerance", ["Conservative", "Moderate", "Aggressive"])
        
        st.subheader("Portfolio Holdings")
        input_method = st.radio("Input Method", ["Manual Entry", "Upload CSV"])
        holdings_data = []
        total_investment = 0
        
        if input_method == "Manual Entry":
            if 'num_holdings' not in st.session_state:
                st.session_state.num_holdings = 3
            
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.form_submit_button("Add Holding"):
                    st.session_state.num_holdings += 1
                    st.rerun()
            
            st.write("**Enter your investments:**")
            
            # Create headers
            col_headers = st.columns([3, 2, 1])
            with col_headers[0]:
                st.write("**Ticker Symbol**")
            with col_headers[1]:
                st.write("**Investment Amount ($)**")
            with col_headers[2]:
                st.write("**Remove**")
            
            investment_amounts = []
            valid_holdings = []
            
            for i in range(st.session_state.num_holdings):
                cols = st.columns([3, 2, 1])
                
                with cols[0]:
                    ticker = st.text_input(f"Ticker", key=f"ticker_{i}", placeholder="e.g., AAPL")
                
                with cols[1]:
                    investment = st.number_input(
                        f"Investment", 
                        min_value=0.0, 
                        value=0.0, 
                        step=100.0,
                        key=f"investment_{i}",
                        format="%.2f"
                    )
                
                with cols[2]:
                    if st.form_submit_button(f"❌ Remove {i}"):
                        if st.session_state.num_holdings > 1:
                            st.session_state.num_holdings -= 1
                            st.rerun()
                
                if ticker and ticker.strip() and investment > 0:
                    valid_holdings.append({
                        'ticker': ticker.strip().upper(),
                        'investment': investment
                    })
                    investment_amounts.append(investment)
            
            total_investment = sum(investment_amounts)
            
            if total_investment > 0 and valid_holdings:
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Investment", f"${total_investment:,.2f}")
                with col2:
                    st.metric("Number of Holdings", len(valid_holdings))
                with col3:
                    avg_weight = 100 / len(valid_holdings)
                    st.metric("Average Weight", f"{avg_weight:.1f}%")
                
                st.subheader("Portfolio Allocation Summary")
                summary_data = []
                for holding in valid_holdings:
                    weight = (holding['investment'] / total_investment)
                    summary_data.append({
                        'Ticker': holding['ticker'],
                        'Investment': f"${holding['investment']:,.2f}",
                        'Weight': f"{weight*100:.1f}%"
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                holdings_data = valid_holdings

        else:
            col1, col2 = st.columns([3, 1])
            with col1:
                uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            with col2:
                csv_template = generate_csv_template()
                st.download_button(
                    label="📥 Download Template",
                    data=csv_template,
                    file_name="portfolio_template.csv",
                    mime="text/csv",
                    help="Download a sample CSV template"
                )
            
            st.info("**CSV Format:** Use columns 'ticker' and 'investment'")
            with st.expander("📋 View CSV Format Example"):
                st.code("""ticker,investment\nAAPL,25000\nGOOGL,20000\nMSFT,15000\nTSLA,10000""")
            
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    df.columns = df.columns.str.lower().str.strip()
                    if 'ticker' not in df.columns or 'investment' not in df.columns:
                        st.error("CSV must contain 'ticker' and 'investment' columns")
                    else:
                        df = df[(df['investment'] > 0) & (df['ticker'].notna()) & (df['ticker'] != '')]
                        df['ticker'] = df['ticker'].str.strip().str.upper()
                        total_investment = df['investment'].sum()
                        df['weight'] = (df['investment'] / total_investment * 100).round(1)
                        display_df = df.copy()
                        display_df['investment'] = display_df['investment'].apply(lambda x: f"${x:,.2f}")
                        display_df['weight'] = display_df['weight'].apply(lambda x: f"{x:.1f}%")
                        st.dataframe(display_df[['ticker', 'investment', 'weight']], use_container_width=True)
                        holdings_data = df[['ticker', 'investment']].to_dict('records')
                        st.success(f"Loaded {len(holdings_data)} holdings from CSV")
                        st.info(f"Total Investment: ${total_investment:,.2f}")
                except Exception as e:
                    st.error(f"Error reading CSV: {str(e)}")

        submitted = st.form_submit_button("Create Portfolio", type="primary")
        if submitted:
            if not name or not name.strip():
                st.error("Portfolio name is required")
                return
            if not holdings_data:
                st.error("Please add at least one holding")
                return
            if total_investment <= 0:
                total_investment = sum([h['investment'] for h in holdings_data])
            if total_investment <= 0:
                st.error("Total investment must be greater than 0")
                return
            portfolio_data = {
                'name': name.strip(),
                'description': description.strip() if description else "",
                'tags': tags.strip() if tags else "",
                'initial_value': total_investment,
                'risk_tolerance': risk_tolerance,
                'holdings': holdings_data
            }
            try:
                if st.session_state.db_manager.create_portfolio(user_id, portfolio_data):
                    st.success(f"Portfolio '{name}' created successfully!")
                    st.success(f"Total investment: ${total_investment:,.2f}")
                    st.success(f"Number of holdings: {len(holdings_data)}")
                    if 'num_holdings' in st.session_state:
                        del st.session_state.num_holdings
                    st.rerun()
                else:
                    st.error("Failed to create portfolio. Please try again.")
            except Exception as e:
                st.error(f"Error creating portfolio: {str(e)}")


def display_portfolio_card(portfolio):
    with st.container():
        st.markdown(f"""
        <div class="portfolio-card">
            <h3>📊 {portfolio['name']}</h3>
            <p><strong>Description:</strong> {portfolio['description']}</p>
            <p><strong>Tags:</strong> {portfolio['tags']}</p>
            <p><strong>Total Investment:</strong> ${portfolio.get('initial_value', 0):,.2f}</p>
            <p><strong>Risk Tolerance:</strong> {portfolio.get('risk_tolerance', 'Not specified')}</p>
            <p><strong>Created:</strong> {portfolio['created_at']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show holdings summary
        holdings = st.session_state.db_manager.get_portfolio_holdings(portfolio['id'])
        if holdings:
            total_investment = sum([h['investment'] for h in holdings])
            st.write("**Holdings:**")
            holdings_summary = []
            for holding in holdings[:5]:  # Show first 5 holdings
                weight_pct = (holding['investment'] / total_investment * 100) if total_investment > 0 else 0
                holdings_summary.append(f"{holding['ticker']} ({weight_pct:.1f}% - ${holding['investment']:,.0f})")
            
            holdings_text = ", ".join(holdings_summary)
            if len(holdings) > 5:
                holdings_text += f" + {len(holdings) - 5} more"
            st.write(holdings_text)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            if st.button("📈 Analyze", key=f"analyze_{portfolio['id']}"):
                st.session_state.selected_portfolio_id = portfolio['id']
                st.session_state.current_page = 'Analysis'
                st.rerun()
        with col2:
            if st.button("✏️ Edit", key=f"edit_{portfolio['id']}"):
                st.info("Edit functionality coming soon!")
        with col3:
            if st.button("📊 Clone", key=f"clone_{portfolio['id']}"):
                st.info("Clone functionality coming soon!")
        with col4:
            if st.button("📄 Report", key=f"report_{portfolio['id']}"):
                try:
                    pdf = generate_portfolio_report(portfolio)
                    st.download_button("Download Report", data=pdf, file_name=f"{portfolio['name']}_report.pdf", mime='application/pdf')
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
        with col5:
            if st.button("🗑️ Delete", key=f"delete_{portfolio['id']}"):
                if st.session_state.db_manager.delete_portfolio(portfolio['id']):
                    st.success("Portfolio deleted!")
                    st.rerun()

def show_analysis_page():
    st.markdown('<h1 class="main-header">📈 Portfolio Analysis</h1>', unsafe_allow_html=True)
    user = st.session_state.auth_manager.get_current_user()
    portfolios = st.session_state.db_manager.get_user_portfolios(user['id'])
    if not portfolios:
        st.warning("No portfolios found. Please create a portfolio first.")
        return
    portfolio_names = {p['id']: f"{p['name']} - {p['description'][:30]}..." for p in portfolios}
    if 'selected_portfolio_id' in st.session_state:
        selected_id = st.session_state.selected_portfolio_id
        del st.session_state.selected_portfolio_id
    else:
        selected_id = st.selectbox("Select Portfolio", options=list(portfolio_names.keys()), 
                                   format_func=lambda x: portfolio_names[x])
    if selected_id:
        portfolio = next(p for p in portfolios if p['id'] == selected_id)
        analyze_portfolio(portfolio)

# Fixed analyze_portfolio function in app.py

def analyze_portfolio(portfolio):
    st.subheader(f"Analysis: {portfolio['name']}")
    holdings = st.session_state.db_manager.get_portfolio_holdings(portfolio['id'])
    
    # Add debugging information
    st.write(f"*Debug Info:*")
    st.write(f"- Portfolio ID: {portfolio['id']}")  
    st.write(f"- Holdings found: {len(holdings)}")
    
    if holdings:
        st.write("*Holdings details:*")
        for i, holding in enumerate(holdings):
            st.write(f"  {i+1}. {holding}")
    
    if not holdings:
        st.warning("No holdings found for this portfolio")
        # Check if portfolio exists
        portfolio_check = st.session_state.db_manager.get_portfolio_by_id(portfolio['id'])
        if portfolio_check:
            st.write("Portfolio exists in database but has no holdings")
        else:
            st.error("Portfolio not found in database!")
        return
    
    # Calculate weights from investments
    holdings = calculate_weights_from_investments(holdings)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())
    with col3:
        benchmark = st.selectbox("Benchmark", ["^GSPC", "^DJI", "^IXIC"], format_func=lambda x: {
            "^GSPC": "S&P 500", "^DJI": "Dow Jones", "^IXIC": "NASDAQ"
        }[x])
    
    # Display holdings info
    st.subheader("📋 Portfolio Holdings")
    holdings_display = []
    total_investment = sum([h['investment'] for h in holdings])
    
    for holding in holdings:
        holdings_display.append({
            'Ticker': holding['ticker'],
            'Investment': f"${holding['investment']:,.2f}",
            'Weight': f"{holding['weight']*100:.1f}%"
        })
    
    holdings_df = pd.DataFrame(holdings_display)
    st.dataframe(holdings_df, use_container_width=True)
    
    st.info(f"Total Portfolio Investment: ${total_investment:,.2f}")
    
    # Prepare data for analysis
    tickers = [h['ticker'] for h in holdings]
    investments = [h['investment'] for h in holdings]  # Use actual investment amounts
    
    # Debug output
    st.write("*Analysis Input Debug:*")
    st.write(f"- Tickers: {tickers}")
    st.write(f"- Investments: {investments}")
    st.write(f"- Total investment: {sum(investments)}")
    
    with st.spinner("Fetching data and analyzing..."):
        try:
            # Call analyzer with investments (it will normalize them internally)
            analysis = st.session_state.portfolio_analyzer.analyze_portfolio(
                tickers, investments, start_date, end_date, benchmark
            )
            
            if analysis and isinstance(analysis, dict):
                if 'error' in analysis:
                    st.error(f"Analysis error: {analysis['error']}")
                    st.write("*Full analysis result:*")
                    st.json(analysis)
                else:
                    display_portfolio_analysis(analysis, portfolio, holdings)
            else:
                st.error("Failed to analyze portfolio - invalid or empty analysis result")
                st.write("*Analysis result:*")
                st.write(analysis)
                
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# Make sure this function exists in your app.py (it should already be there)
def calculate_weights_from_investments(holdings):
    """Calculate weights from investment amounts"""
    total_investment = sum([h['investment'] for h in holdings])
    if total_investment == 0:
        return holdings
    
    for holding in holdings:
        holding['weight'] = holding['investment'] / total_investment
    
    return holdings
    
    # Calculate weights from investments
    holdings = calculate_weights_from_investments(holdings)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())
    with col3:
        benchmark = st.selectbox("Benchmark", ["^GSPC", "^DJI", "^IXIC"], format_func=lambda x: {
            "^GSPC": "S&P 500", "^DJI": "Dow Jones", "^IXIC": "NASDAQ"
        }[x])
    
    # Display holdings info
    st.subheader("📋 Portfolio Holdings")
    holdings_display = []
    total_investment = sum([h['investment'] for h in holdings])
    
    for holding in holdings:
        holdings_display.append({
            'Ticker': holding['ticker'],
            'Investment': f"${holding['investment']:,.2f}",
            'Weight': f"{holding['weight']*100:.1f}%"
        })
    
    holdings_df = pd.DataFrame(holdings_display)
    st.dataframe(holdings_df, use_container_width=True)
    
    st.info(f"Total Portfolio Investment: ${total_investment:,.2f}")
    
    tickers = [h['ticker'] for h in holdings]
    weights = [h['weight'] for h in holdings]
    
    with st.spinner("Fetching data and analyzing..."):
        try:
            analysis = st.session_state.portfolio_analyzer.analyze_portfolio(tickers, weights, start_date, end_date, benchmark)
            if analysis and isinstance(analysis, dict):
                display_portfolio_analysis(analysis, portfolio, holdings)
            else:
                st.error("Failed to analyze portfolio - invalid or empty analysis result")
                if analysis and 'error' in analysis:
                    st.error(f"Error details: {analysis['error']}")
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

def display_portfolio_analysis(analysis, portfolio, holdings):
    st.subheader("📊 Key Performance Metrics")
    
    metrics = analysis.get('metrics', {})
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        expected_return = metrics.get('expected_return', 0)
        st.metric("Expected Return", f"{expected_return:.2%}")
    with col2:
        volatility = metrics.get('volatility', 0)
        st.metric("Volatility", f"{volatility:.2%}")
    with col3:
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.3f}")
    with col4:
        max_drawdown = metrics.get('max_drawdown', 0)
        st.metric("Max Drawdown", f"{max_drawdown:.2%}")
    with col5:
        var_5 = metrics.get('var_5', 0)
        st.metric("VaR (5%)", f"{var_5:.2%}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sortino_ratio = metrics.get('sortino_ratio', 0)
        st.metric("Sortino Ratio", f"{sortino_ratio:.3f}")
    with col2:
        calmar_ratio = metrics.get('calmar_ratio', 0)
        st.metric("Calmar Ratio", f"{calmar_ratio:.3f}")
    with col3:
        beta = metrics.get('beta', 0)
        st.metric("Beta", f"{beta:.3f}")
    with col4:
        alpha = metrics.get('alpha', 0)
        st.metric("Alpha", f"{alpha:.3%}")
    
    # Performance chart
    st.subheader("📈 Performance Analysis")
    portfolio_returns = analysis.get('portfolio_returns')
    benchmark_returns = analysis.get('benchmark_returns')
    
    if portfolio_returns is not None:
     fig_performance = go.Figure()
    try:
        if hasattr(portfolio_returns, 'index'):
            fig_performance.add_trace(go.Scatter(
                x=portfolio_returns.index,
                y=(1 + portfolio_returns).cumprod() * 100,
                mode='lines',
                name='Portfolio',
                line=dict(color='#1f77b4', width=2)
            ))

            if benchmark_returns is not None and hasattr(benchmark_returns, 'index'):
                fig_performance.add_trace(go.Scatter(
                    x=benchmark_returns.index,
                    y=(1 + benchmark_returns).cumprod() * 100,
                    mode='lines',
                    name='Benchmark',
                    line=dict(color='red', width=2, dash='dash')
                ))

            fig_performance.update_layout(
                title='Cumulative Returns Comparison',
                xaxis_title='Date',
                yaxis_title='Cumulative Return (%)',
                hovermode='x unified',
                template='plotly_white'
            )
            st.plotly_chart(fig_performance, use_container_width=True)
        else:
            st.warning("⚠️ Portfolio returns data is not in expected format.")
    except Exception as e:
        st.error(f"🚫 Error creating performance chart: {str(e)}")

    
    # Risk-Return Scatter Plot
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Risk-Return Profile")
        if 'individual_metrics' in analysis:
            individual_metrics = analysis['individual_metrics']
            risk_return_data = []
            
            for ticker in individual_metrics:
                if ticker in individual_metrics:
                    metrics_data = individual_metrics[ticker]
                    risk_return_data.append({
                        'Ticker': ticker,
                        'Return': metrics_data.get('expected_return', 0) * 100,
                        'Risk': metrics_data.get('volatility', 0) * 100,
                        'Weight': next((h['weight'] for h in holdings if h['ticker'] == ticker), 0) * 100
                    })
            
            if risk_return_data:
                df_risk_return = pd.DataFrame(risk_return_data)
                
                fig_scatter = px.scatter(
                    df_risk_return,
                    x='Risk',
                    y='Return',
                    size='Weight',
                    hover_name='Ticker',
                    title='Risk vs Return Profile',
                    labels={'Risk': 'Volatility (%)', 'Return': 'Expected Return (%)'},
                    template='plotly_white'
                )
                
                # Add portfolio point
                portfolio_return = expected_return * 100
                portfolio_risk = volatility * 100
                
                fig_scatter.add_trace(go.Scatter(
                    x=[portfolio_risk],
                    y=[portfolio_return],
                    mode='markers',
                    marker=dict(size=20, color='red', symbol='diamond'),
                    name='Portfolio',
                    hovertemplate='Portfolio<br>Risk: %{x:.2f}%<br>Return: %{y:.2f}%'
                ))
                
                st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        st.subheader("🥧 Portfolio Allocation")
        # Create pie chart for portfolio allocation
        allocation_data = []
        for holding in holdings:
            allocation_data.append({
                'Ticker': holding['ticker'],
                'Weight': holding['weight'] * 100,
                'Investment': holding['investment']
            })
        
        df_allocation = pd.DataFrame(allocation_data)
        
        fig_pie = px.pie(
            df_allocation,
            values='Weight',
            names='Ticker',
            title='Portfolio Allocation',
            template='plotly_white'
        )
        
        fig_pie.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='%{label}<br>Weight: %{percent}<br>Investment: $%{customdata:,.0f}',
            customdata=df_allocation['Investment']
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Correlation Matrix
    st.subheader("🔗 Correlation Matrix")
    if 'correlation_matrix' in analysis:
        correlation_matrix = analysis['correlation_matrix']
        
        if correlation_matrix is not None and not correlation_matrix.empty:
            fig_corr = px.imshow(
                correlation_matrix,
                title='Asset Correlation Matrix',
                color_continuous_scale='RdBu',
                aspect='auto',
                template='plotly_white'
            )
            
            fig_corr.update_layout(
                xaxis_title='Assets',
                yaxis_title='Assets'
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("Correlation matrix data not available")
    
    # Risk Assessment
    st.subheader("⚠️ Risk Assessment")
    
    risk_level = "Low"
    risk_class = "risk-low"
    
    if volatility > 0.20:
        risk_level = "High"
        risk_class = "risk-high"
    elif volatility > 0.15:
        risk_level = "Medium"
        risk_class = "risk-medium"
    
    st.markdown(f"""
    <div class="risk-card {risk_class}">
        <h4>Risk Level: {risk_level}</h4>
        <p><strong>Portfolio Volatility:</strong> {volatility:.2%}</p>
        <p><strong>Maximum Drawdown:</strong> {max_drawdown:.2%}</p>
        <p><strong>Value at Risk (5%):</strong> {var_5:.2%}</p>
        <p><strong>Beta (vs Benchmark):</strong> {beta:.3f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance Summary
    st.subheader("📈 Performance Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Returns Analysis")
        st.write(f"• **Expected Annual Return:** {expected_return:.2%}")
        st.write(f"• **Annual Volatility:** {volatility:.2%}")
        st.write(f"• **Risk-Adjusted Return (Sharpe):** {sharpe_ratio:.3f}")
        st.write(f"• **Downside Risk-Adjusted (Sortino):** {sortino_ratio:.3f}")
    
    with col2:
        st.markdown("### 🎯 Benchmark Comparison")
        st.write(f"• **Alpha (Excess Return):** {alpha:.3%}")
        st.write(f"• **Beta (Market Sensitivity):** {beta:.3f}")
        st.write(f"• **Calmar Ratio:** {calmar_ratio:.3f}")
        
        # Interpretation
        if alpha > 0:
            st.success("✅ Portfolio is outperforming the benchmark")
        else:
            st.warning("⚠️ Portfolio is underperforming the benchmark")
    
    # Update portfolio metrics in database
    try:
        st.session_state.db_manager.update_portfolio_metrics(
            portfolio['id'],
            expected_return,
            volatility,
            sharpe_ratio
        )
    except Exception as e:
        st.warning(f"Could not update portfolio metrics: {str(e)}")

def show_ml_recommendations():
    st.markdown('<h1 class="main-header">🤖 ML Recommendations</h1>', unsafe_allow_html=True)
    
    user = st.session_state.auth_manager.get_current_user()
    portfolios = st.session_state.db_manager.get_user_portfolios(user['id'])
    
    if not portfolios:
        st.warning("No portfolios found. Please create a portfolio first.")
        return
    
    st.subheader("Select Portfolio for Recommendations")
    
    portfolio_names = {p['id']: f"{p['name']} - {p['description'][:30]}..." for p in portfolios}
    selected_id = st.selectbox("Choose Portfolio", options=list(portfolio_names.keys()), 
                               format_func=lambda x: portfolio_names[x])
    
    if selected_id:
        portfolio = next(p for p in portfolios if p['id'] == selected_id)
        holdings = st.session_state.db_manager.get_portfolio_holdings(portfolio['id'])
        
        if not holdings:
            st.warning("No holdings found for this portfolio")
            return
        
        st.subheader(f"🔍 Recommendations for: {portfolio['name']}")
        
        # Get current tickers
        current_tickers = [h['ticker'] for h in holdings]
        
        col1, col2 = st.columns(2)
        
        with col1:
            recommendation_type = st.selectbox(
                "Recommendation Type",
                ["Portfolio Optimization", "Similar Assets", "Sector Diversification", "Risk Assessment"]
            )
        
        with col2:
            risk_preference = st.selectbox(
                "Risk Preference",
                ["Conservative", "Moderate", "Aggressive"]
            )
        
        if st.button("Generate Recommendations", type="primary"):
            with st.spinner("Generating ML recommendations..."):
                try:
                    recommendations = st.session_state.ml_recommender.get_recommendations(
                        current_tickers, 
                        recommendation_type.lower().replace(' ', '_'),
                        risk_preference.lower()
                    )
                    
                    if recommendations:
                        display_recommendations(recommendations, recommendation_type)
                    else:
                        st.error("Could not generate recommendations. Please try again.")
                        
                except Exception as e:
                    st.error(f"Error generating recommendations: {str(e)}")

def display_recommendations(recommendations, rec_type):
    st.subheader(f"📋 {rec_type} Recommendations")
    
    if rec_type == "Portfolio Optimization":
        if 'optimized_weights' in recommendations:
            st.markdown("### 🎯 Optimized Portfolio Weights")
            
            weights_data = recommendations['optimized_weights']
            df_weights = pd.DataFrame(list(weights_data.items()), columns=['Ticker', 'Optimal Weight'])
            df_weights['Optimal Weight'] = df_weights['Optimal Weight'].apply(lambda x: f"{x:.1%}")
            
            st.dataframe(df_weights, use_container_width=True)
            
            # Show improvement metrics
            if 'improvement' in recommendations:
                improvement = recommendations['improvement']
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Expected Return Improvement", f"{improvement.get('return_improvement', 0):.2%}")
                with col2:
                    st.metric("Risk Reduction", f"{improvement.get('risk_reduction', 0):.2%}")
                with col3:
                    st.metric("Sharpe Ratio Improvement", f"{improvement.get('sharpe_improvement', 0):.3f}")
    
    elif rec_type == "Similar Assets":
        if 'similar_assets' in recommendations:
            st.markdown("### 🔄 Similar Assets You Might Consider")
            
            for asset_group in recommendations['similar_assets']:
                with st.expander(f"Similar to {asset_group['base_asset']}"):
                    for similar in asset_group['recommendations']:
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            st.write(f"**{similar['ticker']}** - {similar.get('name', 'N/A')}")
                        with col2:
                            st.write(f"Similarity: {similar['similarity_score']:.2%}")
                        with col3:
                            st.write(f"Risk Level: {similar.get('risk_level', 'N/A')}")
    
    elif rec_type == "Sector Diversification":
        if 'sector_analysis' in recommendations:
            st.markdown("### 🏭 Sector Diversification Analysis")
            
            sector_data = recommendations['sector_analysis']
            
            # Current sector allocation
            if 'current_allocation' in sector_data:
                current_df = pd.DataFrame(
                    list(sector_data['current_allocation'].items()),
                    columns=['Sector', 'Current Weight']
                )
                current_df['Current Weight'] = current_df['Current Weight'].apply(lambda x: f"{x:.1%}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Current Sector Allocation")
                    st.dataframe(current_df, use_container_width=True)
                
                with col2:
                    st.markdown("#### Recommended Additions")
                    if 'recommendations' in sector_data:
                        for rec in sector_data['recommendations']:
                            st.write(f"• **{rec['sector']}**: Add {rec['suggested_weight']:.1%}")
                            st.write(f"  Suggested: {rec['suggested_ticker']}")
    
    else:  # Risk Assessment
        if 'risk_analysis' in recommendations:
            st.markdown("### ⚠️ Risk Analysis & Recommendations")
            
            risk_data = recommendations['risk_analysis']
            
            # Risk metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                risk_level = risk_data.get('overall_risk', 'Medium')
                risk_color = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}.get(risk_level, 'gray')
                st.markdown(f"**Overall Risk Level:** <span style='color:{risk_color}'>{risk_level}</span>", 
                           unsafe_allow_html=True)
            
            with col2:
                concentration_risk = risk_data.get('concentration_risk', 0)
                st.metric("Concentration Risk", f"{concentration_risk:.2%}")
            
            with col3:
                diversification_score = risk_data.get('diversification_score', 0)
                st.metric("Diversification Score", f"{diversification_score:.1f}/10")
            
            # Risk recommendations
            if 'recommendations' in risk_data:
                st.markdown("#### 💡 Risk Mitigation Recommendations")
                for i, rec in enumerate(risk_data['recommendations'], 1):
                    st.write(f"{i}. {rec}")

def show_admin_panel():
    st.markdown('<h1 class="main-header">👑 Admin Panel</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["User Management", "System Analytics", "Data Management"])
    
    with tab1:
        st.subheader("👥 User Management")
        
        # Get all users
        users = st.session_state.db_manager.get_all_users()
        
        if users:
            st.markdown(f"**Total Users:** {len(users)}")
            
            # Users table
            users_data = []
            for user in users:
                portfolios = st.session_state.db_manager.get_user_portfolios(user['id'])
                users_data.append({
                    'ID': user['id'],
                    'Company': user['company_name'],
                    'Email': user['email'],
                    'Role': user['role'],
                    'Portfolios': len(portfolios),
                    'Created': user['created_at']
                })
            
            users_df = pd.DataFrame(users_data)
            st.dataframe(users_df, use_container_width=True)
            
            # User actions
            st.subheader("User Actions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                user_emails = [u['email'] for u in users]
                selected_user_email = st.selectbox("Select User", user_emails)
                
                if st.button("Reset Password"):
                    st.info("Password reset functionality would be implemented here")
                
                if st.button("Deactivate User"):
                    st.warning("User deactivation functionality would be implemented here")
            
            with col2:
                if st.button("Export User Data"):
                    csv = users_df.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        "users_data.csv",
                        "text/csv"
                    )
        else:
            st.info("No users found")
    
    with tab2:
        st.subheader("📊 System Analytics")
        
        # System metrics
        all_users = st.session_state.db_manager.get_all_users()
        all_portfolios = []
        
        for user in all_users:
            user_portfolios = st.session_state.db_manager.get_user_portfolios(user['id'])
            all_portfolios.extend(user_portfolios)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Users", len(all_users))
        
        with col2:
            st.metric("Total Portfolios", len(all_portfolios))
        
        with col3:
            total_value = sum([p.get('initial_value', 0) for p in all_portfolios])
            st.metric("Total AUM", f"${total_value:,.0f}")
        
        with col4:
            if all_portfolios:
                avg_portfolio_size = total_value / len(all_portfolios)
                st.metric("Avg Portfolio Size", f"${avg_portfolio_size:,.0f}")
        
        # Portfolio creation over time
        if all_portfolios:
            st.subheader("📈 Portfolio Creation Trend")
            
            # Convert creation dates and create trend data
            portfolio_dates = []
            for p in all_portfolios:
                try:
                    if isinstance(p['created_at'], str):
                        portfolio_dates.append(pd.to_datetime(p['created_at']).date())
                    else:
                        portfolio_dates.append(p['created_at'])
                except:
                    continue
            
            if portfolio_dates:
                date_counts = pd.Series(portfolio_dates).value_counts().sort_index()
                
                fig_trend = px.line(
                    x=date_counts.index,
                    y=date_counts.values,
                    title="Portfolio Creation Over Time",
                    labels={'x': 'Date', 'y': 'Number of Portfolios Created'}
                )
                
                st.plotly_chart(fig_trend, use_container_width=True)
    
    with tab3:
        st.subheader("🗄️ Data Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Backup & Export")
            
            if st.button("Export All Data"):
                st.info("Data export functionality would be implemented here")
            
            if st.button("Create Backup"):
                st.info("Backup functionality would be implemented here")
        
        with col2:
            st.markdown("#### Database Maintenance")
            
            if st.button("Clean Old Sessions"):
                st.info("Session cleanup functionality would be implemented here")
            
            if st.button("Optimize Database"):
                st.info("Database optimization functionality would be implemented here")
        
        st.markdown("#### Danger Zone")
        st.warning("⚠️ These actions are irreversible!")
        
        if st.button("🗑️ Clear All Test Data", type="secondary"):
            st.error("This would clear all test data - implement with caution!")

def show_settings_page():
    st.markdown('<h1 class="main-header">⚙️ Settings</h1>', unsafe_allow_html=True)
    
    user = st.session_state.auth_manager.get_current_user()
    
    tab1, tab2, tab3 = st.tabs(["Profile", "Preferences", "API Keys"])
    
    with tab1:
        st.subheader("👤 Profile Settings")
        
        with st.form("profile_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                company_name = st.text_input("Company Name", value=user['company_name'])
                email = st.text_input("Email", value=user['email'])
            
            with col2:
                # Add more profile fields as needed
                phone = st.text_input("Phone Number", value="")
                location = st.text_input("Location", value="")
            
            if st.form_submit_button("Update Profile"):
                # Update profile logic would go here
                st.success("Profile updated successfully!")
    
    with tab2:
        st.subheader("🎛️ Preferences")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Display Settings")
            
            currency = st.selectbox("Currency", ["USD", "EUR", "GBP", "JPY"], index=0)
            date_format = st.selectbox("Date Format", ["MM/DD/YYYY", "DD/MM/YYYY", "YYYY-MM-DD"], index=0)
            theme = st.selectbox("Theme", ["Light", "Dark", "Auto"], index=0)
        
        with col2:
            st.markdown("#### Analysis Settings")
            
            default_period = st.selectbox("Default Analysis Period", ["1Y", "2Y", "3Y", "5Y"], index=0)
            risk_free_rate = st.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
            confidence_level = st.selectbox("VaR Confidence Level", ["90%", "95%", "99%"], index=1)
        
        if st.button("Save Preferences"):
            st.success("Preferences saved successfully!")
    
    with tab3:
        st.subheader("🔑 API Keys & Integrations")
        
        st.markdown("#### External Data Providers")
        
        with st.expander("Yahoo Finance (Currently Used)"):
            st.info("✅ Yahoo Finance is currently being used for market data. No API key required.")
        
        with st.expander("Alpha Vantage Integration"):
            alpha_vantage_key = st.text_input("Alpha Vantage API Key", type="password")
            if st.button("Test Alpha Vantage Connection"):
                st.info("API key testing functionality would be implemented here")
        
        with st.expander("Quandl Integration"):
            quandl_key = st.text_input("Quandl API Key", type="password")
            if st.button("Test Quandl Connection"):
                st.info("API key testing functionality would be implemented here")
        
        st.markdown("#### Export & Reporting")
        
        with st.expander("Email Settings"):
            email_provider = st.selectbox("Email Provider", ["Gmail", "Outlook", "Custom SMTP"])
            smtp_server = st.text_input("SMTP Server")
            smtp_port = st.number_input("SMTP Port", value=587)
            
            if st.button("Test Email Configuration"):
                st.info("Email testing functionality would be implemented here")

if __name__ == "__main__":
    main()