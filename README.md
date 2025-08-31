# 📊 FinQuant Enterprise Portfolio Analyzer

An AI-powered, investment-based portfolio analysis tool built with Python and Streamlit. FinQuant Enterprise helps users create, analyze, and visualize financial portfolios with real stock data, performance metrics, and downloadable reports.

---

## 🚀 Features

- 📥 Manual and CSV-based portfolio creation
- 🧮 **Auto-calculated weights** based on investment amount
- 📈 Performance metrics: 
  - Expected Return
  - Volatility
  - Sharpe Ratio
  - Sortino Ratio
  - Max Drawdown
  - Calmar Ratio
  - Value at Risk (VaR)
- 🔍 Benchmark comparison with S&P 500, Dow Jones, or NASDAQ
- 📄 Generate and download analysis as PDF reports
- 📊 Interactive graphs (Plotly) for returns visualization

---

## 🛠️ Tech Stack

| Component   | Tech                  |
|------------|------------------------|
| Frontend   | Streamlit (Python)     |
| Backend    | Python, SQLite3        |
| Libraries  | yFinance, Pandas, NumPy, Plotly, ReportLab |
| Report     | PDF generation with ReportLab |
| Deployment | Local / Streamlit Cloud ready |

---

## 📂 Project Structure

```
finquant_enterprise_app/
├── app.py                      # Main Streamlit app
├── database.py                 # SQLite database manager
├── portfolio_analyzer.py       # Core logic for returns and metrics
├── report_generator.py         # PDF report generator
├── utils/                      # Utility functions (formatting, validation)
├── assets/                     # Static assets (icons, images)
├── data/                       # CSV template / uploads
└── README.md                   # Project documentation
```

---

## 🧪 How to Run

1. **Clone the repo**

```bash
git clone https://github.com/your-username/finquant-enterprise.git
cd finquant-enterprise
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the app**

```bash
streamlit run app.py
```

4. **Access**

Visit [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📥 Portfolio Input

- **Manual Form**: Add ticker, investment, and shares (weights are auto-calculated)
- **CSV Upload**: Upload file with:

```csv
ticker,investment
AAPL,20000
MSFT,15000
GOOGL,10000
```

---

## 🧠 Behind the Scenes

- Real-time price data fetched from Yahoo Finance via `yfinance`
- Returns and risk metrics calculated with Pandas and NumPy
- Interactive visuals via Plotly
- Modular and extensible structure (new metrics or benchmarks can be added)

---

## 🛡️ Security Note

- This app is for educational or personal finance research only.
- No financial advice is provided or stored.
- All data is stored locally (SQLite); no cloud sync by default.

---

## 📌 Future Enhancements

- User authentication and saved dashboards
- Portfolio rebalancing suggestions
- Cloud storage and sharing
- Forecasting and ML-based asset allocation

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change or improve.

---

## 📄 License

MIT License. See `LICENSE` file for details.

---

## 👤 Author

**Patel Priyanshi**  
[GitHub Profile](https://github.com/patelpriyanshi483)  
Email: priyanshi9112004@gmail.com

---

## 💰 Support

If you found this useful, give a ⭐ on GitHub or consider contributing!
