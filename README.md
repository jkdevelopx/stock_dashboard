# 📊 StockSense Pro

> **Intelligent Market Analytics Platform for Real-Time Stock Screening**

Built by **Wichaya Kanlaya** | [LinkedIn](#) | [Portfolio](#)

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[🚀 Live Demo](#) | [📹 Video Demo](#) | [📄 Documentation](#)

---

## 🎯 Project Overview

**StockSense Pro** is a sophisticated stock screening and technical analysis platform that empowers investors with data-driven insights. The system processes real-time market data, applies advanced technical indicators, and delivers actionable intelligence through an intuitive web interface.

### **Key Achievement**
Built a full-stack financial analytics application processing **50+ stocks** with **10+ technical indicators** in under **30 seconds**, deployed on cloud infrastructure.

---

## ✨ Features

### 🔍 **Smart Stock Screening**
- Real-time data integration via Yahoo Finance API
- Custom composite scoring algorithm (0-100 scale)
- Multi-factor analysis: Momentum, RSI, Volume, Moving Averages

### 📊 **Advanced Visualization**
- Interactive candlestick charts with Plotly
- Multi-indicator overlays (MA20, MA50, MA200, RSI, MACD)
- Volume profile analysis with color-coded bars

### 🎯 **Intelligent Filtering**
- Signal strength classification (Strong/Moderate/Weak)
- Sector-based filtering (Technology, Healthcare, Finance, etc.)
- Performance metrics (1D, 1W, 1M returns)

### 📈 **Portfolio Management**
- Personal watchlist with persistent storage
- Bulk analysis capabilities
- CSV export functionality

---

## 🛠️ Technical Architecture

### **Tech Stack**

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Streamlit 1.25+ | Interactive web UI |
| **Data Processing** | Pandas, NumPy | Data manipulation & computation |
| **Visualization** | Plotly 5.20+ | Interactive charts |
| **Data Source** | yfinance API | Real-time market data |
| **Deployment** | Docker, Cloud | Containerized deployment |

### **System Architecture**

```
┌─────────────────────────────────────────┐
│         Presentation Layer              │
│         (Streamlit Web UI)              │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│       Application Layer                 │
│  • Stock Scanner Engine                 │
│  • Technical Analysis Processor         │
│  • Scoring Algorithm                    │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│          Data Layer                     │
│  • Yahoo Finance API                    │
│  • Historical Price Data                │
│  • Company Fundamentals                 │
└─────────────────────────────────────────┘
```

### **Data Flow**

```
User Input → API Request → Data Cleaning → 
Indicator Calculation → Scoring → Visualization → 
Interactive Analysis
```

---

## 🧮 Scoring Algorithm

The proprietary scoring algorithm combines multiple technical factors:

```python
Score = Weighted Sum of:
  • 32% - Momentum (21-day price change)
  • 20% - Volume Spike (current vs average)
  • 18% - RSI Position (relative strength)
  • 12% - MA Alignment (trend direction)
  • 10% - Higher Lows Pattern (accumulation)
  • 8%  - Revenue Growth (fundamental)

Result: 0-100 scale
  🟢 Strong:    Score ≥ 65
  🟡 Moderate:  45 ≤ Score < 65
  🔴 Weak:      Score < 45
```

---

## 🚀 Quick Start

### **Prerequisites**
- Python 3.11+
- pip package manager

### **Installation**

```bash
# Clone repository
git clone https://github.com/yourusername/stocksense-pro.git
cd stocksense-pro

# Install dependencies
pip install -r requirements.txt

# Run application
python -m streamlit run streamlit_mcrf_dashboard.py
```

### **Docker Deployment**

```bash
# Build image
docker build -t stocksense-pro .

# Run container
docker run -p 8501:8501 stocksense-pro
```

Access at: `http://localhost:8501`

---

## 📊 Key Metrics & Performance

| Metric | Value |
|--------|-------|
| **Data Processing Speed** | 50 stocks in ~30 seconds |
| **Technical Indicators** | 10+ calculated per stock |
| **Supported Universes** | 8 pre-configured lists |
| **Chart Types** | 3 (Candlestick, Volume, RSI) |
| **API Response Time** | < 300ms (cached) |

---

## 💡 Technical Highlights

### **1. Efficient Data Pipeline**
- Implemented caching strategy reducing API calls by 80%
- Parallel processing ready architecture
- Error handling for edge cases (delisted stocks, missing data)

### **2. Advanced Algorithms**
```python
# RSI Calculation (14-period)
delta = df['Close'].diff()
up = delta.clip(lower=0)
down = -1 * delta.clip(upper=0)
roll_up = up.rolling(14).mean()
roll_down = down.rolling(14).mean()
rs = roll_up / (roll_down + 1e-9)
df['RSI14'] = 100 - (100 / (1 + rs))
```

### **3. Interactive Visualization**
- Multi-subplot charts with synchronized zooming
- Color-coded volume bars (red/green)
- Dynamic legend filtering
- Responsive design for mobile/desktop

### **4. Real-Time Updates**
- 5-minute cache TTL for live data
- Automatic refresh mechanism
- Last-updated timestamp display

---

## 📸 Screenshots

### Dashboard Overview
![Dashboard](screenshots/dashboard.png)
*Main scanner interface with universe selection and filtering options*

### Results Table
![Results](screenshots/results.png)
*Color-coded signal strength with performance metrics*

### Detailed Analysis
![Chart](screenshots/chart.png)
*Interactive candlestick chart with technical indicators*

### Technical Architecture
![Architecture](screenshots/architecture.png)
*System design and data flow diagram*

---

## 🎓 Skills Demonstrated

### **Technical Skills**
- ✅ Python Programming (Advanced)
- ✅ Data Engineering & ETL Pipelines
- ✅ API Integration (REST APIs)
- ✅ Data Visualization (Plotly, Streamlit)
- ✅ Financial Data Analysis
- ✅ Docker & Containerization
- ✅ Cloud Deployment
- ✅ Version Control (Git)

### **Soft Skills**
- ✅ Problem Solving
- ✅ System Design
- ✅ User Experience (UX) Design
- ✅ Technical Documentation
- ✅ Performance Optimization

---

## 📁 Project Structure

```
stocksense-pro/
├── streamlit_mcrf_dashboard.py    # Main application
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container configuration
├── .streamlit/
│   └── config.toml                # Streamlit settings
├── screenshots/                    # Portfolio images
├── README.md                       # This file
└── docs/
    ├── architecture.md            # Technical design docs
    └── deployment.md              # Deployment guide
```

---

## 🔮 Future Enhancements

- [ ] **Backtesting Module** - Historical strategy performance
- [ ] **Risk Management Tools** - Stop-loss & position sizing
- [ ] **News Sentiment Analysis** - AI-powered news integration
- [ ] **Multi-Timeframe Analysis** - Hourly/Daily/Weekly signals
- [ ] **Portfolio Tracking** - Real-time P&L monitoring
- [ ] **Mobile App** - React Native companion app

---

## 🤝 Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

**Important:** StockSense Pro is an **educational tool** designed for learning purposes only. This application does NOT provide financial advice, investment recommendations, or trading signals.

- ✅ Use for learning technical analysis
- ✅ Use for market research
- ❌ Do NOT use as sole basis for trading
- ❌ Always consult licensed financial advisors

**Risk Warning:** Trading stocks involves substantial risk. Past performance does not guarantee future results.

---

## 👤 About the Developer

**Wichaya Kanlaya**  
Data Engineer | Financial Technology Enthusiast

I'm passionate about building data-driven solutions that make complex financial information accessible. StockSense Pro showcases my skills in full-stack development, data engineering, and financial analytics.

**Connect with me:**
- 💼 [LinkedIn](https://www.linkedin.com/in/wichaya-k/)
- 🐙 [GitHub](https://github.com/jkdevelopx/)
- 📧 [Email](mailto:wichaya.kly@gmail.com)

---

## 🙏 Acknowledgments

- Yahoo Finance API for market data
- Streamlit team for the amazing framework
- Plotly for interactive visualization tools
- Open-source community for inspiration

---

<div align="center">

**⭐ Star this repo if you find it useful!**

Made with ❤️ by Wichaya Kanlaya

</div>