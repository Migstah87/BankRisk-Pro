# 🏦 BankRisk Pro - Advanced Credit Risk Management Platform

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive credit risk management platform designed for banking institutions, featuring machine learning models, regulatory stress testing, and interactive risk dashboards with intelligent API rate limiting.

## 🎯 **Key Features**

### **🤖 Advanced Risk Analytics**
- **ML Credit Scoring**: Random Forest, Gradient Boosting, Logistic Regression
- **Portfolio Analytics**: VaR calculation, concentration risk, diversification metrics
- **Stress Testing**: CCAR/DFAST scenarios, custom economic shocks
- **Real-time Data**: FRED API integration with intelligent caching

### **🏛️ Regulatory Compliance**
- **Basel III**: Capital adequacy, risk-weighted assets
- **Stress Testing**: Regulatory scenarios, loss projections
- **Reporting**: Executive dashboards, compliance reports
- **Capital Planning**: Economic capital, regulatory capital

### **📊 Professional Dashboards**
- Interactive risk visualizations
- Executive summary reports
- Model performance monitoring
- Geographic risk analysis

### **⚡ Smart API Management**
- **Rate Limiting**: Intelligent delays and caching
- **Fallback Data**: Continues working during API limits
- **Batch Processing**: Optimized API usage
- **Graceful Degradation**: Robust error handling

## 🚀 **Quick Start**

### **Option 1: Automated Setup (Recommended)**
```bash
# 1. Clone repository
git clone https://github.com/Migstah87/BankRisk-Pro.git
cd BankRisk-Pro

# 2. Run setup script
python setup_bankrisk.py

# 3. Run quick demo
python quick_demo.py
```

### **Option 2: Manual Setup**
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up API keys (optional)
cp .env.example .env
# Edit .env with your FRED API key

# 3. Run platform
python bankrisk_pro.py
# Select option 9 for demo
```

## 📊 **API Usage & Rate Limiting**

The platform includes intelligent API management to handle rate limits:

### **Economic Data Sources**
- **FRED API**: Federal Reserve economic data (free key recommended)
- **Yahoo Finance**: Market data with smart rate limiting
- **Fallback Data**: Realistic synthetic data when APIs unavailable

### **Rate Limiting Features**
```python
✅ Intelligent caching (15-30 minute intervals)
✅ Automatic delays between API calls
✅ Graceful fallback to synthetic data
✅ Batch processing to minimize API calls
✅ Session-based data reuse
```

### **API Keys (Optional but Recommended)**
Get your free FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html

**Without API keys**: Platform uses realistic fallback data  
**With API keys**: Enhanced accuracy with real economic data

## 📈 **Sample Output**

### **Portfolio Analysis**
```
📊 PORTFOLIO ANALYSIS RESULTS:
   Total Exposure: $245,680,000
   Number of Loans: 10,000
   Average PD: 8.7%
   Expected Loss: $4,850,000
   Portfolio VaR (95%): $12,300,000
   Portfolio VaR (99%): $18,450,000
```

### **Stress Testing Results**
```
⚡ STRESS TEST RESULTS (Recession):
   Baseline Expected Loss: $4,850,000
   Stressed Expected Loss: $7,230,000
   Loss Increase: 49.1%
   Capital Impact: $3,570,000
```

### **Risk Distribution**
```
🎯 RISK GRADE DISTRIBUTION:
   Grade AAA: 2.1%    Grade BB: 18.7%
   Grade AA: 4.3%     Grade B: 12.4%
   Grade A: 8.9%      Grade CCC: 5.2%
   Grade BBB: 23.8%   Grade CC: 1.8%
```

## 🏗️ **Architecture**

```
Economic APIs → Data Caching → Portfolio Generation → ML Models → Risk Analytics → Dashboards
     ↓              ↓              ↓                  ↓            ↓             ↓
  FRED API      Smart Cache    Synthetic Data    Credit Scoring   VaR/EL     Interactive
  Yahoo Finance  Rate Limits   10K+ Borrowers    Risk Grading    Stress Test   Plotly
```

## 📋 **Use Cases**

- **Credit Risk Officers**: Portfolio risk assessment and monitoring
- **Regulators**: Stress testing and capital adequacy analysis  
- **Risk Managers**: Concentration risk and diversification analysis
- **Executives**: High-level risk reporting and strategic planning
- **Analysts**: Model development and validation

## 🛠️ **Technical Stack**

- **Python 3.8+**: Core development
- **Scikit-learn**: Machine learning models
- **Plotly**: Interactive dashboards
- **Pandas/NumPy**: Data processing
- **FRED/Yahoo Finance APIs**: Economic data with rate limiting
- **Monte Carlo**: VaR simulations

## 📁 **Project Structure**

```
BankRisk-Pro/
├── bankrisk_pro.py           # Main platform
├── data_collector.py         # Economic data with caching
├── portfolio_generator.py    # Optimized portfolio generation
├── credit_models.py          # ML risk models
├── risk_analytics.py         # Portfolio analysis
├── stress_testing.py         # Regulatory scenarios
├── visualizations.py         # Interactive dashboards
├── utils/helpers.py          # Utilities & reporting
├── quick_demo.py             # Quick demonstration
├── setup_bankrisk.py         # Automated setup
├── requirements.txt          # Dependencies
├── .env.example              # Configuration template
└── outputs/                  # Generated reports
```

## 🔧 **Configuration**

### **Environment Variables (.env)**
```bash
# API Configuration
FRED_API_KEY=your_key_here                    # Optional
ALPHA_VANTAGE_API_KEY=your_key_here          # Optional

# Rate Limiting
YAHOO_FINANCE_DELAY=2                        # Seconds between calls
FRED_API_DELAY=1                             # Seconds between calls

# Platform Settings
BANKRISK_CACHE_ENABLED=true                  # Enable caching
BANKRISK_MAX_PORTFOLIO_SIZE=50000            # Max borrowers
```

## 🚨 **Troubleshooting**

### **Rate Limiting Issues**
```bash
# Symptoms: "YFRateLimitError" or slow API responses
✅ Solution 1: Wait 5-10 minutes, then retry
✅ Solution 2: Use smaller portfolio sizes (1000-2000)
✅ Solution 3: Platform automatically uses fallback data
✅ Solution 4: Get free FRED API key for better limits
```

### **Common Issues**
```bash
# Missing packages
pip install -r requirements.txt

# Permission errors
python -m pip install --user package_name

# Import errors
python setup_bankrisk.py
```

## 📊 **Performance Optimization**

The platform includes several optimizations for production use:

- **Data Caching**: Economic data cached for 15-30 minutes
- **Batch Processing**: Single API calls for multiple data points
- **Smart Retries**: Exponential backoff for failed requests
- **Memory Management**: Efficient data structures for large portfolios
- **Parallel Processing**: Multi-threaded model training

## 🤝 **Contributing**

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 **License**

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## 🎓 **Learning & Development**

This platform demonstrates:
- **Advanced Python Programming**: Object-oriented design, error handling
- **Financial Risk Modeling**: Credit risk, portfolio theory, regulatory compliance
- **Machine Learning**: Model development, validation, feature engineering
- **API Integration**: Rate limiting, caching, fallback strategies
- **Data Visualization**: Interactive dashboards, executive reporting
- **Software Engineering**: Modular design, testing, documentation

## 👨‍💼 **Professional Applications**

**For Banking Professionals:**
- Portfolio risk monitoring and reporting
- Regulatory stress testing and compliance
- Credit policy development and validation
- Capital planning and optimization

**For Risk Managers:**
- Concentration risk analysis
- Early warning system development
- Model governance and validation
- Scenario analysis and planning

**For Executives:**
- Strategic risk assessment
- Board-level reporting
- Regulatory compliance oversight
- Business planning and optimization

---

**Built for banking professionals by a risk management specialist**

*Showcasing expertise in quantitative finance, machine learning, and regulatory compliance*
