# 🏦 BankRisk Pro - Credit Risk Analytics Demonstration Platform

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

BankRisk Pro is an AI-assisted educational and portfolio project demonstrating an end-to-end credit-risk analytics workflow using Python, machine learning, synthetic borrower data, economic-data integration, Monte Carlo risk measurement, stress-testing concepts, and interactive visualization.

The project is designed to demonstrate applied data-science, risk-analytics, and software-development concepts. It is not a production banking system, validated risk model, regulatory reporting platform, or substitute for an institution's governed model-risk and regulatory processes.

## 🎯 **Key Features**

### **🤖 Credit Risk Analytics**
- **ML Credit Scoring**: Random Forest, Gradient Boosting, Logistic Regression
- **Portfolio Analytics**: VaR calculation, concentration risk, diversification metrics
- **Stress Testing**: CCAR/DFAST-style stress scenarios and custom economic shocks for educational scenario simulations inspired by regulatory stress-testing concepts; not regulatory-compliant CCAR/DFAST implementations
- **Real-time Data**: FRED API integration with intelligent caching

### **🏛️ Regulatory Risk Concepts**
- **Basel III-related Concepts**: Capital adequacy and risk-weighted assets
- **Stress-Scenario Analysis**: Scenario analysis and loss projections
- **Risk Reporting**: Executive dashboards and risk reporting
- **Capital Planning Concepts**: Economic capital and regulatory capital concepts

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
- **Fallback Data**: Realistic synthetic data when external APIs are unavailable

External economic data is used when configured and successfully retrieved. When it is unavailable, the project may use realistic synthetic or fallback data. The borrower portfolio and default outcomes are synthetic.

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

**Without API keys**: Project uses realistic fallback data  
**With API keys**: Uses real economic data when successfully retrieved

## 📈 **Sample Output**
<img width="587" height="944" alt="BankRisk1_Menu" src="https://github.com/user-attachments/assets/069f6cef-231b-4827-be15-5107a9f86536" />
<img width="453" height="960" alt="BankRisk4_PortfolioStats" src="https://github.com/user-attachments/assets/5f7a97f2-f945-4c9a-803f-05275ec35fdd" />
<img width="1699" height="1200" alt="newplot" src="https://github.com/user-attachments/assets/607c3f3e-d1c2-4931-9c82-ffde6ceea75e" />
<img width="671" height="803" alt="image" src="https://github.com/user-attachments/assets/77afabd6-3453-4fd1-a531-3ffb6d0ed592" />
<img width="678" height="796" alt="image" src="https://github.com/user-attachments/assets/dc9a2080-1864-44c4-a354-3ebca698a4b0" />




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

## 📐 **Modeling Scope & Limitations**

- Borrower and loan records are synthetically generated.
- Default outcomes used for ML training are synthetically generated from borrower and economic risk characteristics.
- Reported AUC and model-performance results measure performance against the simulated target and should not be interpreted as evidence of real-world default-prediction performance.
- Monte Carlo VaR uses simplified portfolio assumptions, including borrower-level PDs and fixed LGD assumptions, and demonstrates methodology rather than replicating an enterprise credit portfolio model.
- Stress-testing scenarios demonstrate scenario-analysis concepts and are not validated regulatory stress tests.

## 📋 **Use Cases**

- **Credit-Risk Analytics Learning**: Explore end-to-end credit-risk workflows
- **Portfolio Analytics Demonstrations**: Examine exposure, concentration, expected loss, and VaR concepts
- **ML Model-Development Practice**: Experiment with classification models, features, and evaluation
- **Stress-Testing Methodology Demonstrations**: Explore scenario analysis and loss projections
- **Executive Visualization and Risk Storytelling**: Practice communicating analytics through dashboards and reports
- **Software-Engineering and API-Integration Practice**: Work with modular Python components, external data sources, caching, and fallback strategies

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

The project includes several engineering patterns that demonstrate production-oriented development concepts:

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
- **Financial Risk Modeling**: Credit risk, portfolio theory, regulatory risk concepts
- **Machine Learning**: Model development, validation, feature engineering
- **API Integration**: Rate limiting, caching, fallback strategies
- **Data Visualization**: Interactive dashboards, executive reporting
- **Software Engineering**: Modular design, testing, documentation

## 👨‍💼 **Skills Demonstrated**

**Risk Analytics:**
- Portfolio risk measurement
- Expected-loss analysis
- Concentration analysis
- Stress/scenario analysis
- Risk reporting and visualization

**Data Science:**
- Classification modeling
- Model comparison and validation
- Feature engineering
- ROC-AUC evaluation
- Synthetic-data experimentation

**Software Engineering:**
- Modular Python application design
- API integration
- Caching and fallback strategies
- Error handling
- Interactive visualization

## Methodology & Limitations

- **Synthetic data and labels.** Borrowers *and* default outcomes are simulated. Defaults are generated from a formula on the same borrower features the models train on, so reported AUC measures how well the models recover that formula, not real-world predictive power. Replace `_generate_default_target` with observed outcomes before drawing any conclusion from model metrics.
- **Portfolio loss distribution.** VaR is a Monte Carlo quantile of portfolio credit loss under a one-factor Gaussian copula (`asset_correlation`, default 0.10, an assumption; Basel IRB retail correlations are roughly 3-16%). Setting it to 0 gives independent defaults, which understates tail losses.
- **PD scoring.** Portfolio PDs for the training portfolio are 5-fold out-of-fold predictions, so no borrower is scored by a model that saw it.
- **Concentration.** `concentration_hhi` is the larger of the industry and state *normalized* HHI (0 = even, 1 = single bucket).
- **Stress tests.** Scenarios apply bounded PD multipliers (capped at 5x) to borrower risk drivers. They are scenario-based sensitivity analyses, not a CCAR/DFAST implementation. Capital impact is loss increase x 1.5, a placeholder assumption.
- **Economic data.** Without a `FRED_API_KEY`, indicators are simulated placeholders (flagged by `EconomicIndicators.is_simulated`), not live data.
- **Capital figures.** The executive dashboard shows economic capital required (99% VaR minus expected loss, as % of exposure). The regulatory-compliance table uses illustrative placeholder ratios and is marked "Not computed".

---

**Independent AI-assisted project exploring the intersection of credit risk, data science, and software development.**

*Built as a learning and portfolio demonstration; not intended for production risk decisions or regulatory reporting.*
