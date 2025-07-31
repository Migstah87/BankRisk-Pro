"""
Utilities and Helper Functions
=============================

Supporting functions for BankRisk Pro platform including:
- Environment setup
- Data export utilities
- Report generation
- Configuration management
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Any
import subprocess
import warnings
warnings.filterwarnings('ignore')

def setup_environment():
    """
    Setup environment for BankRisk Pro platform
    Checks dependencies, creates directories, validates API keys
    """
    
    print("🔧 Setting up BankRisk Pro environment...")
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required")
        sys.exit(1)
    
    # Check and install required packages
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'requests', 
        'yfinance', 'python-dotenv', 'matplotlib', 'seaborn', 'plotly'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"📦 Installing missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")
    
    # Create .env file if it doesn't exist
    if not os.path.exists('.env'):
        create_sample_env_file()
    
    # Create output directories
    directories = ['outputs', 'models', 'reports', 'dashboards']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Environment setup completed")

def create_sample_env_file():
    """Create sample .env file with API key placeholders"""
    
    env_content = """# BankRisk Pro Configuration
# API Keys for Economic Data

# Federal Reserve Economic Data (FRED) API Key
# Get your free key at: https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY=your_fred_api_key_here

# Alpha Vantage API Key (optional, for additional market data)
# Get your free key at: https://www.alphavantage.co/support/#api-key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here

# Example with real keys:
# FRED_API_KEY=abcd1234567890efgh
# ALPHA_VANTAGE_API_KEY=DEMO123456789
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print("📝 Created .env file template")
        print("   Please add your API keys to the .env file")
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")

def save_results_to_csv(borrowers: List, portfolio_metrics, filename: Optional[str] = None) -> str:
    """
    Save analysis results to CSV file
    
    Args:
        borrowers: List of Borrower objects
        portfolio_metrics: PortfolioMetrics object
        filename: Optional custom filename
        
    Returns:
        Filename of saved CSV
    """
    
    if not filename:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"outputs/bankrisk_analysis_{timestamp}.csv"
    
    try:
        # Convert borrowers to DataFrame
        borrower_data = []
        for borrower in borrowers:
            data_dict = {
                'borrower_id': borrower.borrower_id,
                'age': borrower.age,
                'income': borrower.income,
                'credit_score': borrower.credit_score,
                'debt_to_income_ratio': borrower.debt_to_income,
                'employment_length': borrower.employment_length,
                'loan_amount': borrower.loan_amount,
                'loan_purpose': borrower.loan_purpose,
                'loan_term': borrower.loan_term,
                'interest_rate': borrower.interest_rate,
                'home_ownership': borrower.home_ownership,
                'state': borrower.state,
                'industry': borrower.industry,
                'education_level': borrower.education_level,
                'loan_grade': borrower.loan_grade,
                'local_unemployment': borrower.local_unemployment,
                'regional_gdp_growth': borrower.regional_gdp_growth,
                'industry_risk_factor': borrower.industry_risk_factor,
                'probability_default_pct': borrower.probability_default,
                'risk_grade': borrower.risk_grade,
                'expected_loss_dollars': borrower.expected_loss,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            borrower_data.append(data_dict)
        
        df = pd.DataFrame(borrower_data)
        
        # Add portfolio summary at the top
        summary_rows = []
        if portfolio_metrics:
            summary_rows.extend([
                ['PORTFOLIO_SUMMARY', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
                ['Total_Exposure', portfolio_metrics.total_exposure, '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
                ['Number_of_Loans', portfolio_metrics.number_of_loans, '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
                ['Average_PD_Percent', portfolio_metrics.average_pd, '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
                ['Expected_Loss', portfolio_metrics.expected_loss, '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
                ['Portfolio_VaR_95', portfolio_metrics.portfolio_var_95, '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
                ['Portfolio_VaR_99', portfolio_metrics.portfolio_var_99, '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
                ['Concentration_HHI', portfolio_metrics.concentration_hhi, '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
                ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
                ['INDIVIDUAL_LOANS', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']
            ])
        
        # Create summary DataFrame and concatenate
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows, columns=df.columns)
            final_df = pd.concat([summary_df, df], ignore_index=True)
        else:
            final_df = df
        
        # Save to CSV
        final_df.to_csv(filename, index=False)
        
        print(f"💾 Results saved to: {filename}")
        print(f"📊 Exported {len(borrower_data):,} loan records")
        
        return filename
        
    except Exception as e:
        print(f"❌ Error saving CSV: {e}")
        return ""

def generate_executive_report(borrowers: List, portfolio_metrics, economic_snapshot, 
                            filename: Optional[str] = None) -> str:
    """
    Generate executive PDF report
    
    Args:
        borrowers: List of Borrower objects
        portfolio_metrics: PortfolioMetrics object
        economic_snapshot: EconomicIndicators object
        filename: Optional custom filename
        
    Returns:
        Filename of generated report
    """
    
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"reports/bankrisk_executive_report_{timestamp}.pdf"
        
        # Create PDF document
        doc = SimpleDocTemplate(filename, pagesize=letter,
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.darkblue
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.darkblue
        )
        
        # Build document content
        story = []
        
        # Title
        title = Paragraph("BankRisk Pro<br/>Credit Risk Analysis Report", title_style)
        story.append(title)
        story.append(Spacer(1, 12))
        
        # Report date and summary
        report_date = datetime.now().strftime('%B %d, %Y at %I:%M %p')
        intro_text = f"""
        <b>Report Generated:</b> {report_date}<br/>
        <b>Analysis Period:</b> Current Portfolio Snapshot<br/>
        <b>Methodology:</b> Machine Learning Credit Risk Models<br/><br/>
        
        This report provides a comprehensive analysis of the loan portfolio using advanced 
        credit risk modeling techniques, incorporating current economic conditions and 
        regulatory stress testing scenarios.
        """
        
        intro = Paragraph(intro_text, styles['Normal'])
        story.append(intro)
        story.append(Spacer(1, 20))
        
        # Executive Summary
        exec_heading = Paragraph("Executive Summary", heading_style)
        story.append(exec_heading)
        
        if portfolio_metrics:
            exec_summary = f"""
            <b>Portfolio Overview:</b><br/>
            • Total Portfolio Exposure: ${portfolio_metrics.total_exposure:,.0f}<br/>
            • Number of Loans: {portfolio_metrics.number_of_loans:,}<br/>
            • Average Probability of Default: {portfolio_metrics.average_pd:.2f}%<br/>
            • Expected Loss: ${portfolio_metrics.expected_loss:,.0f}<br/>
            • Portfolio VaR (95%): ${portfolio_metrics.portfolio_var_95:,.0f}<br/>
            • Portfolio VaR (99%): ${portfolio_metrics.portfolio_var_99:,.0f}<br/>
            • Concentration Risk (HHI): {portfolio_metrics.concentration_hhi:.3f}<br/><br/>
            
            <b>Risk Assessment:</b><br/>
            The portfolio demonstrates {"strong" if portfolio_metrics.average_pd < 5 else "moderate" if portfolio_metrics.average_pd < 10 else "elevated"} 
            credit quality with an average probability of default of {portfolio_metrics.average_pd:.2f}%. 
            The concentration risk is {"low" if portfolio_metrics.concentration_hhi < 0.1 else "moderate" if portfolio_metrics.concentration_hhi < 0.25 else "high"} 
            based on the HHI index of {portfolio_metrics.concentration_hhi:.3f}.
            """
        else:
            exec_summary = "Portfolio metrics not available for this analysis."
        
        summary_para = Paragraph(exec_summary, styles['Normal'])
        story.append(summary_para)
        story.append(Spacer(1, 20))
        
        # Economic Environment
        econ_heading = Paragraph("Economic Environment", heading_style)
        story.append(econ_heading)
        
        if economic_snapshot:
            econ_text = f"""
            <b>Current Economic Indicators:</b><br/>
            • Federal Funds Rate: {economic_snapshot.federal_funds_rate:.2f}%<br/>
            • Unemployment Rate: {economic_snapshot.unemployment_rate:.1f}%<br/>
            • GDP Growth: {economic_snapshot.gdp_growth:.1f}%<br/>
            • Inflation Rate: {economic_snapshot.inflation_rate:.1f}%<br/>
            • Credit Spread: {economic_snapshot.credit_spread:.2f}%<br/>
            • Market Volatility (VIX): {economic_snapshot.vix_volatility:.1f}<br/><br/>
            
            The current economic environment is characterized by 
            {"accommodative" if economic_snapshot.federal_funds_rate < 3 else "neutral" if economic_snapshot.federal_funds_rate < 5 else "restrictive"} 
            monetary policy and {"low" if economic_snapshot.unemployment_rate < 4 else "moderate" if economic_snapshot.unemployment_rate < 6 else "elevated"} 
            unemployment levels.
            """
        else:
            econ_text = "Economic data not available for this analysis."
        
        econ_para = Paragraph(econ_text, styles['Normal'])
        story.append(econ_para)
        story.append(Spacer(1, 20))
        
        # Risk Grade Distribution Table
        if portfolio_metrics and portfolio_metrics.grade_distribution:
            grade_heading = Paragraph("Risk Grade Distribution", heading_style)
            story.append(grade_heading)
            
            # Create table data
            table_data = [['Risk Grade', 'Number of Loans', 'Percentage', 'Risk Level']]
            
            risk_levels = {
                'AAA': 'Minimal', 'AA': 'Very Low', 'A': 'Low', 'BBB': 'Moderate',
                'BB': 'Moderate High', 'B': 'High', 'CCC': 'Very High', 
                'CC': 'Extremely High', 'C': 'Default Risk'
            }
            
            total_loans = portfolio_metrics.number_of_loans
            for grade, percentage in sorted(portfolio_metrics.grade_distribution.items()):
                num_loans = int(total_loans * percentage / 100)
                risk_level = risk_levels.get(grade, 'Unknown')
                table_data.append([grade, f"{num_loans:,}", f"{percentage:.1f}%", risk_level])
            
            # Create and style table
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(table)
            story.append(Spacer(1, 20))
        
        # Portfolio Statistics
        if borrowers:
            stats_heading = Paragraph("Portfolio Statistics", heading_style)
            story.append(stats_heading)
            
            # Calculate statistics
            df = pd.DataFrame([b.__dict__ for b in borrowers])
            
            stats_text = f"""
            <b>Borrower Demographics:</b><br/>
            • Average Age: {df['age'].mean():.1f} years<br/>
            • Median Income: ${df['income'].median():,.0f}<br/>
            • Average Credit Score: {df['credit_score'].mean():.0f}<br/>
            • Average Debt-to-Income: {df['debt_to_income'].mean():.1%}<br/><br/>
            
            <b>Loan Characteristics:</b><br/>
            • Average Loan Amount: ${df['loan_amount'].mean():,.0f}<br/>
            • Average Interest Rate: {df['interest_rate'].mean():.2f}%<br/>
            • Average Loan Term: {df['loan_term'].mean():.0f} months<br/><br/>
            
            <b>Geographic Distribution (Top 5 States):</b><br/>
            """
            
            # Add top 5 states
            top_states = df['state'].value_counts().head(5)
            for state, count in top_states.items():
                pct = count / len(df) * 100
                stats_text += f"• {state}: {count:,} loans ({pct:.1f}%)<br/>"
            
            stats_para = Paragraph(stats_text, styles['Normal'])
            story.append(stats_para)
            story.append(Spacer(1, 20))
        
        # Recommendations
        rec_heading = Paragraph("Risk Management Recommendations", heading_style)
        story.append(rec_heading)
        
        recommendations = """
        <b>Immediate Actions:</b><br/>
        • Monitor borrowers in BB+ risk grades for early warning signals<br/>
        • Review geographic concentration limits, particularly in high-unemployment states<br/>
        • Implement enhanced monitoring for borrowers in volatile industries<br/><br/>
        
        <b>Strategic Considerations:</b><br/>
        • Consider tightening underwriting standards if economic conditions deteriorate<br/>
        • Evaluate pricing adjustments for higher-risk segments<br/>
        • Enhance stress testing frequency during economic uncertainty<br/>
        • Review and update risk appetite statements based on current portfolio composition<br/><br/>
        
        <b>Regulatory Compliance:</b><br/>
        • Ensure adequate capital reserves based on current Expected Loss calculations<br/>
        • Document stress testing methodologies for regulatory review<br/>
        • Maintain robust model validation and back-testing procedures<br/>
        """
        
        rec_para = Paragraph(recommendations, styles['Normal'])
        story.append(rec_para)
        story.append(Spacer(1, 20))
        
        # Footer
        footer_text = f"""
        <i>This report was generated by BankRisk Pro, an advanced credit risk management platform.
        For questions about methodology or findings, please contact the Risk Management team.</i><br/><br/>
        
        <b>Disclaimer:</b> This analysis is based on synthetic data and machine learning models. 
        Results should be validated with actual portfolio data and supplemented with expert judgment 
        before making business decisions.
        """
        
        footer_para = Paragraph(footer_text, styles['Normal'])
        story.append(footer_para)
        
        # Build PDF
        doc.build(story)
        
        print(f"📄 Executive report generated: {filename}")
        return filename
        
    except ImportError:
        # Fallback to text report if reportlab not available
        print("⚠️  ReportLab not installed. Generating text report instead.")
        return generate_text_report(borrowers, portfolio_metrics, economic_snapshot, filename)
    
    except Exception as e:
        print(f"❌ Error generating PDF report: {e}")
        return ""

def generate_text_report(borrowers: List, portfolio_metrics, economic_snapshot, 
                        filename: Optional[str] = None) -> str:
    """
    Generate text-based executive report as fallback
    """
    
    if not filename:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"reports/bankrisk_executive_report_{timestamp}.txt"
    
    try:
        with open(filename, 'w') as f:
            f.write("BankRisk Pro - Credit Risk Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Report Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}\n")
            f.write(f"Analysis Method: Machine Learning Credit Risk Models\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 20 + "\n")
            
            if portfolio_metrics:
                f.write(f"Total Portfolio Exposure: ${portfolio_metrics.total_exposure:,.0f}\n")
                f.write(f"Number of Loans: {portfolio_metrics.number_of_loans:,}\n")
                f.write(f"Average Probability of Default: {portfolio_metrics.average_pd:.2f}%\n")
                f.write(f"Expected Loss: ${portfolio_metrics.expected_loss:,.0f}\n")
                f.write(f"Portfolio VaR (95%): ${portfolio_metrics.portfolio_var_95:,.0f}\n")
                f.write(f"Portfolio VaR (99%): ${portfolio_metrics.portfolio_var_99:,.0f}\n")
                f.write(f"Concentration Risk (HHI): {portfolio_metrics.concentration_hhi:.3f}\n\n")
            
            # Economic Environment
            f.write("ECONOMIC ENVIRONMENT\n")
            f.write("-" * 20 + "\n")
            
            if economic_snapshot:
                f.write(f"Federal Funds Rate: {economic_snapshot.federal_funds_rate:.2f}%\n")
                f.write(f"Unemployment Rate: {economic_snapshot.unemployment_rate:.1f}%\n")
                f.write(f"GDP Growth: {economic_snapshot.gdp_growth:.1f}%\n")
                f.write(f"Inflation Rate: {economic_snapshot.inflation_rate:.1f}%\n")
                f.write(f"Credit Spread: {economic_snapshot.credit_spread:.2f}%\n")
                f.write(f"Market Volatility (VIX): {economic_snapshot.vix_volatility:.1f}\n\n")
            
            # Risk Grade Distribution
            if portfolio_metrics and portfolio_metrics.grade_distribution:
                f.write("RISK GRADE DISTRIBUTION\n")
                f.write("-" * 23 + "\n")
                
                total_loans = portfolio_metrics.number_of_loans
                for grade, percentage in sorted(portfolio_metrics.grade_distribution.items()):
                    num_loans = int(total_loans * percentage / 100)
                    f.write(f"Grade {grade}: {num_loans:,} loans ({percentage:.1f}%)\n")
                f.write("\n")
            
            # Portfolio Statistics
            if borrowers:
                df = pd.DataFrame([b.__dict__ for b in borrowers])
                
                f.write("PORTFOLIO STATISTICS\n")
                f.write("-" * 19 + "\n")
                f.write(f"Average Age: {df['age'].mean():.1f} years\n")
                f.write(f"Median Income: ${df['income'].median():,.0f}\n")
                f.write(f"Average Credit Score: {df['credit_score'].mean():.0f}\n")
                f.write(f"Average Debt-to-Income: {df['debt_to_income'].mean():.1%}\n")
                f.write(f"Average Loan Amount: ${df['loan_amount'].mean():,.0f}\n")
                f.write(f"Average Interest Rate: {df['interest_rate'].mean():.2f}%\n\n")
            
            f.write("RISK MANAGEMENT RECOMMENDATIONS\n")
            f.write("-" * 32 + "\n")
            f.write("• Monitor high-risk borrowers for early warning signals\n")
            f.write("• Review geographic and industry concentration limits\n")
            f.write("• Consider stress testing under adverse economic scenarios\n")
            f.write("• Ensure adequate capital reserves for expected losses\n")
            f.write("• Update risk appetite based on current portfolio composition\n\n")
            
            f.write("Report generated by BankRisk Pro Credit Risk Management Platform\n")
        
        print(f"📄 Text report generated: {filename}")
        return filename
        
    except Exception as e:
        print(f"❌ Error generating text report: {e}")
        return ""

def validate_api_keys() -> Dict[str, bool]:
    """
    Validate API keys and return status
    
    Returns:
        Dictionary with API key validation status
    """
    
    from dotenv import load_dotenv
    load_dotenv()
    
    validation_results = {}
    
    # Check FRED API key
    fred_key = os.getenv('FRED_API_KEY')
    if fred_key and fred_key != 'your_fred_api_key_here':
        # Test FRED API with a simple request
        try:
            import requests
            url = "https://api.stlouisfed.org/fred/series"
            params = {'series_id': 'FEDFUNDS', 'api_key': fred_key, 'file_type': 'json', 'limit': 1}
            response = requests.get(url, params=params, timeout=5)
            validation_results['FRED'] = response.status_code == 200
        except:
            validation_results['FRED'] = False
    else:
        validation_results['FRED'] = False
    
    # Check Alpha Vantage API key
    av_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if av_key and av_key != 'your_alpha_vantage_key_here':
        validation_results['Alpha_Vantage'] = True  # Basic check
    else:
        validation_results['Alpha_Vantage'] = False
    
    return validation_results

def create_project_structure():
    """Create complete project directory structure"""
    
    directories = [
        'data',
        'models/saved_models',
        'outputs/csv',
        'outputs/plots', 
        'reports/pdf',
        'reports/html',
        'dashboards',
        'config',
        'logs',
        'utils',
        'tests'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Create __init__.py files for Python packages
    init_dirs = ['utils']
    for init_dir in init_dirs:
        init_file = os.path.join(init_dir, '__init__.py')
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('"""BankRisk Pro utilities package"""\n')
    
    print("✅ Project structure created")

def format_currency(amount: float) -> str:
    """Format currency values for display"""
    if amount >= 1e9:
        return f"${amount/1e9:.1f}B"
    elif amount >= 1e6:
        return f"${amount/1e6:.1f}M"
    elif amount >= 1e3:
        return f"${amount/1e3:.1f}K"
    else:
        return f"${amount:.0f}"

def format_percentage(value: float, decimals: int = 2) -> str:
    """Format percentage values for display"""
    return f"{value:.{decimals}f}%"

def calculate_portfolio_diversification(borrowers: List) -> Dict[str, float]:
    """
    Calculate portfolio diversification metrics
    
    Returns:
        Dictionary with diversification metrics
    """
    
    if not borrowers:
        return {}
    
    df = pd.DataFrame([b.__dict__ for b in borrowers])
    total_exposure = df['loan_amount'].sum()
    
    diversification_metrics = {}
    
    # Geographic diversification (HHI by state)
    state_exposure = df.groupby('state')['loan_amount'].sum()
    state_weights = state_exposure / total_exposure
    geographic_hhi = (state_weights ** 2).sum()
    diversification_metrics['geographic_hhi'] = geographic_hhi
    diversification_metrics['geographic_diversification'] = 1 - geographic_hhi
    
    # Industry diversification
    industry_exposure = df.groupby('industry')['loan_amount'].sum()
    industry_weights = industry_exposure / total_exposure
    industry_hhi = (industry_weights ** 2).sum()
    diversification_metrics['industry_hhi'] = industry_hhi
    diversification_metrics['industry_diversification'] = 1 - industry_hhi
    
    # Credit score diversification
    credit_bins = pd.cut(df['credit_score'], bins=[0, 580, 670, 740, 800, 850], 
                        labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
    credit_exposure = df.groupby(credit_bins)['loan_amount'].sum()
    credit_weights = credit_exposure / total_exposure
    credit_hhi = (credit_weights ** 2).sum()
    diversification_metrics['credit_hhi'] = credit_hhi
    diversification_metrics['credit_diversification'] = 1 - credit_hhi
    
    return diversification_metrics

def log_analysis_run(analysis_type: str, portfolio_size: int, results: Dict[str, Any]):
    """Log analysis runs for audit trail"""
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'analysis_type': analysis_type,
        'portfolio_size': portfolio_size,
        'results_summary': results
    }
    
    log_file = 'logs/analysis_log.json'
    
    try:
        # Read existing log
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = pd.read_json(f, lines=True).to_dict('records')
        else:
            logs = []
        
        # Add new entry
        logs.append(log_entry)
        
        # Write back to file
        with open(log_file, 'w') as f:
            for log in logs:
                f.write(pd.Series(log).to_json() + '\n')
                
    except Exception as e:
        print(f"⚠️  Could not write to log file: {e}")

def get_system_info() -> Dict[str, str]:
    """Get system information for diagnostics"""
    
    import platform
    
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'processor': platform.processor(),
        'timestamp': datetime.now().isoformat()
    }
    
    # Check package versions
    try:
        import pandas as pd
        info['pandas_version'] = pd.__version__
    except:
        info['pandas_version'] = 'Not installed'
    
    try:
        import sklearn
        info['sklearn_version'] = sklearn.__version__
    except:
        info['sklearn_version'] = 'Not installed'
    
    try:
        import numpy as np
        info['numpy_version'] = np.__version__
    except:
        info['numpy_version'] = 'Not installed'
    
    return info