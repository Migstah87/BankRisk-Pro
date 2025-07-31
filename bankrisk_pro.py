#!/usr/bin/env python3
"""
BankRisk Pro - Credit Risk Management Platform
==============================================

A comprehensive banking risk management platform featuring:
- Real-time economic data integration
- ML-powered credit scoring
- Portfolio risk analytics
- Regulatory stress testing
- Executive dashboards

Author: [Your Name]
Created for Banking & Risk Management Professionals
"""

import os
import sys
from datetime import datetime
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_collector import EconomicDataCollector
from portfolio_generator import SyntheticPortfolioGenerator
from credit_models import CreditRiskModeler
from risk_analytics import PortfolioRiskAnalyzer
from visualizations import RiskDashboard
from stress_testing import StressTester
from utils.helpers import setup_environment, save_results_to_csv, generate_executive_report

class BankRiskPro:
    """
    Main BankRisk Pro Platform
    
    Comprehensive credit risk management system for banking institutions.
    Integrates economic data, ML models, and regulatory compliance tools.
    """
    
    def __init__(self):
        """Initialize the BankRisk Pro platform"""
        print("🏦 BankRisk Pro - Credit Risk Management Platform")
        print("=" * 60)
        
        # Initialize components
        self.economic_collector = EconomicDataCollector()
        self.portfolio_generator = SyntheticPortfolioGenerator(self.economic_collector)
        self.credit_modeler = CreditRiskModeler()
        self.risk_analyzer = PortfolioRiskAnalyzer(self.economic_collector)
        self.dashboard = RiskDashboard()
        self.stress_tester = StressTester(self.economic_collector)
        
        # Session data
        self.current_portfolio = []
        self.economic_snapshot = None
        self.model_performance = {}
        self.portfolio_metrics = None
        
        print("✅ Platform initialized successfully")
        self._display_system_status()
    
    def _display_system_status(self):
        """Display current system status"""
        print(f"\n📊 SYSTEM STATUS:")
        print(f"   Economic Data Source: {'FRED API' if self.economic_collector.fred_api_key else 'Fallback Data'}")
        print(f"   Market Data Source: Yahoo Finance")
        print(f"   ML Models: Ready for Training")
        print(f"   Current Portfolio: {len(self.current_portfolio):,} borrowers")
        
        # Get current economic snapshot
        try:
            self.economic_snapshot = self.economic_collector.get_current_economic_snapshot()
            print(f"\n🌍 CURRENT ECONOMIC CONDITIONS:")
            print(f"   Federal Funds Rate: {self.economic_snapshot.federal_funds_rate:.2f}%")
            print(f"   Unemployment Rate: {self.economic_snapshot.unemployment_rate:.1f}%")
            print(f"   VIX Volatility: {self.economic_snapshot.vix_volatility:.1f}")
            print(f"   Credit Spread: {self.economic_snapshot.credit_spread:.2f}%")
        except Exception as e:
            print(f"⚠️  Could not fetch economic data: {e}")
    
    def generate_portfolio(self, size: int = 10000, force_regenerate: bool = False):
        """Generate synthetic loan portfolio"""
        if self.current_portfolio and not force_regenerate:
            print(f"📊 Using existing portfolio of {len(self.current_portfolio):,} borrowers")
            return
        
        print(f"\n🏭 PORTFOLIO GENERATION")
        print("-" * 30)
        
        try:
            self.current_portfolio = self.portfolio_generator.generate_portfolio(size)
            print(f"✅ Generated portfolio of {len(self.current_portfolio):,} synthetic borrowers")
            
            # Display portfolio summary
            self._display_portfolio_summary()
            
        except Exception as e:
            print(f"❌ Error generating portfolio: {e}")
            raise
    
    def _display_portfolio_summary(self):
        """Display portfolio composition summary"""
        if not self.current_portfolio:
            return
        
        import pandas as pd
        df = pd.DataFrame([borrower.__dict__ for borrower in self.current_portfolio])
        
        print(f"\n📋 PORTFOLIO COMPOSITION:")
        print(f"   Total Exposure: ${df['loan_amount'].sum():,.0f}")
        print(f"   Average Loan Size: ${df['loan_amount'].mean():,.0f}")
        print(f"   Average Credit Score: {df['credit_score'].mean():.0f}")
        print(f"   Average DTI Ratio: {df['debt_to_income'].mean():.1%}")
        
        # Top states by exposure
        state_exposure = df.groupby('state')['loan_amount'].sum().sort_values(ascending=False).head(5)
        print(f"\n   Top States by Exposure:")
        for state, exposure in state_exposure.items():
            print(f"     {state}: ${exposure:,.0f}")
        
        # Industry distribution
        industry_dist = df['industry'].value_counts().head(5)
        print(f"\n   Top Industries:")
        for industry, count in industry_dist.items():
            print(f"     {industry}: {count:,} loans")
    
    def train_credit_models(self):
        """Train machine learning credit risk models"""
        if not self.current_portfolio:
            print("❌ No portfolio available. Generate portfolio first.")
            return
        
        print(f"\n🤖 CREDIT MODEL TRAINING")
        print("-" * 30)
        
        try:
            self.model_performance = self.credit_modeler.train_models(self.current_portfolio)
            
            print(f"\n📊 MODEL TRAINING RESULTS:")
            for model_name, metrics in self.model_performance.items():
                print(f"   {model_name.title()}: AUC = {metrics['auc_score']:.3f}")
            
            # Update portfolio with predictions
            self.current_portfolio = self.credit_modeler.predict_default_probability(
                self.current_portfolio, 'random_forest'
            )
            
            print(f"✅ Portfolio updated with credit scores and risk grades")
            
        except Exception as e:
            print(f"❌ Error training models: {e}")
            raise
    
    def analyze_portfolio_risk(self):
        """Perform comprehensive portfolio risk analysis"""
        if not self.current_portfolio:
            print("❌ No portfolio available. Generate portfolio first.")
            return
        
        if not hasattr(self.current_portfolio[0], 'probability_default'):
            print("❌ Credit models not trained. Train models first.")
            return
        
        print(f"\n📊 PORTFOLIO RISK ANALYSIS")
        print("-" * 30)
        
        try:
            self.portfolio_metrics = self.risk_analyzer.calculate_portfolio_metrics(self.current_portfolio)
            
            # Display key metrics
            print(f"📈 KEY RISK METRICS:")
            print(f"   Total Exposure: ${self.portfolio_metrics.total_exposure:,.0f}")
            print(f"   Expected Loss: ${self.portfolio_metrics.expected_loss:,.0f}")
            print(f"   Portfolio VaR (95%): ${self.portfolio_metrics.portfolio_var_95:,.0f}")
            print(f"   Portfolio VaR (99%): ${self.portfolio_metrics.portfolio_var_99:,.0f}")
            print(f"   Average PD: {self.portfolio_metrics.average_pd:.2f}%")
            print(f"   Concentration (HHI): {self.portfolio_metrics.concentration_hhi:.3f}")
            
            # Grade distribution
            print(f"\n🎯 RISK GRADE DISTRIBUTION:")
            for grade, percentage in sorted(self.portfolio_metrics.grade_distribution.items()):
                print(f"   Grade {grade}: {percentage:.1f}%")
            
        except Exception as e:
            print(f"❌ Error analyzing portfolio: {e}")
            raise
    
    def run_stress_tests(self, scenarios: Optional[List[str]] = None):
        """Run regulatory stress testing scenarios"""
        if not self.current_portfolio:
            print("❌ No portfolio available. Generate portfolio first.")
            return
        
        print(f"\n⚡ STRESS TESTING")
        print("-" * 30)
        
        if scenarios is None:
            scenarios = ['recession', 'interest_rate_shock', 'unemployment_spike']
        
        try:
            stress_results = {}
            
            for scenario in scenarios:
                print(f"\n🧪 Running {scenario.replace('_', ' ').title()} scenario...")
                results = self.stress_tester.run_stress_scenario(self.current_portfolio, scenario)
                stress_results[scenario] = results
                
                print(f"   Stressed Expected Loss: ${results.stressed_expected_loss:,.0f}")
                print(f"   Loss Increase: {results.loss_increase_pct:.1f}%")
                print(f"   Capital Impact: ${results.capital_impact:,.0f}")
            
            self.stress_results = stress_results
            print(f"\n✅ Stress testing completed for {len(scenarios)} scenarios")
            
        except Exception as e:
            print(f"❌ Error running stress tests: {e}")
            raise
    
    def create_dashboard(self, save_path: str = None):
        """Create comprehensive risk management dashboard"""
        if not self.portfolio_metrics:
            print("❌ Portfolio analysis not completed. Run analysis first.")
            return
        
        print(f"\n📊 CREATING RISK DASHBOARD")
        print("-" * 30)
        
        try:
            dashboard_data = {
                'portfolio': self.current_portfolio,
                'metrics': self.portfolio_metrics,
                'economic_snapshot': self.economic_snapshot,
                'model_performance': self.model_performance,
                'stress_results': getattr(self, 'stress_results', {})
            }
            
            if save_path:
                self.dashboard.create_executive_dashboard(dashboard_data, save_path)
                print(f"✅ Dashboard saved to: {save_path}")
            else:
                self.dashboard.create_executive_dashboard(dashboard_data)
                print(f"✅ Dashboard created successfully")
                
        except Exception as e:
            print(f"❌ Error creating dashboard: {e}")
            raise
    
    def export_results(self, export_format: str = 'csv'):
        """Export analysis results"""
        if not self.current_portfolio:
            print("❌ No results to export")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            if export_format.lower() == 'csv':
                filename = f"bankrisk_analysis_{timestamp}.csv"
                save_results_to_csv(self.current_portfolio, self.portfolio_metrics, filename)
                print(f"✅ Results exported to: {filename}")
            
            elif export_format.lower() == 'report':
                filename = f"bankrisk_executive_report_{timestamp}.pdf"
                generate_executive_report(
                    self.current_portfolio, 
                    self.portfolio_metrics, 
                    self.economic_snapshot,
                    filename
                )
                print(f"✅ Executive report generated: {filename}")
            
        except Exception as e:
            print(f"❌ Error exporting results: {e}")
    
    def run_full_analysis(self, portfolio_size: int = 10000):
        """Run complete end-to-end analysis"""
        print(f"\n🚀 FULL RISK ANALYSIS PIPELINE")
        print("=" * 50)
        
        try:
            # Step 1: Generate Portfolio
            print(f"Step 1/6: Generating portfolio...")
            self.generate_portfolio(portfolio_size)
            
            # Step 2: Train Models
            print(f"\nStep 2/6: Training credit models...")
            self.train_credit_models()
            
            # Step 3: Analyze Risk
            print(f"\nStep 3/6: Analyzing portfolio risk...")
            self.analyze_portfolio_risk()
            
            # Step 4: Stress Testing
            print(f"\nStep 4/6: Running stress tests...")
            self.run_stress_tests()
            
            # Step 5: Create Dashboard
            print(f"\nStep 5/6: Creating dashboard...")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            dashboard_path = f"bankrisk_dashboard_{timestamp}.html"
            self.create_dashboard(dashboard_path)
            
            # Step 6: Export Results
            print(f"\nStep 6/6: Exporting results...")
            self.export_results('csv')
            self.export_results('report')
            
            print(f"\n🎉 ANALYSIS COMPLETE!")
            print(f"📁 Generated Files:")
            print(f"   📊 Dashboard: bankrisk_dashboard_{timestamp}.html")
            print(f"   📈 Data: bankrisk_analysis_{timestamp}.csv")
            print(f"   📄 Report: bankrisk_executive_report_{timestamp}.pdf")
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            raise

def main():
    """Main interactive interface"""
    
    # Setup environment
    setup_environment()
    
    # Initialize platform
    platform = BankRiskPro()
    
    while True:
        print(f"\n🏦 BANKRISK PRO - MAIN MENU")
        print("=" * 40)
        print("1. 📊 Generate Synthetic Portfolio")
        print("2. 🤖 Train Credit Risk Models")
        print("3. 📈 Analyze Portfolio Risk")
        print("4. ⚡ Run Stress Tests")
        print("5. 📋 Create Risk Dashboard")
        print("6. 💾 Export Results")
        print("7. 🚀 Run Full Analysis Pipeline")
        print("8. 📊 View Current Status")
        print("9. 🎮 Demo Mode (Quick Analysis)")
        print("10. ❌ Exit")
        
        try:
            choice = input(f"\n🎯 Select option (1-10): ").strip()
            
            if choice == '1':
                size = input("Portfolio size [10000]: ").strip()
                size = int(size) if size else 10000
                platform.generate_portfolio(size)
                
            elif choice == '2':
                platform.train_credit_models()
                
            elif choice == '3':
                platform.analyze_portfolio_risk()
                
            elif choice == '4':
                print("Available scenarios: recession, interest_rate_shock, unemployment_spike")
                scenarios_input = input("Enter scenarios (comma-separated) or press Enter for all: ").strip()
                scenarios = [s.strip() for s in scenarios_input.split(',')] if scenarios_input else None
                platform.run_stress_tests(scenarios)
                
            elif choice == '5':
                save_option = input("Save dashboard to file? (y/n): ").strip().lower()
                if save_option == 'y':
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    save_path = f"bankrisk_dashboard_{timestamp}.html"
                    platform.create_dashboard(save_path)
                else:
                    platform.create_dashboard()
                
            elif choice == '6':
                export_format = input("Export format (csv/report): ").strip().lower()
                platform.export_results(export_format)
                
            elif choice == '7':
                size = input("Portfolio size for full analysis [5000]: ").strip()
                size = int(size) if size else 5000
                platform.run_full_analysis(size)
                
            elif choice == '8':
                platform._display_system_status()
                if platform.portfolio_metrics:
                    print(f"\n📊 PORTFOLIO STATUS:")
                    print(f"   Loans: {platform.portfolio_metrics.number_of_loans:,}")
                    print(f"   Total Exposure: ${platform.portfolio_metrics.total_exposure:,.0f}")
                    print(f"   Expected Loss: ${platform.portfolio_metrics.expected_loss:,.0f}")
                
            elif choice == '9':
                print("🎮 Running demo analysis with 2,000 borrowers...")
                platform.run_full_analysis(2000)
                
            elif choice == '10':
                print("👋 Thank you for using BankRisk Pro!")
                break
                
            else:
                print("❌ Invalid option. Please select 1-10.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
        
        # Continue prompt
        if choice in ['1', '2', '3', '4', '5', '6', '7', '9']:
            input("\n📍 Press Enter to continue...")

if __name__ == "__main__":
    main()
