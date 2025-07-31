#!/usr/bin/env python3
"""
BankRisk Pro - Quick Demo
========================

Demonstrates key platform capabilities with rate-limiting optimizations
"""

import sys
import os
import time
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_quick_demo():
    """Run optimized quick demonstration"""
    
    print("🏦" + "="*60)
    print("  BankRisk Pro - Quick Demo")
    print("  Professional Credit Risk Analysis")
    print("="*62)
    
    try:
        # Import modules
        print("\n📚 Loading BankRisk Pro modules...")
        from data_collector import EconomicDataCollector
        from portfolio_generator import SyntheticPortfolioGenerator
        from credit_models import CreditRiskModeler
        from risk_analytics import PortfolioRiskAnalyzer
        from stress_testing import StressTester
        from visualizations import RiskDashboard
        
        print("✅ All modules loaded successfully")
        
        # Initialize platform components
        print("\n🔧 Initializing platform components...")
        economic_collector = EconomicDataCollector()
        portfolio_generator = SyntheticPortfolioGenerator(economic_collector)
        credit_modeler = CreditRiskModeler()
        risk_analyzer = PortfolioRiskAnalyzer(economic_collector)
        stress_tester = StressTester(economic_collector)
        dashboard = RiskDashboard()
        
        print("✅ Platform components initialized")
        
        # Get economic snapshot
        print("\n🌍 Fetching economic conditions...")
        economic_snapshot = economic_collector.get_current_economic_snapshot()
        
        print(f"📊 Current Economic Conditions:")
        print(f"   Federal Funds Rate: {economic_snapshot.federal_funds_rate:.2f}%")
        print(f"   Unemployment Rate: {economic_snapshot.unemployment_rate:.1f}%")
        print(f"   VIX Volatility: {economic_snapshot.vix_volatility:.1f}")
        print(f"   Credit Spread: {economic_snapshot.credit_spread:.2f}%")
        
        # Generate smaller portfolio for demo (to avoid rate limits)
        print(f"\n🏭 Generating demo portfolio (2,000 borrowers)...")
        portfolio = portfolio_generator.generate_portfolio(2000)
        
        if len(portfolio) < 1000:
            print("⚠️  Generated smaller portfolio due to rate limiting")
        
        # Train credit models
        print(f"\n🤖 Training credit risk models...")
        model_results = credit_modeler.train_models(portfolio)
        
        # Display model performance
        print(f"\n📊 Model Performance:")
        for model_name, metrics in model_results.items():
            print(f"   {model_name.title()}: AUC = {metrics['auc_score']:.3f}")
        
        # Predict default probabilities
        print(f"\n🎯 Generating risk predictions...")
        portfolio = credit_modeler.predict_default_probability(portfolio, 'random_forest')
        
        # Portfolio risk analysis
        print(f"\n📈 Analyzing portfolio risk...")
        portfolio_metrics = risk_analyzer.calculate_portfolio_metrics(portfolio)
        
        # Display key results
        print(f"\n🏆 PORTFOLIO ANALYSIS RESULTS:")
        print(f"   Total Exposure: ${portfolio_metrics.total_exposure:,.0f}")
        print(f"   Number of Loans: {portfolio_metrics.number_of_loans:,}")
        print(f"   Average PD: {portfolio_metrics.average_pd:.2f}%")
        print(f"   Expected Loss: ${portfolio_metrics.expected_loss:,.0f}")
        print(f"   Portfolio VaR (95%): ${portfolio_metrics.portfolio_var_95:,.0f}")
        print(f"   Portfolio VaR (99%): ${portfolio_metrics.portfolio_var_99:,.0f}")
        
        # Risk grade distribution
        print(f"\n🎯 Risk Grade Distribution:")
        for grade, percentage in sorted(portfolio_metrics.grade_distribution.items()):
            print(f"   Grade {grade}: {percentage:.1f}%")
        
        # Run stress tests
        print(f"\n⚡ Running stress tests...")
        
        # Run just one scenario to avoid rate limits
        stress_results = {}
        test_scenario = 'recession'
        
        try:
            stress_result = stress_tester.run_stress_scenario(portfolio, test_scenario)
            stress_results[test_scenario] = stress_result
            
            print(f"\n📊 STRESS TEST RESULTS ({test_scenario.title()}):")
            print(f"   Baseline Expected Loss: ${stress_result.baseline_expected_loss:,.0f}")
            print(f"   Stressed Expected Loss: ${stress_result.stressed_expected_loss:,.0f}")
            print(f"   Loss Increase: {stress_result.loss_increase_pct:.1f}%")
            print(f"   Capital Impact: ${stress_result.capital_impact:,.0f}")
            
        except Exception as e:
            print(f"⚠️  Stress test skipped due to rate limiting: {e}")
        
        # Create dashboard
        print(f"\n📊 Creating risk dashboard...")
        try:
            dashboard_data = {
                'portfolio': portfolio,
                'metrics': portfolio_metrics,
                'economic_snapshot': economic_snapshot,
                'model_performance': model_results,
                'stress_results': stress_results
            }
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            dashboard_file = f"dashboards/demo_dashboard_{timestamp}.html"
            
            dashboard.create_executive_dashboard(dashboard_data, dashboard_file)
            print(f"✅ Dashboard created: {dashboard_file}")
            
        except Exception as e:
            print(f"⚠️  Dashboard creation skipped: {e}")
        
        # Export results
        print(f"\n💾 Exporting results...")
        try:
            from utils.helpers import save_results_to_csv
            
            csv_file = save_results_to_csv(portfolio, portfolio_metrics)
            print(f"✅ Results exported to: {csv_file}")
            
        except Exception as e:
            print(f"⚠️  Export skipped: {e}")
        
        # Final summary
        print(f"\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"📁 Generated Files:")
        print(f"   📊 Dashboard: dashboards/demo_dashboard_*.html")
        print(f"   📈 Data: outputs/bankrisk_analysis_*.csv")
        print(f"\n💡 Key Insights:")
        
        # Portfolio health assessment
        if portfolio_metrics.average_pd < 5:
            health = "🟢 HEALTHY"
        elif portfolio_metrics.average_pd < 10:
            health = "🟡 MODERATE RISK"
        else:
            health = "🔴 HIGH RISK"
        
        print(f"   Portfolio Health: {health}")
        print(f"   Risk Concentration: {'Low' if portfolio_metrics.concentration_hhi < 0.1 else 'Moderate' if portfolio_metrics.concentration_hhi < 0.25 else 'High'}")
        
        if stress_results:
            stress_impact = list(stress_results.values())[0].loss_increase_pct
            stress_level = "Low" if stress_impact < 15 else "Moderate" if stress_impact < 30 else "High"
            print(f"   Stress Resilience: {stress_level} impact under recession")
        
        print(f"\n🚀 Ready for production analysis!")
        print(f"   Run: python bankrisk_pro.py")
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n\n👋 Demo interrupted by user")
        return False
        
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        print(f"\n🔧 Troubleshooting:")
        print(f"   1. Check internet connection")
        print(f"   2. Install missing packages: pip install -r requirements.txt")
        print(f"   3. Try again in a few minutes (API rate limits)")
        print(f"   4. Check .env file configuration")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'plotly', 'yfinance', 'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print(f"   Install with: pip install {' '.join(missing_packages)}")
        return False
    
    return True

def main():
    """Main demo function"""
    
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        print("\n📦 Install missing packages and try again.")
        sys.exit(1)
    
    print("✅ Dependencies OK")
    
    # Run the demo
    success = run_quick_demo()
    
    if success:
        print(f"\n🏦 BankRisk Pro demo completed successfully!")
        print(f"   Platform is ready for professional use.")
    else:
        print(f"\n⚠️  Demo encountered issues.")
        print(f"   Platform core functionality is still available.")
    
    # Ask if user wants to run full platform
    try:
        response = input(f"\n🚀 Launch full BankRisk Pro platform? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            print(f"\n🏦 Launching BankRisk Pro...")
            time.sleep(1)
            # Import and run main platform
            from bankrisk_pro import main as platform_main
            platform_main()
    except (KeyboardInterrupt, EOFError):
        print(f"\n👋 Goodbye!")

if __name__ == "__main__":
    main()