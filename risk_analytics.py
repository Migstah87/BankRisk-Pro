"""
Risk Analytics Module
====================

Portfolio-level risk analysis including:
- Value at Risk (VaR) calculations
- Expected Loss computations
- Concentration risk metrics
- Portfolio optimization
- Regulatory capital calculations
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PortfolioMetrics:
    """Portfolio-level risk metrics"""
    total_exposure: float
    number_of_loans: int
    average_pd: float
    portfolio_var_95: float
    portfolio_var_99: float
    expected_loss: float
    unexpected_loss: float
    diversification_ratio: float
    concentration_hhi: float
    grade_distribution: Dict[str, float]
    sector_concentration: Dict[str, float] = None
    geographic_concentration: Dict[str, float] = None

class PortfolioRiskAnalyzer:
    """
    Comprehensive portfolio risk analysis system
    """
    
    def __init__(self, economic_data_collector):
        """
        Initialize portfolio risk analyzer
        
        Args:
            economic_data_collector: Economic data source for stress testing
        """
        self.economic_data = economic_data_collector
        
        # Risk parameters
        self.lgd_assumption = 0.45  # Loss Given Default
        self.confidence_levels = [0.95, 0.99, 0.999]
        
        print("📊 Portfolio Risk Analyzer initialized")
    
    def calculate_portfolio_metrics(self, borrowers: List) -> PortfolioMetrics:
        """
        Calculate comprehensive portfolio risk metrics
        
        Args:
            borrowers: List of Borrower objects with risk scores
            
        Returns:
            PortfolioMetrics object with all calculated metrics
        """
        
        print("📊 Calculating portfolio risk metrics...")
        
        if not borrowers:
            raise ValueError("No borrowers provided for analysis")
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame([borrower.__dict__ for borrower in borrowers])
        
        # Basic portfolio statistics
        total_exposure = df['loan_amount'].sum()
        number_of_loans = len(borrowers)
        
        # Check if we have risk metrics
        if 'probability_default' not in df.columns or df['probability_default'].isna().all():
            print("⚠️  No default probabilities found. Using estimated values.")
            # Use estimated PDs based on credit scores
            df['probability_default'] = self._estimate_pd_from_credit_score(df['credit_score'])
        
        average_pd = df['probability_default'].mean()
        
        # Expected Loss calculation
        if 'expected_loss' in df.columns and not df['expected_loss'].isna().all():
            expected_loss = df['expected_loss'].sum()
        else:
            # Calculate EL = PD * LGD * EAD
            df['expected_loss'] = (df['probability_default'] / 100) * self.lgd_assumption * df['loan_amount']
            expected_loss = df['expected_loss'].sum()
        
        # Portfolio VaR calculation
        portfolio_var_95, portfolio_var_99 = self._calculate_portfolio_var(df)
        
        # Unexpected Loss
        unexpected_loss = portfolio_var_95 - expected_loss
        
        # Diversification metrics
        diversification_ratio = self._calculate_diversification_ratio(df)
        concentration_hhi = self._calculate_concentration_hhi(df)
        
        # Risk grade distribution
        grade_distribution = self._calculate_grade_distribution(df)
        
        # Sector and geographic concentration
        sector_concentration = self._calculate_sector_concentration(df)
        geographic_concentration = self._calculate_geographic_concentration(df)
        
        metrics = PortfolioMetrics(
            total_exposure=total_exposure,
            number_of_loans=number_of_loans,
            average_pd=average_pd,
            portfolio_var_95=portfolio_var_95,
            portfolio_var_99=portfolio_var_99,
            expected_loss=expected_loss,
            unexpected_loss=max(0, unexpected_loss),
            diversification_ratio=diversification_ratio,
            concentration_hhi=concentration_hhi,
            grade_distribution=grade_distribution,
            sector_concentration=sector_concentration,
            geographic_concentration=geographic_concentration
        )
        
        print(f"✅ Portfolio metrics calculated for {number_of_loans:,} loans")
        
        return metrics
    
    def _estimate_pd_from_credit_score(self, credit_scores: pd.Series) -> pd.Series:
        """Estimate PD based on credit scores when no model predictions available"""
        
        # Typical PD ranges by credit score
        def score_to_pd(score):
            if score >= 800:
                return np.random.uniform(0.5, 1.5)
            elif score >= 750:
                return np.random.uniform(1.0, 3.0)
            elif score >= 700:
                return np.random.uniform(2.0, 5.0)
            elif score >= 650:
                return np.random.uniform(4.0, 8.0)
            elif score >= 600:
                return np.random.uniform(7.0, 15.0)
            elif score >= 550:
                return np.random.uniform(12.0, 25.0)
            else:
                return np.random.uniform(20.0, 40.0)
        
        return credit_scores.apply(score_to_pd)
    
    def _calculate_portfolio_var(self, df: pd.DataFrame, num_simulations: int = 10000) -> Tuple[float, float]:
        """
        Calculate Portfolio Value at Risk using Monte Carlo simulation
        
        Args:
            df: DataFrame with loan data
            num_simulations: Number of Monte Carlo simulations
            
        Returns:
            Tuple of (VaR_95%, VaR_99%)
        """
        
        print("   🎲 Running Monte Carlo simulation for VaR...")
        
        # Extract key variables
        loan_amounts = df['loan_amount'].values
        default_probs = df['probability_default'].values / 100  # Convert to decimal
        
        # Monte Carlo simulation
        portfolio_losses = []
        
        for sim in range(num_simulations):
            # Generate random defaults based on individual PDs
            random_draws = np.random.random(len(loan_amounts))
            defaults = random_draws < default_probs
            
            # Calculate total loss for this simulation
            total_loss = np.sum(defaults * loan_amounts * self.lgd_assumption)
            portfolio_losses.append(total_loss)
        
        portfolio_losses = np.array(portfolio_losses)
        
        # Calculate VaR at different confidence levels
        var_95 = np.percentile(portfolio_losses, 95)
        var_99 = np.percentile(portfolio_losses, 99)
        
        return var_95, var_99
    
    def _calculate_diversification_ratio(self, df: pd.DataFrame) -> float:
        """
        Calculate portfolio diversification ratio
        
        A measure of how well diversified the portfolio is
        Range: 0 (perfectly diversified) to 1 (concentrated)
        """
        
        # Calculate Herfindahl-Hirschman Index for multiple dimensions
        total_exposure = df['loan_amount'].sum()
        
        # Geographic diversification
        state_exposure = df.groupby('state')['loan_amount'].sum()
        state_weights = state_exposure / total_exposure
        geo_hhi = (state_weights ** 2).sum()
        
        # Industry diversification
        industry_exposure = df.groupby('industry')['loan_amount'].sum()
        industry_weights = industry_exposure / total_exposure
        industry_hhi = (industry_weights ** 2).sum()
        
        # Credit quality diversification
        credit_bins = pd.cut(df['credit_score'], bins=[0, 600, 700, 750, 800, 850])
        credit_exposure = df.groupby(credit_bins)['loan_amount'].sum()
        credit_weights = credit_exposure / total_exposure
        credit_hhi = (credit_weights ** 2).sum()
        
        # Combined diversification ratio (average of different dimensions)
        diversification_ratio = (geo_hhi + industry_hhi + credit_hhi) / 3
        
        return diversification_ratio
    
    def _calculate_concentration_hhi(self, df: pd.DataFrame) -> float:
        """
        Calculate Herfindahl-Hirschman Index for overall concentration
        """
        
        total_exposure = df['loan_amount'].sum()
        
        # Individual loan concentration (should be low for good diversification)
        loan_weights = df['loan_amount'] / total_exposure
        hhi = (loan_weights ** 2).sum()
        
        return hhi
    
    def _calculate_grade_distribution(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate distribution of risk grades"""
        
        if 'risk_grade' not in df.columns:
            return {}
        
        grade_counts = df['risk_grade'].value_counts()
        total_loans = len(df)
        
        grade_distribution = {}
        for grade, count in grade_counts.items():
            grade_distribution[grade] = (count / total_loans) * 100
        
        return grade_distribution
    
    def _calculate_sector_concentration(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate sector/industry concentration"""
        
        if 'industry' not in df.columns:
            return {}
        
        total_exposure = df['loan_amount'].sum()
        sector_exposure = df.groupby('industry')['loan_amount'].sum()
        
        sector_concentration = {}
        for sector, exposure in sector_exposure.items():
            sector_concentration[sector] = (exposure / total_exposure) * 100
        
        return dict(sorted(sector_concentration.items(), key=lambda x: x[1], reverse=True))
    
    def _calculate_geographic_concentration(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate geographic concentration"""
        
        if 'state' not in df.columns:
            return {}
        
        total_exposure = df['loan_amount'].sum()
        geo_exposure = df.groupby('state')['loan_amount'].sum()
        
        geo_concentration = {}
        for state, exposure in geo_exposure.items():
            geo_concentration[state] = (exposure / total_exposure) * 100
        
        return dict(sorted(geo_concentration.items(), key=lambda x: x[1], reverse=True))
    
    def calculate_regulatory_capital(self, portfolio_metrics: PortfolioMetrics, 
                                   capital_method: str = 'basel_iii') -> Dict[str, float]:
        """
        Calculate regulatory capital requirements
        
        Args:
            portfolio_metrics: Portfolio metrics object
            capital_method: Regulatory framework ('basel_iii', 'stress_test')
            
        Returns:
            Dictionary with capital calculations
        """
        
        print(f"💰 Calculating regulatory capital using {capital_method} method...")
        
        capital_calculations = {}
        
        if capital_method == 'basel_iii':
            # Basel III standardized approach calculations
            
            # Expected Loss (already calculated)
            expected_loss = portfolio_metrics.expected_loss
            
            # Unexpected Loss (UL) - simplified calculation
            unexpected_loss = portfolio_metrics.unexpected_loss
            
            # Risk-Weighted Assets (simplified)
            # Assuming average risk weight of 100% for unsecured consumer loans
            risk_weighted_assets = portfolio_metrics.total_exposure * 1.0
            
            # Minimum capital requirements (Basel III)
            # Common Equity Tier 1: 4.5%
            # Tier 1 Capital: 6%
            # Total Capital: 8%
            cet1_minimum = risk_weighted_assets * 0.045
            tier1_minimum = risk_weighted_assets * 0.06
            total_capital_minimum = risk_weighted_assets * 0.08
            
            # Capital Conservation Buffer: 2.5%
            conservation_buffer = risk_weighted_assets * 0.025
            
            # Total required capital with buffers
            total_required_capital = total_capital_minimum + conservation_buffer
            
            capital_calculations = {
                'expected_loss': expected_loss,
                'unexpected_loss': unexpected_loss,
                'risk_weighted_assets': risk_weighted_assets,
                'cet1_minimum': cet1_minimum,
                'tier1_minimum': tier1_minimum,
                'total_capital_minimum': total_capital_minimum,
                'conservation_buffer': conservation_buffer,
                'total_required_capital': total_required_capital,
                'capital_ratio_minimum': 10.5,  # 8% + 2.5% buffer
                'economic_capital': unexpected_loss  # Economic capital approximation
            }
            
        elif capital_method == 'stress_test':
            # CCAR/DFAST style stress testing capital
            
            # Severely Adverse Scenario multipliers (simplified)
            stress_multiplier = 2.5  # Increase losses by 150% under stress
            
            stressed_expected_loss = portfolio_metrics.expected_loss * stress_multiplier
            stressed_unexpected_loss = portfolio_metrics.unexpected_loss * stress_multiplier
            
            # Stress capital requirement
            stress_capital_requirement = stressed_expected_loss + stressed_unexpected_loss
            
            capital_calculations = {
                'baseline_expected_loss': portfolio_metrics.expected_loss,
                'stressed_expected_loss': stressed_expected_loss,
                'baseline_unexpected_loss': portfolio_metrics.unexpected_loss,
                'stressed_unexpected_loss': stressed_unexpected_loss,
                'stress_capital_requirement': stress_capital_requirement,
                'stress_multiplier': stress_multiplier
            }
        
        print(f"✅ Capital calculations completed")
        return capital_calculations
    
    def perform_back_testing(self, historical_predictions: pd.DataFrame, 
                           actual_defaults: pd.Series) -> Dict[str, float]:
        """
        Perform model back-testing and validation
        
        Args:
            historical_predictions: DataFrame with predicted PDs
            actual_defaults: Series with actual default outcomes
            
        Returns:
            Dictionary with back-testing results
        """
        
        print("🔍 Performing model back-testing...")
        
        # Hosmer-Lemeshow Test for calibration
        def hosmer_lemeshow_test(y_true, y_prob, n_bins=10):
            """Simplified Hosmer-Lemeshow goodness-of-fit test"""
            
            # Create bins based on predicted probabilities
            bin_edges = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
            bin_edges[0] = 0  # Ensure first bin starts at 0
            bin_edges[-1] = 1  # Ensure last bin ends at 1
            
            bins = pd.cut(y_prob, bins=bin_edges, include_lowest=True)
            
            # Calculate observed vs expected in each bin
            df_test = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob, 'bin': bins})
            grouped = df_test.groupby('bin')
            
            observed = grouped['y_true'].sum()
            expected = grouped['y_prob'].sum()
            total = grouped.size()
            
            # Chi-square statistic
            chi_square = ((observed - expected) ** 2 / (expected + 1e-10)).sum()
            
            return chi_square, len(observed) - 2  # degrees of freedom
        
        # Convert predictions to probabilities if needed
        if 'probability_default' in historical_predictions.columns:
            y_prob = historical_predictions['probability_default'] / 100
        else:
            raise ValueError("No probability_default column found in predictions")
        
        y_true = actual_defaults.astype(int)
        
        # Discrimination metrics
        from sklearn.metrics import roc_auc_score, average_precision_score
        
        auc_score = roc_auc_score(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)
        
        # Calibration metrics
        chi_square, df = hosmer_lemeshow_test(y_true, y_prob)
        
        # Brier Score (mean squared difference between predicted and actual)
        brier_score = np.mean((y_prob - y_true) ** 2)
        
        # Population Stability Index (PSI)
        def calculate_psi(expected, actual, bins=10):
            """Calculate Population Stability Index"""
            
            # Create bins
            bin_edges = np.percentile(expected, np.linspace(0, 100, bins + 1))
            
            # Calculate distributions
            expected_dist = pd.cut(expected, bins=bin_edges, include_lowest=True).value_counts(normalize=True)
            actual_dist = pd.cut(actual, bins=bin_edges, include_lowest=True).value_counts(normalize=True)
            
            # Calculate PSI
            psi = ((actual_dist - expected_dist) * np.log(actual_dist / (expected_dist + 1e-10))).sum()
            
            return psi
        
        # For PSI, we'd need historical data - using dummy calculation
        psi_score = 0.05  # Placeholder - would need actual historical comparison
        
        back_testing_results = {
            'auc_score': auc_score,
            'average_precision': avg_precision,
            'brier_score': brier_score,
            'hosmer_lemeshow_chi2': chi_square,
            'hosmer_lemeshow_df': df,
            'population_stability_index': psi_score,
            'calibration_status': 'Good' if chi_square < 15.507 else 'Poor',  # 95% confidence
            'discrimination_status': 'Good' if auc_score > 0.7 else 'Fair' if auc_score > 0.6 else 'Poor'
        }
        
        print(f"✅ Back-testing completed - AUC: {auc_score:.3f}, Brier: {brier_score:.3f}")
        
        return back_testing_results
    
    def calculate_economic_capital(self, portfolio_metrics: PortfolioMetrics, 
                                 confidence_level: float = 0.999) -> Dict[str, float]:
        """
        Calculate economic capital using internal models
        
        Args:
            portfolio_metrics: Portfolio metrics object
            confidence_level: Confidence level for economic capital
            
        Returns:
            Dictionary with economic capital calculations
        """
        
        print(f"📊 Calculating economic capital at {confidence_level:.1%} confidence level...")
        
        # Economic capital is typically UL at very high confidence level
        # EC = VaR(confidence_level) - Expected_Loss
        
        # For simplicity, scale the 99% VaR to the desired confidence level
        if confidence_level == 0.999:
            # 99.9% VaR is typically ~1.3x the 99% VaR for credit portfolios
            var_999 = portfolio_metrics.portfolio_var_99 * 1.3
        elif confidence_level == 0.99:
            var_999 = portfolio_metrics.portfolio_var_99
        elif confidence_level == 0.95:
            var_999 = portfolio_metrics.portfolio_var_95
        else:
            # Linear interpolation for other confidence levels
            var_999 = portfolio_metrics.portfolio_var_99 * (confidence_level / 0.99)
        
        economic_capital = var_999 - portfolio_metrics.expected_loss
        
        # Risk-adjusted return on capital (RAROC)
        # Assuming some net income from the portfolio
        assumed_net_income = portfolio_metrics.total_exposure * 0.02  # 2% assumed margin
        
        raroc = (assumed_net_income - portfolio_metrics.expected_loss) / economic_capital * 100
        
        economic_capital_metrics = {
            'economic_capital': economic_capital,
            'var_at_confidence_level': var_999,
            'confidence_level': confidence_level,
            'raroc_percent': raroc,
            'ec_ratio_to_exposure': (economic_capital / portfolio_metrics.total_exposure) * 100,
            'net_income_assumption': assumed_net_income
        }
        
        print(f"✅ Economic capital: ${economic_capital:,.0f} ({economic_capital/portfolio_metrics.total_exposure*100:.2f}% of exposure)")
        
        return economic_capital_metrics
    
    def generate_risk_report(self, portfolio_metrics: PortfolioMetrics) -> str:
        """
        Generate comprehensive risk analysis report
        
        Args:
            portfolio_metrics: Portfolio metrics object
            
        Returns:
            Formatted risk report string
        """
        
        report = "\n📊 PORTFOLIO RISK ANALYSIS REPORT\n"
        report += "=" * 60 + "\n\n"
        
        # Portfolio Overview
        report += "🔍 PORTFOLIO OVERVIEW:\n"
        report += f"   Total Exposure: ${portfolio_metrics.total_exposure:,.0f}\n"
        report += f"   Number of Loans: {portfolio_metrics.number_of_loans:,}\n"
        report += f"   Average Loan Size: ${portfolio_metrics.total_exposure/portfolio_metrics.number_of_loans:,.0f}\n"
        report += f"   Average PD: {portfolio_metrics.average_pd:.2f}%\n\n"
        
        # Risk Metrics
        report += "⚠️  RISK METRICS:\n"
        report += f"   Expected Loss: ${portfolio_metrics.expected_loss:,.0f}\n"
        report += f"   Unexpected Loss: ${portfolio_metrics.unexpected_loss:,.0f}\n"
        report += f"   Portfolio VaR (95%): ${portfolio_metrics.portfolio_var_95:,.0f}\n"
        report += f"   Portfolio VaR (99%): ${portfolio_metrics.portfolio_var_99:,.0f}\n"
        report += f"   EL as % of Exposure: {portfolio_metrics.expected_loss/portfolio_metrics.total_exposure*100:.2f}%\n\n"
        
        # Concentration Analysis
        report += "🎯 CONCENTRATION ANALYSIS:\n"
        report += f"   Overall HHI: {portfolio_metrics.concentration_hhi:.4f}\n"
        report += f"   Diversification Ratio: {portfolio_metrics.diversification_ratio:.3f}\n"
        
        # Concentration interpretation
        if portfolio_metrics.concentration_hhi < 0.01:
            conc_level = "Very Low (Well Diversified)"
        elif portfolio_metrics.concentration_hhi < 0.05:
            conc_level = "Low"
        elif portfolio_metrics.concentration_hhi < 0.15:
            conc_level = "Moderate"
        else:
            conc_level = "High (Concentrated)"
        
        report += f"   Concentration Level: {conc_level}\n\n"
        
        # Risk Grade Distribution
        if portfolio_metrics.grade_distribution:
            report += "📈 RISK GRADE DISTRIBUTION:\n"
            for grade in ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C']:
                if grade in portfolio_metrics.grade_distribution:
                    pct = portfolio_metrics.grade_distribution[grade]
                    report += f"   Grade {grade}: {pct:.1f}%\n"
            report += "\n"
        
        # Top Concentrations
        if portfolio_metrics.sector_concentration:
            report += "🏭 TOP SECTOR CONCENTRATIONS:\n"
            for sector, pct in list(portfolio_metrics.sector_concentration.items())[:5]:
                report += f"   {sector}: {pct:.1f}%\n"
            report += "\n"
        
        if portfolio_metrics.geographic_concentration:
            report += "🗺️  TOP GEOGRAPHIC CONCENTRATIONS:\n"
            for state, pct in list(portfolio_metrics.geographic_concentration.items())[:5]:
                report += f"   {state}: {pct:.1f}%\n"
            report += "\n"
        
        # Risk Assessment
        report += "🚨 RISK ASSESSMENT:\n"
        
        # Overall risk level based on multiple factors
        risk_factors = []
        
        if portfolio_metrics.average_pd > 10:
            risk_factors.append("High average PD")
        elif portfolio_metrics.average_pd > 5:
            risk_factors.append("Moderate PD levels")
        
        if portfolio_metrics.concentration_hhi > 0.15:
            risk_factors.append("High concentration risk")
        elif portfolio_metrics.concentration_hhi > 0.05:
            risk_factors.append("Moderate concentration")
        
        el_ratio = portfolio_metrics.expected_loss / portfolio_metrics.total_exposure
        if el_ratio > 0.05:
            risk_factors.append("High expected loss ratio")
        elif el_ratio > 0.02:
            risk_factors.append("Moderate expected loss ratio")
        
        if risk_factors:
            report += f"   Risk Factors Identified: {', '.join(risk_factors)}\n"
        else:
            report += "   No significant risk factors identified\n"
        
        # Overall risk rating
        if len(risk_factors) == 0:
            overall_risk = "LOW"
        elif len(risk_factors) <= 2:
            overall_risk = "MODERATE"
        else:
            overall_risk = "HIGH"
        
        report += f"   Overall Risk Rating: {overall_risk}\n\n"
        
        # Recommendations
        report += "💡 RECOMMENDATIONS:\n"
        
        if portfolio_metrics.concentration_hhi > 0.1:
            report += "   • Consider diversification to reduce concentration risk\n"
        
        if portfolio_metrics.average_pd > 8:
            report += "   • Review underwriting standards for high-risk segments\n"
        
        if el_ratio > 0.03:
            report += "   • Increase pricing or reduce exposure in high-risk categories\n"
        
        report += "   • Conduct regular stress testing under adverse scenarios\n"
        report += "   • Monitor early warning indicators for portfolio deterioration\n"
        report += "   • Ensure adequate capital reserves for unexpected losses\n\n"
        
        report += f"Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d at %H:%M:%S')}\n"
        
        return report