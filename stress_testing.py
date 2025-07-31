"""
Stress Testing Module
====================

Regulatory stress testing capabilities including:
- Economic scenario generation
- CCAR/DFAST style stress tests
- Custom adverse scenarios
- Portfolio impact analysis
- Capital planning under stress
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class StressScenario:
    """Economic stress scenario definition"""
    name: str
    description: str
    unemployment_shock: float  # Percentage point increase
    gdp_shock: float          # Percentage decline
    interest_rate_shock: float # Percentage point increase
    credit_spread_shock: float # Percentage point increase
    duration_quarters: int     # Scenario duration
    sector_impacts: Dict[str, float] = None  # Industry-specific multipliers

@dataclass
class StressResults:
    """Stress testing results"""
    scenario_name: str
    baseline_expected_loss: float
    stressed_expected_loss: float
    loss_increase: float
    loss_increase_pct: float
    capital_impact: float
    worst_affected_segments: List[Dict[str, Any]]
    portfolio_metrics_stressed: Dict[str, float]

class StressTester:
    """
    Advanced stress testing framework for credit portfolios
    """
    
    def __init__(self, economic_data_collector):
        """
        Initialize stress testing framework
        
        Args:
            economic_data_collector: Economic data source for baseline conditions
        """
        self.economic_data = economic_data_collector
        
        # Pre-defined regulatory scenarios
        self.regulatory_scenarios = {
            'severely_adverse_2024': StressScenario(
                name='Severely Adverse 2024',
                description='2024 CCAR Severely Adverse Scenario',
                unemployment_shock=4.0,    # +4 percentage points
                gdp_shock=-3.5,           # -3.5% GDP decline
                interest_rate_shock=0.5,   # +0.5 percentage points
                credit_spread_shock=2.0,   # +2 percentage points
                duration_quarters=9,
                sector_impacts={
                    'Retail': 1.5, 'Hospitality': 1.8, 'Energy': 1.4,
                    'Manufacturing': 1.3, 'Technology': 0.9, 'Healthcare': 0.8
                }
            ),
            'recession': StressScenario(
                name='Economic Recession',
                description='Moderate recession scenario',
                unemployment_shock=2.5,
                gdp_shock=-2.0,
                interest_rate_shock=0.0,   # Rates may fall in recession
                credit_spread_shock=1.5,
                duration_quarters=6,
                sector_impacts={
                    'Retail': 1.3, 'Hospitality': 1.4, 'Construction': 1.6,
                    'Manufacturing': 1.2, 'Technology': 1.0, 'Healthcare': 0.9
                }
            ),
            'interest_rate_shock': StressScenario(
                name='Interest Rate Shock',
                description='Rapid interest rate increase',
                unemployment_shock=1.0,
                gdp_shock=-1.0,
                interest_rate_shock=3.0,   # +3 percentage points
                credit_spread_shock=1.0,
                duration_quarters=4,
                sector_impacts={
                    'Real Estate': 1.4, 'Construction': 1.3, 'Utilities': 1.2,
                    'Technology': 1.1, 'Healthcare': 0.9, 'Education': 0.9
                }
            ),
            'unemployment_spike': StressScenario(
                name='Unemployment Spike',
                description='Sharp increase in unemployment',
                unemployment_shock=3.5,
                gdp_shock=-1.5,
                interest_rate_shock=-0.5,  # Rates may fall
                credit_spread_shock=1.8,
                duration_quarters=8,
                sector_impacts={
                    'Retail': 1.6, 'Hospitality': 1.7, 'Transportation': 1.4,
                    'Manufacturing': 1.3, 'Government': 0.7, 'Healthcare': 0.8
                }
            )
        }
        
        print("⚡ Stress Testing Framework initialized")
        print(f"   Available scenarios: {', '.join(self.regulatory_scenarios.keys())}")
    
    def run_stress_scenario(self, borrowers: List, scenario_name: str) -> StressResults:
        """
        Run stress test on portfolio using specified scenario
        
        Args:
            borrowers: List of Borrower objects
            scenario_name: Name of stress scenario to run
            
        Returns:
            StressResults object with detailed impact analysis
        """
        
        if scenario_name not in self.regulatory_scenarios:
            raise ValueError(f"Scenario '{scenario_name}' not found. Available: {list(self.regulatory_scenarios.keys())}")
        
        scenario = self.regulatory_scenarios[scenario_name]
        
        print(f"⚡ Running stress scenario: {scenario.name}")
        print(f"   {scenario.description}")
        
        # Calculate baseline metrics
        baseline_el = sum(getattr(b, 'expected_loss', 0) for b in borrowers)
        
        # Apply stress scenario
        stressed_borrowers = self._apply_stress_to_portfolio(borrowers, scenario)
        
        # Calculate stressed metrics
        stressed_el = sum(getattr(b, 'expected_loss', 0) for b in stressed_borrowers)
        
        # Calculate impacts
        loss_increase = stressed_el - baseline_el
        loss_increase_pct = (loss_increase / baseline_el * 100) if baseline_el > 0 else 0
        
        # Estimate capital impact (simplified)
        capital_impact = loss_increase * 1.5  # Assume 1.5x multiplier for capital
        
        # Identify worst affected segments
        worst_segments = self._identify_worst_segments(borrowers, stressed_borrowers)
        
        # Calculate stressed portfolio metrics
        stressed_metrics = self._calculate_stressed_portfolio_metrics(stressed_borrowers)
        
        results = StressResults(
            scenario_name=scenario.name,
            baseline_expected_loss=baseline_el,
            stressed_expected_loss=stressed_el,
            loss_increase=loss_increase,
            loss_increase_pct=loss_increase_pct,
            capital_impact=capital_impact,
            worst_affected_segments=worst_segments,
            portfolio_metrics_stressed=stressed_metrics
        )
        
        print(f"✅ Stress test completed")
        print(f"   Loss increase: ${loss_increase:,.0f} ({loss_increase_pct:.1f}%)")
        
        return results
    
    def _apply_stress_to_portfolio(self, borrowers: List, scenario: StressScenario) -> List:
        """
        Apply stress scenario to individual borrowers
        
        Args:
            borrowers: List of Borrower objects
            scenario: StressScenario to apply
            
        Returns:
            List of stressed Borrower objects
        """
        
        stressed_borrowers = []
        
        for borrower in borrowers:
            # Create copy of borrower for stress testing
            import copy
            stressed_borrower = copy.deepcopy(borrower)
            
            # Calculate stress multiplier based on borrower characteristics
            stress_multiplier = self._calculate_individual_stress_multiplier(
                stressed_borrower, scenario
            )
            
            # Apply stress to probability of default
            if hasattr(stressed_borrower, 'probability_default') and stressed_borrower.probability_default:
                original_pd = stressed_borrower.probability_default
                stressed_pd = min(100.0, original_pd * stress_multiplier)
                stressed_borrower.probability_default = stressed_pd
                
                # Recalculate expected loss with stressed PD
                lgd = 0.45  # Loss Given Default assumption
                ead = stressed_borrower.loan_amount  # Exposure at Default
                stressed_borrower.expected_loss = (stressed_pd / 100) * lgd * ead
            
            stressed_borrowers.append(stressed_borrower)
        
        return stressed_borrowers
    
    def _calculate_individual_stress_multiplier(self, borrower, scenario: StressScenario) -> float:
        """
        Calculate stress multiplier for individual borrower
        
        Args:
            borrower: Borrower object
            scenario: StressScenario being applied
            
        Returns:
            Stress multiplier (1.0 = no change, >1.0 = increased risk)
        """
        
        base_multiplier = 1.0
        
        # Economic stress impact based on borrower characteristics
        
        # 1. Regional unemployment impact
        if hasattr(borrower, 'local_unemployment'):
            current_unemployment = borrower.local_unemployment
            stressed_unemployment = current_unemployment + scenario.unemployment_shock
            
            # Higher baseline unemployment areas are more sensitive
            unemployment_sensitivity = 1.0 + (current_unemployment - 4.0) * 0.05
            unemployment_impact = (stressed_unemployment / current_unemployment - 1) * unemployment_sensitivity
            base_multiplier += unemployment_impact
        else:
            # National average impact
            base_multiplier += scenario.unemployment_shock * 0.15
        
        # 2. Industry-specific impact
        if hasattr(borrower, 'industry') and borrower.industry:
            if scenario.sector_impacts and borrower.industry in scenario.sector_impacts:
                industry_multiplier = scenario.sector_impacts[borrower.industry]
                base_multiplier *= industry_multiplier
            else:
                # Default industry impact
                base_multiplier *= 1.2
        
        # 3. Credit quality sensitivity
        if hasattr(borrower, 'credit_score'):
            # Lower credit scores are more sensitive to economic stress
            if borrower.credit_score < 600:
                credit_sensitivity = 1.8
            elif borrower.credit_score < 650:
                credit_sensitivity = 1.5
            elif borrower.credit_score < 700:
                credit_sensitivity = 1.3
            elif borrower.credit_score < 750:
                credit_sensitivity = 1.1
            else:
                credit_sensitivity = 1.0
            
            base_multiplier *= credit_sensitivity
        
        # 4. Debt-to-income sensitivity
        if hasattr(borrower, 'debt_to_income'):
            # Higher DTI borrowers are more sensitive
            dti_impact = 1.0 + (borrower.debt_to_income - 0.3) * 1.5
            dti_impact = max(1.0, dti_impact)  # Minimum of 1.0
            base_multiplier *= dti_impact
        
        # 5. Interest rate sensitivity (for variable rate exposures)
        if scenario.interest_rate_shock != 0:
            # Assume some borrowers have variable rate exposure
            rate_sensitivity = 1.0 + abs(scenario.interest_rate_shock) * 0.1
            base_multiplier *= rate_sensitivity
        
        # 6. Age and employment stability
        if hasattr(borrower, 'age') and hasattr(borrower, 'employment_length'):
            # Younger borrowers with short employment history are more vulnerable
            if borrower.age < 30 and borrower.employment_length < 2:
                stability_factor = 1.4
            elif borrower.age < 35 and borrower.employment_length < 5:
                stability_factor = 1.2
            else:
                stability_factor = 1.0
            
            base_multiplier *= stability_factor
        
        # Ensure reasonable bounds
        return max(1.0, min(5.0, base_multiplier))
    
    def _identify_worst_segments(self, baseline_borrowers: List, stressed_borrowers: List) -> List[Dict]:
        """
        Identify portfolio segments most affected by stress
        """
        
        # Convert to DataFrames for analysis
        baseline_df = pd.DataFrame([b.__dict__ for b in baseline_borrowers])
        stressed_df = pd.DataFrame([b.__dict__ for b in stressed_borrowers])
        
        worst_segments = []
        
        # Analyze by industry
        if 'industry' in baseline_df.columns and 'expected_loss' in baseline_df.columns:
            baseline_industry = baseline_df.groupby('industry')['expected_loss'].sum()
            stressed_industry = stressed_df.groupby('industry')['expected_loss'].sum()
            industry_impact = ((stressed_industry - baseline_industry) / baseline_industry * 100).sort_values(ascending=False)
            
            for industry, impact_pct in industry_impact.head(3).items():
                worst_segments.append({
                    'segment_type': 'Industry',
                    'segment_name': industry,
                    'impact_percent': impact_pct,
                    'baseline_loss': baseline_industry[industry],
                    'stressed_loss': stressed_industry[industry]
                })
        
        # Analyze by state
        if 'state' in baseline_df.columns:
            baseline_state = baseline_df.groupby('state')['expected_loss'].sum()
            stressed_state = stressed_df.groupby('state')['expected_loss'].sum()
            state_impact = ((stressed_state - baseline_state) / baseline_state * 100).sort_values(ascending=False)
            
            for state, impact_pct in state_impact.head(2).items():
                worst_segments.append({
                    'segment_type': 'Geographic',
                    'segment_name': state,
                    'impact_percent': impact_pct,
                    'baseline_loss': baseline_state[state],
                    'stressed_loss': stressed_state[state]
                })
        
        # Analyze by credit score bands
        if 'credit_score' in baseline_df.columns:
            baseline_df['credit_band'] = pd.cut(baseline_df['credit_score'], 
                                              bins=[0, 600, 650, 700, 750, 850],
                                              labels=['<600', '600-649', '650-699', '700-749', '750+'])
            stressed_df['credit_band'] = pd.cut(stressed_df['credit_score'], 
                                              bins=[0, 600, 650, 700, 750, 850],
                                              labels=['<600', '600-649', '650-699', '700-749', '750+'])
            
            baseline_credit = baseline_df.groupby('credit_band')['expected_loss'].sum()
            stressed_credit = stressed_df.groupby('credit_band')['expected_loss'].sum()
            credit_impact = ((stressed_credit - baseline_credit) / baseline_credit * 100).sort_values(ascending=False)
            
            for band, impact_pct in credit_impact.head(2).items():
                worst_segments.append({
                    'segment_type': 'Credit Score',
                    'segment_name': str(band),
                    'impact_percent': impact_pct,
                    'baseline_loss': baseline_credit[band],
                    'stressed_loss': stressed_credit[band]
                })
        
        return sorted(worst_segments, key=lambda x: x['impact_percent'], reverse=True)
    
    def _calculate_stressed_portfolio_metrics(self, stressed_borrowers: List) -> Dict[str, float]:
        """
        Calculate key portfolio metrics under stress
        """
        
        if not stressed_borrowers:
            return {}
        
        df = pd.DataFrame([b.__dict__ for b in stressed_borrowers])
        
        metrics = {}
        
        # Basic metrics
        metrics['total_exposure'] = df['loan_amount'].sum()
        metrics['number_of_loans'] = len(stressed_borrowers)
        
        if 'probability_default' in df.columns:
            metrics['average_pd'] = df['probability_default'].mean()
            
            # Calculate stressed VaR (simplified)
            loan_amounts = df['loan_amount'].values
            default_probs = df['probability_default'].values / 100
            
            # Simple VaR calculation
            losses = loan_amounts * default_probs * 0.45  # Assuming 45% LGD
            portfolio_loss = losses.sum()
            
            # Add correlation for portfolio effect (simplified)
            correlation_adjustment = 1.2  # Assume some correlation in stress
            metrics['stressed_portfolio_loss'] = portfolio_loss * correlation_adjustment
        
        if 'expected_loss' in df.columns:
            metrics['total_expected_loss'] = df['expected_loss'].sum()
            metrics['el_as_percent_exposure'] = (metrics['total_expected_loss'] / metrics['total_exposure']) * 100
        
        # Risk grade distribution under stress
        if 'risk_grade' in df.columns:
            grade_dist = df['risk_grade'].value_counts(normalize=True) * 100
            for grade, pct in grade_dist.items():
                metrics[f'grade_{grade}_percent'] = pct
        
        return metrics
    
    def run_comprehensive_stress_test(self, borrowers: List, 
                                    scenarios: List[str] = None) -> Dict[str, StressResults]:
        """
        Run multiple stress scenarios for comprehensive analysis
        
        Args:
            borrowers: List of Borrower objects
            scenarios: List of scenario names to run (if None, runs all)
            
        Returns:
            Dictionary mapping scenario names to StressResults
        """
        
        if scenarios is None:
            scenarios = list(self.regulatory_scenarios.keys())
        
        print(f"⚡ Running comprehensive stress testing with {len(scenarios)} scenarios...")
        
        results = {}
        
        for scenario_name in scenarios:
            print(f"\n   🧪 Testing scenario: {scenario_name}")
            try:
                result = self.run_stress_scenario(borrowers, scenario_name)
                results[scenario_name] = result
            except Exception as e:
                print(f"   ❌ Error in scenario {scenario_name}: {e}")
                continue
        
        print(f"\n✅ Comprehensive stress testing completed")
        self._print_stress_summary(results)
        
        return results
    
    def _print_stress_summary(self, results: Dict[str, StressResults]):
        """Print summary of stress test results"""
        
        print(f"\n📊 STRESS TEST SUMMARY")
        print("=" * 50)
        
        for scenario_name, result in results.items():
            print(f"\n🧪 {scenario_name.upper()}:")
            print(f"   Baseline Expected Loss: ${result.baseline_expected_loss:,.0f}")
            print(f"   Stressed Expected Loss: ${results.stressed_expected_loss:,.0f}")
            print(f"   Loss Increase: {results.loss_increase_pct:.1f}%")
            print(f"   Capital Impact: ${results.capital_impact:,.0f}")
            
            if result.worst_affected_segments:
                print(f"   Worst Affected Segment: {result.worst_affected_segments[0]['segment_name']} "
                      f"({result.worst_affected_segments[0]['impact_percent']:.1f}% increase)")
    
    def create_custom_scenario(self, name: str, description: str,
                              unemployment_shock: float = 0,
                              gdp_shock: float = 0,
                              interest_rate_shock: float = 0,
                              credit_spread_shock: float = 0,
                              duration_quarters: int = 4,
                              sector_impacts: Dict[str, float] = None) -> str:
        """
        Create custom stress scenario
        
        Args:
            name: Scenario name
            description: Scenario description
            unemployment_shock: Unemployment rate increase (percentage points)
            gdp_shock: GDP decline (percentage)
            interest_rate_shock: Interest rate increase (percentage points)
            credit_spread_shock: Credit spread increase (percentage points)
            duration_quarters: Duration in quarters
            sector_impacts: Industry-specific impact multipliers
            
        Returns:
            Scenario name for reference
        """
        
        custom_scenario = StressScenario(
            name=name,
            description=description,
            unemployment_shock=unemployment_shock,
            gdp_shock=gdp_shock,
            interest_rate_shock=interest_rate_shock,
            credit_spread_shock=credit_spread_shock,
            duration_quarters=duration_quarters,
            sector_impacts=sector_impacts or {}
        )
        
        self.regulatory_scenarios[name] = custom_scenario
        
        print(f"✅ Created custom scenario: {name}")
        return name
    
    def calculate_capital_adequacy_under_stress(self, baseline_capital: float,
                                              stress_results: Dict[str, StressResults]) -> Dict[str, Dict]:
        """
        Calculate capital adequacy under various stress scenarios
        
        Args:
            baseline_capital: Current capital level
            stress_results: Results from stress testing
            
        Returns:
            Dictionary with capital adequacy analysis
        """
        
        print("💰 Calculating capital adequacy under stress...")
        
        capital_analysis = {}
        
        # Regulatory minimum capital ratios
        minimum_ratios = {
            'cet1_minimum': 0.045,      # 4.5% CET1
            'tier1_minimum': 0.06,      # 6% Tier 1
            'total_capital_minimum': 0.08,  # 8% Total Capital
            'capital_buffer': 0.025     # 2.5% Conservation Buffer
        }
        
        for scenario_name, result in stress_results.items():
            scenario_analysis = {}
            
            # Calculate post-stress capital
            capital_depletion = result.capital_impact
            post_stress_capital = baseline_capital - capital_depletion
            
            # Assume total exposure as proxy for risk-weighted assets
            rwa_proxy = result.baseline_expected_loss / 0.02  # Assume 2% baseline loss rate
            
            # Calculate capital ratios
            if rwa_proxy > 0:
                post_stress_capital_ratio = post_stress_capital / rwa_proxy
                
                scenario_analysis = {
                    'scenario_name': scenario_name,
                    'baseline_capital': baseline_capital,
                    'capital_depletion': capital_depletion,
                    'post_stress_capital': post_stress_capital,
                    'post_stress_capital_ratio': post_stress_capital_ratio,
                    'meets_minimum_requirement': post_stress_capital_ratio >= minimum_ratios['total_capital_minimum'],
                    'meets_buffer_requirement': post_stress_capital_ratio >= (minimum_ratios['total_capital_minimum'] + minimum_ratios['capital_buffer']),
                    'capital_shortfall': max(0, (minimum_ratios['total_capital_minimum'] + minimum_ratios['capital_buffer']) * rwa_proxy - post_stress_capital),
                    'loss_absorption_capacity': post_stress_capital / baseline_capital if baseline_capital > 0 else 0
                }
            else:
                scenario_analysis = {
                    'scenario_name': scenario_name,
                    'error': 'Unable to calculate ratios - insufficient data'
                }
            
            capital_analysis[scenario_name] = scenario_analysis
        
        print(f"✅ Capital adequacy analysis completed for {len(stress_results)} scenarios")
        
        return capital_analysis
    
    def generate_stress_test_report(self, stress_results: Dict[str, StressResults],
                                   capital_analysis: Dict[str, Dict] = None) -> str:
        """
        Generate comprehensive stress test report
        
        Args:
            stress_results: Results from stress testing
            capital_analysis: Capital adequacy analysis results
            
        Returns:
            Formatted stress test report
        """
        
        report = "\n⚡ COMPREHENSIVE STRESS TEST REPORT\n"
        report += "=" * 60 + "\n"
        
        report += f"\nReport Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"Scenarios Tested: {len(stress_results)}\n\n"
        
        # Executive Summary
        report += "📋 EXECUTIVE SUMMARY\n"
        report += "-" * 30 + "\n"
        
        if stress_results:
            worst_scenario = max(stress_results.values(), key=lambda x: x.loss_increase_pct)
            best_scenario = min(stress_results.values(), key=lambda x: x.loss_increase_pct)
            
            report += f"Most Severe Scenario: {worst_scenario.scenario_name}\n"
            report += f"   Loss Increase: {worst_scenario.loss_increase_pct:.1f}%\n"
            report += f"   Capital Impact: ${worst_scenario.capital_impact:,.0f}\n\n"
            
            report += f"Least Severe Scenario: {best_scenario.scenario_name}\n"
            report += f"   Loss Increase: {best_scenario.loss_increase_pct:.1f}%\n"
            report += f"   Capital Impact: ${best_scenario.capital_impact:,.0f}\n\n"
        
        # Detailed Results by Scenario
        report += "📊 DETAILED SCENARIO RESULTS\n"
        report += "-" * 35 + "\n"
        
        for scenario_name, result in stress_results.items():
            report += f"\n🧪 {result.scenario_name}:\n"
            report += f"   Baseline Expected Loss: ${result.baseline_expected_loss:,.0f}\n"
            report += f"   Stressed Expected Loss: ${result.stressed_expected_loss:,.0f}\n"
            report += f"   Absolute Increase: ${result.loss_increase:,.0f}\n"
            report += f"   Percentage Increase: {result.loss_increase_pct:.1f}%\n"
            report += f"   Estimated Capital Impact: ${result.capital_impact:,.0f}\n"
            
            # Worst affected segments
            if result.worst_affected_segments:
                report += f"   Most Affected Segments:\n"
                for i, segment in enumerate(result.worst_affected_segments[:3], 1):
                    report += f"      {i}. {segment['segment_type']}: {segment['segment_name']} "
                    report += f"({segment['impact_percent']:.1f}% increase)\n"
            
            report += "\n"
        
        # Capital Adequacy Analysis
        if capital_analysis:
            report += "💰 CAPITAL ADEQUACY UNDER STRESS\n"
            report += "-" * 35 + "\n"
            
            for scenario_name, analysis in capital_analysis.items():
                if 'error' not in analysis:
                    report += f"\n{scenario_name}:\n"
                    report += f"   Post-Stress Capital: ${analysis['post_stress_capital']:,.0f}\n"
                    report += f"   Capital Ratio: {analysis['post_stress_capital_ratio']:.2%}\n"
                    report += f"   Meets Minimum Requirement: {'Yes' if analysis['meets_minimum_requirement'] else 'No'}\n"
                    report += f"   Meets Buffer Requirement: {'Yes' if analysis['meets_buffer_requirement'] else 'No'}\n"
                    
                    if analysis['capital_shortfall'] > 0:
                        report += f"   Capital Shortfall: ${analysis['capital_shortfall']:,.0f}\n"
        
        # Risk Management Recommendations
        report += "\n💡 RISK MANAGEMENT RECOMMENDATIONS\n"
        report += "-" * 40 + "\n"
        
        # Generate recommendations based on results
        if stress_results:
            max_loss_increase = max(result.loss_increase_pct for result in stress_results.values())
            
            if max_loss_increase > 50:
                report += "🚨 HIGH RISK PORTFOLIO:\n"
                report += "   • Immediate review of underwriting standards required\n"
                report += "   • Consider reducing exposure in high-risk segments\n"
                report += "   • Increase capital reserves significantly\n"
                report += "   • Implement enhanced monitoring procedures\n\n"
            elif max_loss_increase > 25:
                report += "⚠️  MODERATE RISK PORTFOLIO:\n"
                report += "   • Review pricing for high-risk segments\n"
                report += "   • Strengthen risk monitoring capabilities\n"
                report += "   • Consider portfolio diversification strategies\n"
                report += "   • Ensure adequate capital planning\n\n"
            else:
                report += "✅ RESILIENT PORTFOLIO:\n"
                report += "   • Portfolio shows good stress resilience\n"
                report += "   • Continue current risk management practices\n"
                report += "   • Monitor for early warning indicators\n"
                report += "   • Regular stress testing recommended\n\n"
        
        report += "📝 GENERAL RECOMMENDATIONS:\n"
        report += "   • Conduct stress testing quarterly\n"
        report += "   • Update scenarios based on evolving economic conditions\n"
        report += "   • Integrate stress testing into capital planning process\n"
        report += "   • Develop contingency plans for severe stress scenarios\n"
        report += "   • Enhance data quality for more accurate stress testing\n\n"
        
        report += f"Report generated by BankRisk Pro Stress Testing Framework\n"
        report += f"For questions about methodology, please contact the Risk Management team.\n"
        
        return report
    
    def export_stress_results(self, stress_results: Dict[str, StressResults], 
                            filename: str = None) -> str:
        """
        Export stress test results to CSV
        
        Args:
            stress_results: Stress test results
            filename: Optional filename for export
            
        Returns:
            Filename of exported results
        """
        
        if not filename:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            filename = f"outputs/stress_test_results_{timestamp}.csv"
        
        # Prepare data for export
        export_data = []
        
        for scenario_name, result in stress_results.items():
            row = {
                'scenario_name': result.scenario_name,
                'baseline_expected_loss': result.baseline_expected_loss,
                'stressed_expected_loss': result.stressed_expected_loss,
                'loss_increase_dollars': result.loss_increase,
                'loss_increase_percent': result.loss_increase_pct,
                'capital_impact': result.capital_impact,
                'export_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Add worst affected segments
            if result.worst_affected_segments:
                for i, segment in enumerate(result.worst_affected_segments[:3], 1):
                    row[f'worst_segment_{i}_type'] = segment.get('segment_type', '')
                    row[f'worst_segment_{i}_name'] = segment.get('segment_name', '')
                    row[f'worst_segment_{i}_impact_pct'] = segment.get('impact_percent', 0)
            
            export_data.append(row)
        
        # Create DataFrame and export
        df = pd.DataFrame(export_data)
        df.to_csv(filename, index=False)
        
        print(f"💾 Stress test results exported to: {filename}")
        return filename