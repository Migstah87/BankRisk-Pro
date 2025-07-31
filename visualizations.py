"""
Risk Dashboard Visualizations
============================

Interactive dashboards and visualizations for credit risk analysis including:
- Executive risk dashboards
- Portfolio composition charts
- Risk distribution plots
- Stress testing visualizations
- Regulatory reporting charts
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class RiskDashboard:
    """
    Interactive risk management dashboard creation
    """
    
    def __init__(self):
        """Initialize dashboard creator"""
        
        # Color schemes for consistent visualization
        self.risk_colors = {
            'AAA': '#2E8B57', 'AA': '#32CD32', 'A': '#90EE90',
            'BBB': '#FFD700', 'BB': '#FFA500', 'B': '#FF6347',
            'CCC': '#DC143C', 'CC': '#8B0000', 'C': '#4B0000'
        }
        
        self.status_colors = {
            'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545',
            'Good': '#28a745', 'Fair': '#ffc107', 'Poor': '#dc3545'
        }
        
        print("📊 Risk Dashboard Visualizer initialized")
    
    def create_executive_dashboard(self, dashboard_data: Dict[str, Any], 
                                 save_path: Optional[str] = None) -> str:
        """
        Create comprehensive executive risk dashboard
        
        Args:
            dashboard_data: Dictionary containing all dashboard data
            save_path: Optional path to save HTML dashboard
            
        Returns:
            HTML content or filename if saved
        """
        
        print("📊 Creating executive risk dashboard...")
        
        # Extract data components
        portfolio = dashboard_data.get('portfolio', [])
        metrics = dashboard_data.get('metrics')
        economic_snapshot = dashboard_data.get('economic_snapshot')
        model_performance = dashboard_data.get('model_performance', {})
        stress_results = dashboard_data.get('stress_results', {})
        
        if not portfolio:
            print("❌ No portfolio data available for dashboard")
            return ""
        
        # Create main dashboard with subplots
        fig = make_subplots(
            rows=4, cols=3,
            subplot_titles=[
                'Portfolio Risk Distribution', 'Expected Loss by Grade', 'Geographic Concentration',
                'Industry Concentration', 'Credit Score Distribution', 'Default Probability Histogram',
                'Economic Indicators', 'Model Performance', 'Stress Test Results',
                'Portfolio Metrics Summary', 'Risk vs Return Analysis', 'Capital Adequacy'
            ],
            specs=[
                [{"type": "pie"}, {"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "histogram"}, {"type": "histogram"}],
                [{"type": "indicator"}, {"type": "bar"}, {"type": "bar"}],
                [{"type": "table"}, {"type": "scatter"}, {"type": "indicator"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.05
        )
        
        # Convert portfolio to DataFrame
        df = pd.DataFrame([borrower.__dict__ for borrower in portfolio])
        
        # 1. Portfolio Risk Distribution (Pie Chart)
        if 'risk_grade' in df.columns and not df['risk_grade'].isna().all():
            risk_counts = df['risk_grade'].value_counts()
            colors = [self.risk_colors.get(grade, '#888888') for grade in risk_counts.index]
            
            fig.add_trace(
                go.Pie(
                    labels=risk_counts.index,
                    values=risk_counts.values,
                    marker=dict(colors=colors),
                    name="Risk Grades"
                ),
                row=1, col=1
            )
        
        # 2. Expected Loss by Grade
        if 'risk_grade' in df.columns and 'expected_loss' in df.columns:
            grade_loss = df.groupby('risk_grade')['expected_loss'].sum().sort_index()
            colors = [self.risk_colors.get(grade, '#888888') for grade in grade_loss.index]
            
            fig.add_trace(
                go.Bar(
                    x=grade_loss.index,
                    y=grade_loss.values,
                    marker=dict(color=colors),
                    name="Expected Loss by Grade"
                ),
                row=1, col=2
            )
        
        # 3. Geographic Concentration
        if 'state' in df.columns:
            state_exposure = df.groupby('state')['loan_amount'].sum().sort_values(ascending=False).head(10)
            
            fig.add_trace(
                go.Bar(
                    x=state_exposure.values,
                    y=state_exposure.index,
                    orientation='h',
                    marker=dict(color='lightblue'),
                    name="Geographic Exposure"
                ),
                row=1, col=3
            )
        
        # 4. Industry Concentration
        if 'industry' in df.columns:
            industry_exposure = df.groupby('industry')['loan_amount'].sum().sort_values(ascending=False).head(8)
            
            fig.add_trace(
                go.Bar(
                    x=industry_exposure.index,
                    y=industry_exposure.values,
                    marker=dict(color='lightgreen'),
                    name="Industry Exposure"
                ),
                row=2, col=1
            )
        
        # 5. Credit Score Distribution
        if 'credit_score' in df.columns:
            fig.add_trace(
                go.Histogram(
                    x=df['credit_score'],
                    nbinsx=20,
                    marker=dict(color='orange', opacity=0.7),
                    name="Credit Score Distribution"
                ),
                row=2, col=2
            )
        
        # 6. Default Probability Distribution
        if 'probability_default' in df.columns:
            fig.add_trace(
                go.Histogram(
                    x=df['probability_default'],
                    nbinsx=25,
                    marker=dict(color='red', opacity=0.7),
                    name="Default Probability Distribution"
                ),
                row=2, col=3
            )
        
        # 7. Economic Indicators (Gauge)
        if economic_snapshot:
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=economic_snapshot.unemployment_rate,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Unemployment Rate (%)"},
                    gauge={
                        'axis': {'range': [None, 10]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 4], 'color': "lightgray"},
                            {'range': [4, 7], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 8
                        }
                    }
                ),
                row=3, col=1
            )
        
        # 8. Model Performance
        if model_performance:
            model_names = list(model_performance.keys())
            auc_scores = [metrics.get('auc_score', 0) for metrics in model_performance.values()]
            
            fig.add_trace(
                go.Bar(
                    x=model_names,
                    y=auc_scores,
                    marker=dict(color=['green' if score > 0.8 else 'orange' if score > 0.7 else 'red' 
                                     for score in auc_scores]),
                    name="Model AUC Scores"
                ),
                row=3, col=2
            )
        
        # 9. Stress Test Results
        if stress_results:
            scenarios = list(stress_results.keys())
            loss_increases = [result.loss_increase_pct for result in stress_results.values()]
            
            fig.add_trace(
                go.Bar(
                    x=scenarios,
                    y=loss_increases,
                    marker=dict(color=['red' if inc > 30 else 'orange' if inc > 15 else 'green' 
                                     for inc in loss_increases]),
                    name="Stress Test Loss Increase (%)"
                ),
                row=3, col=3
            )
        
        # 10. Portfolio Metrics Summary (Table)
        if metrics:
            metrics_table = [
                ['Total Exposure', f'${metrics.total_exposure:,.0f}'],
                ['Number of Loans', f'{metrics.number_of_loans:,}'],
                ['Average PD', f'{metrics.average_pd:.2f}%'],
                ['Expected Loss', f'${metrics.expected_loss:,.0f}'],
                ['Portfolio VaR 95%', f'${metrics.portfolio_var_95:,.0f}'],
                ['Concentration HHI', f'{metrics.concentration_hhi:.3f}']
            ]
            
            fig.add_trace(
                go.Table(
                    header=dict(values=['Metric', 'Value'],
                              fill_color='paleturquoise',
                              align='left'),
                    cells=dict(values=list(zip(*metrics_table)),
                             fill_color='lavender',
                             align='left')
                ),
                row=4, col=1
            )
        
        # 11. Risk vs Return Analysis
        if 'probability_default' in df.columns and 'interest_rate' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['probability_default'],
                    y=df['interest_rate'],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=df['loan_amount'],
                        colorscale='Viridis',
                        showscale=False,
                        opacity=0.6
                    ),
                    name="Risk vs Return"
                ),
                row=4, col=2
            )
        
        # 12. Capital Adequacy Indicator
        if metrics:
            capital_ratio = 12.5  # Example capital ratio
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=capital_ratio,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Capital Ratio (%)"},
                    gauge={
                        'axis': {'range': [None, 20]},
                        'bar': {'color': "darkgreen" if capital_ratio > 10.5 else "orange" if capital_ratio > 8 else "red"},
                        'steps': [
                            {'range': [0, 8], 'color': "lightgray"},
                            {'range': [8, 10.5], 'color': "yellow"},
                            {'range': [10.5, 20], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 8
                        }
                    }
                ),
                row=4, col=3
            )
        
        # Update layout
        fig.update_layout(
            title={
                'text': '🏦 BankRisk Pro - Executive Risk Dashboard',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'color': 'darkblue'}
            },
            height=1200,
            showlegend=False,
            template="plotly_white"
        )
        
        # Save or return
        if save_path:
            fig.write_html(save_path)
            print(f"✅ Executive dashboard saved to: {save_path}")
            return save_path
        else:
            html_content = fig.to_html()
            print("✅ Executive dashboard created")
            return html_content
    
    def create_risk_distribution_chart(self, portfolio: List, save_path: Optional[str] = None) -> str:
        """Create detailed risk distribution visualization"""
        
        df = pd.DataFrame([borrower.__dict__ for borrower in portfolio])
        
        # Create subplots for risk analysis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Risk Grade Distribution', 'PD vs Credit Score', 
                          'Expected Loss Distribution', 'Risk Concentration by Industry'],
            specs=[[{"type": "pie"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "bar"}]]
        )
        
        # Risk Grade Distribution
        if 'risk_grade' in df.columns:
            grade_counts = df['risk_grade'].value_counts()
            colors = [self.risk_colors.get(grade, '#888888') for grade in grade_counts.index]
            
            fig.add_trace(
                go.Pie(
                    labels=grade_counts.index,
                    values=grade_counts.values,
                    marker=dict(colors=colors),
                    hole=0.3
                ),
                row=1, col=1
            )
        
        # PD vs Credit Score scatter
        if 'probability_default' in df.columns and 'credit_score' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['credit_score'],
                    y=df['probability_default'],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=df['probability_default'],
                        colorscale='Reds',
                        opacity=0.6
                    )
                ),
                row=1, col=2
            )
        
        # Expected Loss Distribution
        if 'expected_loss' in df.columns:
            fig.add_trace(
                go.Histogram(
                    x=df['expected_loss'],
                    nbinsx=30,
                    marker=dict(color='lightcoral', opacity=0.7)
                ),
                row=2, col=1
            )
        
        # Risk by Industry
        if 'industry' in df.columns and 'probability_default' in df.columns:
            industry_risk = df.groupby('industry')['probability_default'].mean().sort_values(ascending=False)
            
            fig.add_trace(
                go.Bar(
                    x=industry_risk.index,
                    y=industry_risk.values,
                    marker=dict(color='lightblue')
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title='Risk Distribution Analysis',
            height=800,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"✅ Risk distribution chart saved to: {save_path}")
            return save_path
        else:
            return fig.to_html()
    
    def create_stress_test_visualization(self, stress_results: Dict[str, Any], 
                                       save_path: Optional[str] = None) -> str:
        """Create stress testing visualization dashboard"""
        
        if not stress_results:
            print("❌ No stress test results to visualize")
            return ""
        
        # Create subplots for stress test analysis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Loss Increase by Scenario', 'Capital Impact Analysis',
                          'Most Affected Segments', 'Scenario Comparison'],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Extract data from stress results
        scenarios = []
        loss_increases = []
        capital_impacts = []
        baseline_losses = []
        stressed_losses = []
        
        for scenario_name, result in stress_results.items():
            scenarios.append(scenario_name)
            loss_increases.append(result.loss_increase_pct)
            capital_impacts.append(result.capital_impact)
            baseline_losses.append(result.baseline_expected_loss)
            stressed_losses.append(result.stressed_expected_loss)
        
        # 1. Loss Increase by Scenario
        colors = ['red' if inc > 30 else 'orange' if inc > 15 else 'green' for inc in loss_increases]
        fig.add_trace(
            go.Bar(
                x=scenarios,
                y=loss_increases,
                marker=dict(color=colors),
                name="Loss Increase %"
            ),
            row=1, col=1
        )
        
        # 2. Capital Impact Analysis
        fig.add_trace(
            go.Bar(
                x=scenarios,
                y=capital_impacts,
                marker=dict(color='darkred'),
                name="Capital Impact"
            ),
            row=1, col=2
        )
        
        # 3. Most Affected Segments (from first scenario as example)
        first_result = list(stress_results.values())[0]
        if first_result.worst_affected_segments:
            segments = []
            impacts = []
            for segment in first_result.worst_affected_segments[:6]:
                segments.append(f"{segment['segment_name']}")
                impacts.append(segment['impact_percent'])
            
            fig.add_trace(
                go.Bar(
                    x=segments,
                    y=impacts,
                    marker=dict(color='coral'),
                    name="Segment Impact %"
                ),
                row=2, col=1
            )
        
        # 4. Baseline vs Stressed Comparison
        fig.add_trace(
            go.Scatter(
                x=baseline_losses,
                y=stressed_losses,
                mode='markers+text',
                text=scenarios,
                textposition="top center",
                marker=dict(size=10, color='purple'),
                name="Baseline vs Stressed"
            ),
            row=2, col=2
        )
        
        # Add diagonal line for reference
        max_loss = max(max(baseline_losses), max(stressed_losses))
        fig.add_trace(
            go.Scatter(
                x=[0, max_loss],
                y=[0, max_loss],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name="No Change Line"
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Stress Testing Analysis Dashboard',
            height=800,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"✅ Stress test visualization saved to: {save_path}")
            return save_path
        else:
            return fig.to_html()
    
    def create_portfolio_composition_chart(self, portfolio: List, 
                                         save_path: Optional[str] = None) -> str:
        """Create portfolio composition analysis"""
        
        df = pd.DataFrame([borrower.__dict__ for borrower in portfolio])
        
        # Create comprehensive composition dashboard
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=['Loan Amount Distribution', 'Age Demographics',
                          'Income Distribution', 'Employment Length',
                          'Geographic Distribution', 'Industry Composition'],
            specs=[[{"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "pie"}, {"type": "pie"}]]
        )
        
        # 1. Loan Amount Distribution
        fig.add_trace(
            go.Histogram(
                x=df['loan_amount'],
                nbinsx=25,
                marker=dict(color='lightblue', opacity=0.7),
                name="Loan Amount"
            ),
            row=1, col=1
        )
        
        # 2. Age Demographics
        fig.add_trace(
            go.Histogram(
                x=df['age'],
                nbinsx=20,
                marker=dict(color='lightgreen', opacity=0.7),
                name="Age"
            ),
            row=1, col=2
        )
        
        # 3. Income Distribution
        fig.add_trace(
            go.Histogram(
                x=df['income'],
                nbinsx=25,
                marker=dict(color='lightyellow', opacity=0.7),
                name="Income"
            ),
            row=2, col=1
        )
        
        # 4. Employment Length
        fig.add_trace(
            go.Histogram(
                x=df['employment_length'],
                nbinsx=15,
                marker=dict(color='lightcoral', opacity=0.7),
                name="Employment Length"
            ),
            row=2, col=2
        )
        
        # 5. Geographic Distribution (Top 10 states)
        if 'state' in df.columns:
            state_counts = df['state'].value_counts().head(10)
            fig.add_trace(
                go.Pie(
                    labels=state_counts.index,
                    values=state_counts.values,
                    hole=0.3
                ),
                row=3, col=1
            )
        
        # 6. Industry Composition
        if 'industry' in df.columns:
            industry_counts = df['industry'].value_counts().head(8)
            fig.add_trace(
                go.Pie(
                    labels=industry_counts.index,
                    values=industry_counts.values,
                    hole=0.3
                ),
                row=3, col=2
            )
        
        fig.update_layout(
            title='Portfolio Composition Analysis',
            height=1000,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"✅ Portfolio composition chart saved to: {save_path}")
            return save_path
        else:
            return fig.to_html()
    
    def create_model_performance_dashboard(self, model_performance: Dict[str, Any], 
                                         save_path: Optional[str] = None) -> str:
        """Create model performance analysis dashboard"""
        
        if not model_performance:
            print("❌ No model performance data to visualize")
            return ""
        
        # Create model performance visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Model AUC Scores', 'Cross-Validation Performance',
                          'Feature Importance (Random Forest)', 'Model Comparison'],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "table"}]]
        )
        
        # Extract model data
        model_names = list(model_performance.keys())
        auc_scores = [metrics.get('auc_score', 0) for metrics in model_performance.values()]
        cv_means = [metrics.get('cv_mean', 0) for metrics in model_performance.values()]
        cv_stds = [metrics.get('cv_std', 0) for metrics in model_performance.values()]
        
        # 1. Model AUC Scores
        colors = ['green' if score > 0.8 else 'orange' if score > 0.7 else 'red' for score in auc_scores]
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=auc_scores,
                marker=dict(color=colors),
                name="AUC Scores"
            ),
            row=1, col=1
        )
        
        # 2. Cross-Validation Performance
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=cv_means,
                error_y=dict(type='data', array=cv_stds),
                marker=dict(color='lightblue'),
                name="CV Performance"
            ),
            row=1, col=2
        )
        
        # 3. Feature Importance (Random Forest)
        rf_performance = model_performance.get('random_forest', {})
        feature_importance = rf_performance.get('feature_importance', {})
        
        if feature_importance:
            top_features = dict(list(feature_importance.items())[:10])
            fig.add_trace(
                go.Bar(
                    x=list(top_features.values()),
                    y=list(top_features.keys()),
                    orientation='h',
                    marker=dict(color='lightgreen'),
                    name="Feature Importance"
                ),
                row=2, col=1
            )
        
        # 4. Model Comparison Table
        comparison_data = []
        for model_name, metrics in model_performance.items():
            comparison_data.append([
                model_name,
                f"{metrics.get('auc_score', 0):.3f}",
                f"{metrics.get('cv_mean', 0):.3f}",
                f"{metrics.get('cv_std', 0):.3f}",
                metrics.get('description', 'N/A')
            ])
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Model', 'AUC', 'CV Mean', 'CV Std', 'Description'],
                          fill_color='paleturquoise',
                          align='left'),
                cells=dict(values=list(zip(*comparison_data)),
                         fill_color='lavender',
                         align='left')
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Credit Risk Model Performance Dashboard',
            height=800,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"✅ Model performance dashboard saved to: {save_path}")
            return save_path
        else:
            return fig.to_html()
    
    def create_economic_indicators_chart(self, economic_snapshot, 
                                       save_path: Optional[str] = None) -> str:
        """Create economic indicators visualization"""
        
        if not economic_snapshot:
            print("❌ No economic data to visualize")
            return ""
        
        # Create economic indicators dashboard
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=['Interest Rates', 'Employment', 'Market Volatility',
                          'Economic Growth', 'Credit Conditions', 'Inflation'],
            specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]]
        )
        
        # 1. Federal Funds Rate
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=economic_snapshot.federal_funds_rate,
                title={'text': "Fed Funds Rate (%)"},
                gauge={
                    'axis': {'range': [None, 8]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 2], 'color': "lightgray"},
                        {'range': [2, 5], 'color': "yellow"},
                        {'range': [5, 8], 'color': "red"}
                    ]
                }
            ),
            row=1, col=1
        )
        
        # 2. Unemployment Rate
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=economic_snapshot.unemployment_rate,
                title={'text': "Unemployment (%)"},
                gauge={
                    'axis': {'range': [None, 12]},
                    'bar': {'color': "green" if economic_snapshot.unemployment_rate < 5 else "orange" if economic_snapshot.unemployment_rate < 8 else "red"},
                    'steps': [
                        {'range': [0, 5], 'color': "lightgreen"},
                        {'range': [5, 8], 'color': "yellow"},
                        {'range': [8, 12], 'color': "lightcoral"}
                    ]
                }
            ),
            row=1, col=2
        )
        
        # 3. VIX Volatility
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=economic_snapshot.vix_volatility,
                title={'text': "Market Volatility (VIX)"},
                gauge={
                    'axis': {'range': [None, 50]},
                    'bar': {'color': "green" if economic_snapshot.vix_volatility < 20 else "orange" if economic_snapshot.vix_volatility < 30 else "red"},
                    'steps': [
                        {'range': [0, 20], 'color': "lightgreen"},
                        {'range': [20, 30], 'color': "yellow"},
                        {'range': [30, 50], 'color': "lightcoral"}
                    ]
                }
            ),
            row=1, col=3
        )
        
        # 4. GDP Growth
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=economic_snapshot.gdp_growth,
                title={'text': "GDP Growth (%)"},
                gauge={
                    'axis': {'range': [-3, 6]},
                    'bar': {'color': "green" if economic_snapshot.gdp_growth > 2 else "orange" if economic_snapshot.gdp_growth > 0 else "red"},
                    'steps': [
                        {'range': [-3, 0], 'color': "lightcoral"},
                        {'range': [0, 2], 'color': "yellow"},
                        {'range': [2, 6], 'color': "lightgreen"}
                    ]
                }
            ),
            row=2, col=1
        )
        
        # 5. Credit Spread
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=economic_snapshot.credit_spread,
                title={'text': "Credit Spread (%)"},
                gauge={
                    'axis': {'range': [None, 8]},
                    'bar': {'color': "green" if economic_snapshot.credit_spread < 3 else "orange" if economic_snapshot.credit_spread < 5 else "red"},
                    'steps': [
                        {'range': [0, 3], 'color': "lightgreen"},
                        {'range': [3, 5], 'color': "yellow"},
                        {'range': [5, 8], 'color': "lightcoral"}
                    ]
                }
            ),
            row=2, col=2
        )
        
        # 6. Inflation Rate
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=economic_snapshot.inflation_rate,
                title={'text': "Inflation Rate (%)"},
                gauge={
                    'axis': {'range': [None, 8]},
                    'bar': {'color': "green" if 1 < economic_snapshot.inflation_rate < 3 else "orange" if economic_snapshot.inflation_rate < 5 else "red"},
                    'steps': [
                        {'range': [0, 2], 'color': "yellow"},
                        {'range': [2, 3], 'color': "lightgreen"},
                        {'range': [3, 5], 'color': "yellow"},
                        {'range': [5, 8], 'color': "lightcoral"}
                    ]
                }
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            title='Economic Indicators Dashboard',
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"✅ Economic indicators chart saved to: {save_path}")
            return save_path
        else:
            return fig.to_html()
    
    def create_regulatory_report_charts(self, portfolio_metrics, capital_calculations: Dict[str, float],
                                      save_path: Optional[str] = None) -> str:
        """Create regulatory compliance visualization"""
        
        # Create regulatory dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Capital Adequacy Ratios', 'Risk-Weighted Assets Breakdown',
                          'Expected vs Unexpected Loss', 'Regulatory Limits Compliance'],
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "table"}]]
        )
        
        # 1. Capital Adequacy Ratios
        if capital_calculations:
            ratios = ['CET1', 'Tier 1', 'Total Capital']
            current_ratios = [11.5, 13.2, 15.8]  # Example ratios
            minimum_ratios = [4.5, 6.0, 8.0]
            
            fig.add_trace(
                go.Bar(
                    x=ratios,
                    y=current_ratios,
                    name="Current Ratios",
                    marker=dict(color='green')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=ratios,
                    y=minimum_ratios,
                    name="Minimum Required",
                    marker=dict(color='red', opacity=0.7)
                ),
                row=1, col=1
            )
        
        # 2. Risk-Weighted Assets Breakdown
        if portfolio_metrics:
            rwa_categories = ['Consumer Loans', 'Credit Cards', 'Mortgages', 'Commercial']
            rwa_amounts = [portfolio_metrics.total_exposure * 0.6, 
                          portfolio_metrics.total_exposure * 0.25,
                          portfolio_metrics.total_exposure * 0.1,
                          portfolio_metrics.total_exposure * 0.05]
            
            fig.add_trace(
                go.Pie(
                    labels=rwa_categories,
                    values=rwa_amounts,
                    hole=0.3
                ),
                row=1, col=2
            )
        
        # 3. Expected vs Unexpected Loss
        if portfolio_metrics:
            loss_types = ['Expected Loss', 'Unexpected Loss']
            loss_amounts = [portfolio_metrics.expected_loss, portfolio_metrics.unexpected_loss]
            
            fig.add_trace(
                go.Bar(
                    x=loss_types,
                    y=loss_amounts,
                    marker=dict(color=['orange', 'red'])
                ),
                row=2, col=1
            )
        
        # 4. Regulatory Compliance Table
        compliance_data = [
            ['Metric', 'Current', 'Required', 'Status'],
            ['CET1 Ratio', '11.5%', '4.5%', '✅ Compliant'],
            ['Tier 1 Ratio', '13.2%', '6.0%', '✅ Compliant'],
            ['Total Capital Ratio', '15.8%', '8.0%', '✅ Compliant'],
            ['Leverage Ratio', '5.2%', '3.0%', '✅ Compliant'],
            ['LCR', '125%', '100%', '✅ Compliant']
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=compliance_data[0],
                          fill_color='paleturquoise',
                          align='center'),
                cells=dict(values=list(zip(*compliance_data[1:])),
                         fill_color='lavender',
                         align='center')
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Regulatory Compliance Dashboard',
            height=800,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"✅ Regulatory report saved to: {save_path}")
            return save_path
        else:
            return fig.to_html()