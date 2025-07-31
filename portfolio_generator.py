"""
Synthetic Portfolio Generator Module
===================================

Generates realistic synthetic loan portfolios with:
- Demographically accurate borrower profiles
- Regional economic variations
- Industry-specific risk factors
- Realistic loan characteristics
- Economic correlation factors

Used to create test data for credit risk modeling without real customer data.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import random

@dataclass
class Borrower:
    """Individual borrower profile with loan and risk characteristics"""
    borrower_id: str
    age: int
    income: float
    credit_score: int
    debt_to_income: float
    employment_length: int
    loan_amount: float
    loan_purpose: str
    home_ownership: str
    state: str
    industry: str
    education_level: str
    loan_term: int
    interest_rate: float
    loan_grade: str = None
    
    # Economic context
    local_unemployment: Optional[float] = None
    regional_gdp_growth: Optional[float] = None
    industry_risk_factor: Optional[float] = None
    
    # Risk metrics (to be populated by models)
    probability_default: Optional[float] = None
    risk_grade: Optional[str] = None
    expected_loss: Optional[float] = None

class SyntheticPortfolioGenerator:
    """
    Generate realistic synthetic loan portfolios for risk analysis
    """
    
    def __init__(self, economic_data_collector):
        """
        Initialize portfolio generator
        
        Args:
            economic_data_collector: Instance of EconomicDataCollector
        """
        self.economic_data = economic_data_collector
        
        # Industry risk factors (multiplier for base risk)
        self.industry_risk_factors = {
            'Technology': 0.8,
            'Healthcare': 0.9,
            'Finance': 1.0,
            'Education': 0.7,
            'Government': 0.6,
            'Manufacturing': 1.2,
            'Retail': 1.4,
            'Hospitality': 1.6,
            'Construction': 1.5,
            'Energy': 1.3,
            'Transportation': 1.1,
            'Utilities': 0.8,
            'Real Estate': 1.3,
            'Professional Services': 0.9,
            'Food Service': 1.4
        }
        
        # State economic strength factors (lower = stronger economy)
        self.state_risk_factors = {
            'CA': 0.9, 'TX': 0.8, 'NY': 1.0, 'FL': 1.1, 'IL': 1.2,
            'PA': 1.0, 'OH': 1.1, 'GA': 0.9, 'NC': 0.8, 'MI': 1.3,
            'NJ': 1.0, 'VA': 0.8, 'WA': 0.9, 'AZ': 1.0, 'MA': 0.9,
            'TN': 0.9, 'IN': 1.0, 'MO': 1.1, 'MD': 0.9, 'WI': 1.0,
            'CO': 0.8, 'MN': 0.9, 'SC': 1.0, 'AL': 1.2, 'LA': 1.3,
            'KY': 1.2, 'OR': 1.0, 'OK': 1.1, 'CT': 1.0, 'UT': 0.8,
            'IA': 0.9, 'NV': 1.2, 'AR': 1.3, 'MS': 1.4, 'KS': 1.0,
            'NM': 1.2, 'NE': 0.9, 'WV': 1.5, 'ID': 1.0, 'NH': 0.9,
            'HI': 1.1, 'ME': 1.1, 'MT': 1.1, 'RI': 1.1, 'DE': 1.0,
            'SD': 0.9, 'ND': 0.8, 'AK': 1.2, 'VT': 1.0, 'WY': 1.0
        }
        
        # Loan purposes with risk characteristics
        self.loan_purposes = {
            'debt_consolidation': {'risk_factor': 1.0, 'probability': 0.35},
            'home_improvement': {'risk_factor': 0.8, 'probability': 0.15},
            'major_purchase': {'risk_factor': 1.1, 'probability': 0.12},
            'medical': {'risk_factor': 1.3, 'probability': 0.08},
            'vacation': {'risk_factor': 1.4, 'probability': 0.06},
            'wedding': {'risk_factor': 1.2, 'probability': 0.05},
            'business': {'risk_factor': 1.5, 'probability': 0.04},
            'education': {'risk_factor': 0.9, 'probability': 0.08},
            'moving': {'risk_factor': 1.1, 'probability': 0.04},
            'other': {'risk_factor': 1.2, 'probability': 0.03}
        }
        
        # Education levels with income impacts
        self.education_levels = {
            'High School': {'income_multiplier': 0.7, 'probability': 0.28},
            'Some College': {'income_multiplier': 0.85, 'probability': 0.32},
            'Bachelor': {'income_multiplier': 1.0, 'probability': 0.25},
            'Master': {'income_multiplier': 1.3, 'probability': 0.12},
            'PhD': {'income_multiplier': 1.5, 'probability': 0.03}
        }
        
        print("🏭 Synthetic Portfolio Generator initialized")
    
    def _select_weighted_choice(self, choices_dict: Dict[str, Dict]) -> str:
        """Select item based on probability weights"""
        items = list(choices_dict.keys())
        weights = [choices_dict[item]['probability'] for item in items]
        return np.random.choice(items, p=weights)
    
    def _generate_demographics(self) -> Dict:
        """Generate realistic demographic profile"""
        
        # Age distribution (peak around 35-45 for borrowers)
        age = max(18, min(80, int(np.random.gamma(4, 8) + 20)))
        
        # Education level affects income potential
        education_level = self._select_weighted_choice(self.education_levels)
        education_multiplier = self.education_levels[education_level]['income_multiplier']
        
        # Income generation (log-normal distribution, adjusted for education)
        # Base income distribution peaks around $60K
        base_income = np.random.lognormal(mean=10.9, sigma=0.7)
        income = base_income * education_multiplier
        
        # Age premium (experience increases income)
        age_multiplier = 1 + (age - 25) * 0.01 if age > 25 else 1.0
        income *= age_multiplier
        
        # Cap income at reasonable levels
        income = min(750000, max(25000, income))
        
        return {
            'age': age,
            'education_level': education_level,
            'income': income
        }
    
    def _generate_credit_profile(self, demographics: Dict) -> Dict:
        """Generate credit profile based on demographics"""
        
        age = demographics['age']
        income = demographics['income']
        
        # Credit score generation
        # Higher income and age generally correlate with better credit
        income_factor = min(2.0, income / 60000)  # Income impact
        age_factor = min(1.5, age / 35)           # Age/experience impact
        
        # Base credit score with income and age adjustments
        base_score = 650 + (income_factor * 60) + (age_factor * 40)
        
        # Add realistic variance
        credit_score = int(max(300, min(850, np.random.normal(base_score, 75))))
        
        # Employment length (correlated with age)
        max_employment = max(1, age - 22)  # Can't work before 22
        employment_length = min(max_employment, max(0, int(np.random.exponential(6))))
        
        # Debt-to-income ratio (lower for higher credit scores)
        if credit_score >= 750:
            dti_mean, dti_std = 0.15, 0.10
        elif credit_score >= 700:
            dti_mean, dti_std = 0.25, 0.15
        elif credit_score >= 650:
            dti_mean, dti_std = 0.35, 0.15
        else:
            dti_mean, dti_std = 0.45, 0.20
        
        debt_to_income = max(0.0, min(0.65, np.random.normal(dti_mean, dti_std)))
        
        return {
            'credit_score': credit_score,
            'employment_length': employment_length,
            'debt_to_income': debt_to_income
        }
    
    def _generate_geographic_profile(self) -> Dict:
        """Generate geographic profile with economic context"""
        
        # Select state based on population (weighted towards larger states)
        state_weights = {
            'CA': 0.12, 'TX': 0.09, 'FL': 0.065, 'NY': 0.06, 'PA': 0.04,
            'IL': 0.04, 'OH': 0.035, 'GA': 0.032, 'NC': 0.032, 'MI': 0.03,
            'NJ': 0.027, 'VA': 0.026, 'WA': 0.023, 'AZ': 0.022, 'MA': 0.021,
            'TN': 0.021, 'IN': 0.020, 'MO': 0.018, 'MD': 0.018, 'WI': 0.018,
            'CO': 0.017, 'MN': 0.017, 'SC': 0.015, 'AL': 0.015, 'LA': 0.014,
            'KY': 0.013, 'OR': 0.013, 'OK': 0.012, 'CT': 0.011, 'UT': 0.010
        }
        
        # Add remaining states with small weights
        all_states = list(self.state_risk_factors.keys())
        remaining_states = [s for s in all_states if s not in state_weights]
        remaining_weight = 1 - sum(state_weights.values())
        for state in remaining_states:
            state_weights[state] = remaining_weight / len(remaining_states)
        
        state = np.random.choice(list(state_weights.keys()), p=list(state_weights.values()))
        
        # Get economic data for the state
        local_unemployment = self.economic_data.get_state_unemployment(state)
        
        # Regional GDP growth (state-specific variation around national)
        national_growth = 2.1  # Baseline
        state_variation = np.random.normal(0, 0.8)
        regional_gdp_growth = national_growth + state_variation
        
        return {
            'state': state,
            'local_unemployment': local_unemployment,
            'regional_gdp_growth': regional_gdp_growth
        }
    
    def _generate_industry_profile(self) -> Dict:
        """Generate industry profile"""
        
        # Industry selection (weighted by employment)
        industry_weights = {
            'Healthcare': 0.14, 'Retail': 0.12, 'Professional Services': 0.10,
            'Manufacturing': 0.09, 'Education': 0.08, 'Food Service': 0.08,
            'Finance': 0.06, 'Construction': 0.06, 'Transportation': 0.05,
            'Technology': 0.05, 'Government': 0.05, 'Hospitality': 0.04,
            'Real Estate': 0.03, 'Energy': 0.03, 'Utilities': 0.02
        }
        
        industry = np.random.choice(list(industry_weights.keys()), p=list(industry_weights.values()))
        industry_risk_factor = self.industry_risk_factors[industry]
        
        return {
            'industry': industry,
            'industry_risk_factor': industry_risk_factor
        }
    
    def _generate_loan_characteristics(self, borrower_profile: Dict) -> Dict:
        """Generate loan characteristics based on borrower profile"""
        
        income = borrower_profile['income']
        credit_score = borrower_profile['credit_score']
        dti = borrower_profile['debt_to_income']
        
        # Loan purpose
        loan_purpose = self._select_weighted_choice(self.loan_purposes)
        purpose_risk_factor = self.loan_purposes[loan_purpose]['risk_factor']
        
        # Loan amount based on income, credit, and DTI
        # Higher credit scores can borrow more relative to income
        if credit_score >= 750:
            max_loan_ratio = 0.6
        elif credit_score >= 700:
            max_loan_ratio = 0.5
        elif credit_score >= 650:
            max_loan_ratio = 0.4
        else:
            max_loan_ratio = 0.3
        
        # Adjust for existing DTI
        available_capacity = max(0.1, 1 - dti)
        max_loan = income * max_loan_ratio * available_capacity
        
        # Actual loan amount (some borrowers take less than maximum)
        loan_utilization = np.random.beta(2, 3)  # Skewed towards smaller loans
        loan_amount = max(1000, min(50000, max_loan * loan_utilization))
        
        # Loan term (36 or 60 months, with preference based on amount)
        if loan_amount > 25000:
            loan_term = np.random.choice([36, 60], p=[0.3, 0.7])
        else:
            loan_term = np.random.choice([36, 60], p=[0.6, 0.4])
        
        # Interest rate calculation
        base_rate = self._calculate_interest_rate(credit_score, loan_amount, loan_term, purpose_risk_factor)
        
        # Home ownership
        ownership_prob = self._calculate_ownership_probability(income, credit_score, borrower_profile['age'])
        home_ownership = np.random.choice(
            ['own', 'rent', 'mortgage'], 
            p=[ownership_prob * 0.3, 1 - ownership_prob, ownership_prob * 0.7]
        )
        
        # Assign preliminary loan grade
        loan_grade = self._assign_loan_grade(credit_score, dti, base_rate)
        
        return {
            'loan_purpose': loan_purpose,
            'loan_amount': loan_amount,
            'loan_term': loan_term,
            'interest_rate': base_rate,
            'home_ownership': home_ownership,
            'loan_grade': loan_grade
        }
    
    def _calculate_interest_rate(self, credit_score: int, loan_amount: float, 
                                loan_term: int, purpose_risk_factor: float) -> float:
        """Calculate interest rate based on borrower and loan characteristics"""
        
        # Get current economic conditions
        try:
            economic_snapshot = self.economic_data.get_current_economic_snapshot()
            base_rate = economic_snapshot.federal_funds_rate + 5.0  # Base consumer rate
        except:
            base_rate = 9.5  # Fallback rate
        
        # Credit score adjustment (major factor)
        if credit_score >= 800:
            credit_adjustment = -2.0
        elif credit_score >= 750:
            credit_adjustment = -1.0
        elif credit_score >= 700:
            credit_adjustment = 0.0
        elif credit_score >= 650:
            credit_adjustment = 2.0
        elif credit_score >= 600:
            credit_adjustment = 4.0
        else:
            credit_adjustment = 6.0
        
        # Loan amount adjustment (larger loans slightly lower rates)
        amount_adjustment = -0.5 if loan_amount > 30000 else 0.0
        
        # Term adjustment (longer terms higher rates)
        term_adjustment = 1.0 if loan_term == 60 else 0.0
        
        # Purpose adjustment
        purpose_adjustment = (purpose_risk_factor - 1.0) * 2.0
        
        # Calculate final rate
        final_rate = base_rate + credit_adjustment + amount_adjustment + term_adjustment + purpose_adjustment
        
        # Ensure reasonable bounds
        return max(5.0, min(35.0, final_rate))
    
    def _calculate_ownership_probability(self, income: float, credit_score: int, age: int) -> float:
        """Calculate probability of home ownership"""
        
        income_factor = min(1.0, income / 75000)
        credit_factor = min(1.0, (credit_score - 500) / 300)
        age_factor = min(1.0, (age - 25) / 15)
        
        ownership_prob = 0.2 + (income_factor * 0.3) + (credit_factor * 0.3) + (age_factor * 0.2)
        return max(0.1, min(0.9, ownership_prob))
    
    def _assign_loan_grade(self, credit_score: int, dti: float, interest_rate: float) -> str:
        """Assign initial loan grade based on risk factors"""
        
        if credit_score >= 780 and dti <= 0.2 and interest_rate <= 12:
            return 'A'
        elif credit_score >= 720 and dti <= 0.3 and interest_rate <= 15:
            return 'B'
        elif credit_score >= 660 and dti <= 0.4 and interest_rate <= 20:
            return 'C'
        elif credit_score >= 600 and dti <= 0.5 and interest_rate <= 25:
            return 'D'
        elif credit_score >= 550 and dti <= 0.6:
            return 'E'
        else:
            return 'F'
    
    def generate_portfolio(self, num_borrowers: int = 10000) -> List[Borrower]:
        """
        Generate complete synthetic loan portfolio with optimized API usage
        
        Args:
            num_borrowers: Number of borrowers to generate
            
        Returns:
            List of Borrower objects
        """
        
        print(f"🏭 Generating synthetic portfolio of {num_borrowers:,} borrowers...")
        
        # 🔧 OPTIMIZATION: Pre-fetch economic data once for entire portfolio
        print("📊 Pre-fetching economic data...")
        try:
            economic_snapshot = self.economic_data.get_current_economic_snapshot()
            base_fed_rate = economic_snapshot.federal_funds_rate
            base_unemployment = economic_snapshot.unemployment_rate
        except Exception as e:
            print(f"⚠️  Using fallback economic data: {e}")
            base_fed_rate = 4.5
            base_unemployment = 4.2
        
        # 🔧 Pre-cache state unemployment rates for major states
        print("📍 Caching state unemployment data...")
        state_unemployment_cache = {}
        major_states = ['CA', 'TX', 'NY', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI', 
                       'NJ', 'VA', 'WA', 'AZ', 'MA', 'TN', 'IN', 'MO', 'MD', 'WI', 'CO']
        
        for state in major_states:
            try:
                state_unemployment_cache[state] = self.economic_data.get_state_unemployment(state)
            except:
                # Use base unemployment with realistic variation
                state_unemployment_cache[state] = base_unemployment + np.random.normal(0, 0.8)
            
            # Small delay to avoid overwhelming APIs
            import time
            time.sleep(0.1)
        
        print(f"✅ Cached unemployment data for {len(state_unemployment_cache)} states")
        
        borrowers = []
        
        for i in range(num_borrowers):
            if i % 1000 == 0 and i > 0:
                print(f"   Generated {i:,} borrowers...")
            
            try:
                # Generate borrower profile components
                demographics = self._generate_demographics()
                credit_profile = self._generate_credit_profile(demographics)
                geographic_profile = self._generate_geographic_profile_cached(state_unemployment_cache, base_unemployment)
                industry_profile = self._generate_industry_profile()
                
                # Combine all profile components
                borrower_profile = {**demographics, **credit_profile, **geographic_profile, **industry_profile}
                
                # Generate loan characteristics (using cached economic data)
                loan_characteristics = self._generate_loan_characteristics_cached(borrower_profile, base_fed_rate)
                borrower_profile.update(loan_characteristics)
                
                # Create Borrower object
                borrower = Borrower(
                    borrower_id=f"SYNTH_{i+1:06d}",
                    **borrower_profile
                )
                
                borrowers.append(borrower)
                
            except Exception as e:
                print(f"⚠️  Error generating borrower {i}: {e}")
                continue
        
        print(f"✅ Successfully generated {len(borrowers):,} synthetic borrowers")
        
        # Generate portfolio statistics
        self._print_portfolio_statistics(borrowers)
        
        return borrowers
    
    def _generate_geographic_profile_cached(self, state_cache: Dict[str, float], base_unemployment: float) -> Dict:
        """Generate geographic profile using cached unemployment data"""
        
        # Select state based on population (weighted towards larger states)
        state_weights = {
            'CA': 0.12, 'TX': 0.09, 'FL': 0.065, 'NY': 0.06, 'PA': 0.04,
            'IL': 0.04, 'OH': 0.035, 'GA': 0.032, 'NC': 0.032, 'MI': 0.03,
            'NJ': 0.027, 'VA': 0.026, 'WA': 0.023, 'AZ': 0.022, 'MA': 0.021,
            'TN': 0.021, 'IN': 0.020, 'MO': 0.018, 'MD': 0.018, 'WI': 0.018,
            'CO': 0.017, 'MN': 0.017, 'SC': 0.015, 'AL': 0.015, 'LA': 0.014,
            'KY': 0.013, 'OR': 0.013, 'OK': 0.012, 'CT': 0.011, 'UT': 0.010
        }
        
        # Add remaining states with small weights
        all_states = list(self.state_risk_factors.keys())
        remaining_states = [s for s in all_states if s not in state_weights]
        remaining_weight = 1 - sum(state_weights.values())
        for state in remaining_states:
            state_weights[state] = remaining_weight / len(remaining_states)
        
        state = np.random.choice(list(state_weights.keys()), p=list(state_weights.values()))
        
        # Get unemployment from cache or use estimate
        if state in state_cache:
            local_unemployment = state_cache[state]
        else:
            # Generate realistic unemployment for non-cached states
            local_unemployment = base_unemployment + np.random.normal(0, 0.8)
            local_unemployment = max(1.0, local_unemployment)  # Ensure positive
        
        # Regional GDP growth (state-specific variation around national)
        national_growth = 2.1  # Baseline
        state_variation = np.random.normal(0, 0.8)
        regional_gdp_growth = national_growth + state_variation
        
        return {
            'state': state,
            'local_unemployment': local_unemployment,
            'regional_gdp_growth': regional_gdp_growth
        }
    
    def _generate_loan_characteristics_cached(self, borrower_profile: Dict, base_fed_rate: float) -> Dict:
        """Generate loan characteristics using cached economic data"""
        
        income = borrower_profile['income']
        credit_score = borrower_profile['credit_score']
        dti = borrower_profile['debt_to_income']
        
        # Loan purpose
        loan_purpose = self._select_weighted_choice(self.loan_purposes)
        purpose_risk_factor = self.loan_purposes[loan_purpose]['risk_factor']
        
        # Loan amount based on income, credit, and DTI
        # Higher credit scores can borrow more relative to income
        if credit_score >= 750:
            max_loan_ratio = 0.6
        elif credit_score >= 700:
            max_loan_ratio = 0.5
        elif credit_score >= 650:
            max_loan_ratio = 0.4
        else:
            max_loan_ratio = 0.3
        
        # Adjust for existing DTI
        available_capacity = max(0.1, 1 - dti)
        max_loan = income * max_loan_ratio * available_capacity
        
        # Actual loan amount (some borrowers take less than maximum)
        loan_utilization = np.random.beta(2, 3)  # Skewed towards smaller loans
        loan_amount = max(1000, min(50000, max_loan * loan_utilization))
        
        # Loan term (36 or 60 months, with preference based on amount)
        if loan_amount > 25000:
            loan_term = np.random.choice([36, 60], p=[0.3, 0.7])
        else:
            loan_term = np.random.choice([36, 60], p=[0.6, 0.4])
        
        # Interest rate calculation using cached fed rate
        base_rate = base_fed_rate + 5.0  # Base consumer rate
        
        # Credit score adjustment (major factor)
        if credit_score >= 800:
            credit_adjustment = -2.0
        elif credit_score >= 750:
            credit_adjustment = -1.0
        elif credit_score >= 700:
            credit_adjustment = 0.0
        elif credit_score >= 650:
            credit_adjustment = 2.0
        elif credit_score >= 600:
            credit_adjustment = 4.0
        else:
            credit_adjustment = 6.0
        
        # Other adjustments
        amount_adjustment = -0.5 if loan_amount > 30000 else 0.0
        term_adjustment = 1.0 if loan_term == 60 else 0.0
        purpose_adjustment = (purpose_risk_factor - 1.0) * 2.0
        
        # Calculate final rate
        final_rate = base_rate + credit_adjustment + amount_adjustment + term_adjustment + purpose_adjustment
        final_rate = max(5.0, min(35.0, final_rate))
        
        # Home ownership
        ownership_prob = self._calculate_ownership_probability(income, credit_score, borrower_profile['age'])
        home_ownership = np.random.choice(
            ['own', 'rent', 'mortgage'], 
            p=[ownership_prob * 0.3, 1 - ownership_prob, ownership_prob * 0.7]
        )
        
        # Assign preliminary loan grade
        loan_grade = self._assign_loan_grade(credit_score, dti, final_rate)
        
        return {
            'loan_purpose': loan_purpose,
            'loan_amount': loan_amount,
            'loan_term': loan_term,
            'interest_rate': final_rate,
            'home_ownership': home_ownership,
            'loan_grade': loan_grade
        }
    
    def _print_portfolio_statistics(self, borrowers: List[Borrower]):
        """Print summary statistics for generated portfolio"""
        
        if not borrowers:
            return
        
        df = pd.DataFrame([borrower.__dict__ for borrower in borrowers])
        
        print(f"\n📊 PORTFOLIO STATISTICS:")
        print(f"   Total Borrowers: {len(borrowers):,}")
        print(f"   Total Loan Volume: ${df['loan_amount'].sum():,.0f}")
        print(f"   Average Loan Size: ${df['loan_amount'].mean():,.0f}")
        print(f"   Median Credit Score: {df['credit_score'].median():.0f}")
        print(f"   Average DTI: {df['debt_to_income'].mean():.1%}")
        print(f"   Average Interest Rate: {df['interest_rate'].mean():.2f}%")
        
        print(f"\n📋 DEMOGRAPHICS:")
        print(f"   Average Age: {df['age'].mean():.1f} years")
        print(f"   Median Income: ${df['income'].median():,.0f}")
        
        print(f"\n🏢 TOP INDUSTRIES:")
        industry_counts = df['industry'].value_counts().head(5)
        for industry, count in industry_counts.items():
            pct = count / len(df) * 100
            print(f"   {industry}: {count:,} ({pct:.1f}%)")
        
        print(f"\n🗺️  TOP STATES:")
        state_counts = df['state'].value_counts().head(5)
        for state, count in state_counts.items():
            pct = count / len(df) * 100
            print(f"   {state}: {count:,} ({pct:.1f}%)")
        
        print(f"\n📈 LOAN GRADES:")
        grade_counts = df['loan_grade'].value_counts().sort_index()
        for grade, count in grade_counts.items():
            pct = count / len(df) * 100
            print(f"   Grade {grade}: {count:,} ({pct:.1f}%)")