"""
Credit Risk Models Module
========================

Machine learning models for credit risk assessment including:
- Probability of Default (PD) modeling
- Loss Given Default (LGD) estimation
- Expected Loss calculation
- Risk grade assignment
- Model validation and performance metrics

Uses multiple ML algorithms for robust credit scoring.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import warnings
warnings.filterwarnings('ignore')

class CreditRiskModeler:
    """
    Advanced credit risk modeling system
    
    Provides multiple ML models for credit risk assessment with
    comprehensive validation and performance tracking.
    """
    
    def __init__(self):
        """Initialize credit risk modeling system"""
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_names = []
        self.is_trained = False
        
        # Risk grade thresholds (PD percentages)
        self.risk_grade_thresholds = {
            'AAA': (0.0, 0.25),
            'AA':  (0.25, 0.5),
            'A':   (0.5, 1.0),
            'BBB': (1.0, 2.5),
            'BB':  (2.5, 5.0),
            'B':   (5.0, 10.0),
            'CCC': (10.0, 20.0),
            'CC':  (20.0, 35.0),
            'C':   (35.0, 100.0)
        }
        
        # Model configurations
        self.model_configs = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000, solver='liblinear'),
                'requires_scaling': True,
                'description': 'Linear probability model with regularization'
            },
            'random_forest': {
                'model': RandomForestClassifier(
                    n_estimators=100, 
                    max_depth=10,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    random_state=42,
                    n_jobs=-1
                ),
                'requires_scaling': False,
                'description': 'Ensemble of decision trees'
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                ),
                'requires_scaling': False,
                'description': 'Sequential ensemble with boosting'
            }
        }
        
        print("🤖 Credit Risk Modeler initialized")
        print(f"   Available models: {', '.join(self.model_configs.keys())}")
    
    def prepare_features(self, borrowers: List) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for machine learning models
        
        Args:
            borrowers: List of Borrower objects
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        
        print("🔧 Preparing features for modeling...")
        
        # Convert borrowers to DataFrame
        borrower_data = []
        for borrower in borrowers:
            borrower_dict = borrower.__dict__.copy()
            borrower_data.append(borrower_dict)
        
        df = pd.DataFrame(borrower_data)
        
        # Create synthetic default target variable
        # This simulates real default data based on risk factors
        y = self._generate_default_target(df)
        
        # Feature engineering
        df = self._engineer_features(df)
        
        # Select and prepare features for modeling
        X = self._select_model_features(df)
        
        print(f"✅ Prepared {X.shape[0]:,} samples with {X.shape[1]} features")
        print(f"   Default rate: {y.mean():.2%}")
        
        return X, y
    
    def _generate_default_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate realistic default target based on borrower characteristics
        
        This creates a synthetic but realistic default indicator based on
        known risk factors and their typical relationships.
        """
        
        # Calculate base default probability for each borrower
        default_score = np.zeros(len(df))
        
        # Credit score impact (most important factor)
        credit_impact = (850 - df['credit_score']) / 550 * 0.4
        default_score += credit_impact
        
        # Debt-to-income impact
        dti_impact = df['debt_to_income'] * 0.3
        default_score += dti_impact
        
        # Income stability (lower income = higher risk)
        income_impact = np.maximum(0, (50000 - df['income']) / 100000) * 0.1
        default_score += income_impact
        
        # Employment length (shorter = higher risk)
        emp_impact = np.maximum(0, (2 - df['employment_length']) / 10) * 0.05
        default_score += emp_impact
        
        # Regional economic factors
        if 'local_unemployment' in df.columns:
            unemployment_impact = (df['local_unemployment'] - 4.0) / 10 * 0.05
            default_score += unemployment_impact
        
        # Industry risk factors
        if 'industry_risk_factor' in df.columns:
            industry_impact = (df['industry_risk_factor'] - 1.0) * 0.05
            default_score += industry_impact
        
        # Interest rate impact (higher rates = higher stress)
        rate_impact = np.maximum(0, (df['interest_rate'] - 15.0) / 20.0) * 0.1
        default_score += rate_impact
        
        # Add random component to simulate market volatility
        random_component = np.random.normal(0, 0.05, len(df))
        default_score += random_component
        
        # Convert to probabilities (sigmoid transformation)
        default_probabilities = 1 / (1 + np.exp(-default_score * 5))
        
        # Set target default rate (typically 8-15% for consumer lending)
        target_default_rate = 0.12
        threshold = np.percentile(default_probabilities, (1 - target_default_rate) * 100)
        
        # Create binary target
        defaults = (default_probabilities > threshold).astype(int)
        
        return pd.Series(defaults, index=df.index)
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features for better model performance"""
        
        # Income-based ratios
        df['income_to_loan_ratio'] = df['income'] / (df['loan_amount'] + 1)
        df['monthly_payment_est'] = df['loan_amount'] / df['loan_term']
        df['payment_to_income_ratio'] = (df['monthly_payment_est'] * 12) / df['income']
        
        # Credit utilization proxy
        df['debt_amount_est'] = df['debt_to_income'] * df['income']
        df['total_debt_with_loan'] = df['debt_amount_est'] + df['loan_amount']
        df['total_dti_with_loan'] = df['total_debt_with_loan'] / df['income']
        
        # Employment stability
        df['employment_stability'] = np.minimum(df['employment_length'] / 5.0, 1.0)
        
        # Age-income interaction
        df['age_income_interaction'] = (df['age'] / 50) * np.log1p(df['income'])
        
        # Credit score categories
        df['credit_tier'] = pd.cut(df['credit_score'], 
                                  bins=[0, 580, 670, 740, 800, 850], 
                                  labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
        
        # Loan amount categories
        df['loan_size_category'] = pd.cut(df['loan_amount'], 
                                         bins=[0, 5000, 15000, 30000, 50000], 
                                         labels=['Small', 'Medium', 'Large', 'Very Large'])
        
        # Home ownership risk factor
        ownership_risk = {'own': 0.7, 'mortgage': 0.8, 'rent': 1.0}
        df['ownership_risk_factor'] = df['home_ownership'].map(ownership_risk)
        
        return df
    
    def _select_model_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and encode features for modeling"""
        
        # Numerical features
        numerical_features = [
            'age', 'income', 'credit_score', 'debt_to_income', 'employment_length',
            'loan_amount', 'loan_term', 'interest_rate', 'local_unemployment',
            'regional_gdp_growth', 'industry_risk_factor', 'income_to_loan_ratio',
            'monthly_payment_est', 'payment_to_income_ratio', 'total_dti_with_loan',
            'employment_stability', 'age_income_interaction', 'ownership_risk_factor'
        ]
        
        # Categorical features to encode
        categorical_features = [
            'loan_purpose', 'home_ownership', 'state', 'industry', 
            'education_level', 'credit_tier', 'loan_size_category'
        ]
        
        # Start with numerical features
        feature_df = df[numerical_features].copy()
        
        # Handle missing values in numerical features
        feature_df = feature_df.fillna(feature_df.median())
        
        # Encode categorical features
        for cat_feature in categorical_features:
            if cat_feature in df.columns:
                # Initialize encoder if not exists
                if cat_feature not in self.label_encoders:
                    self.label_encoders[cat_feature] = LabelEncoder()
                    encoded_values = self.label_encoders[cat_feature].fit_transform(df[cat_feature].astype(str))
                else:
                    # Handle new categories in prediction
                    try:
                        encoded_values = self.label_encoders[cat_feature].transform(df[cat_feature].astype(str))
                    except ValueError:
                        # Handle unseen categories
                        encoded_values = np.zeros(len(df))
                
                feature_df[f'{cat_feature}_encoded'] = encoded_values
        
        # Store feature names for later use
        self.feature_names = list(feature_df.columns)
        
        return feature_df
    
    def train_models(self, borrowers: List) -> Dict[str, Any]:
        """
        Train multiple credit risk models
        
        Args:
            borrowers: List of Borrower objects
            
        Returns:
            Dictionary with training results and performance metrics
        """
        
        print("🤖 Training credit risk models...")
        
        # Prepare data
        X, y = self.prepare_features(borrowers)
        
        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   Training set: {X_train.shape[0]:,} samples")
        print(f"   Test set: {X_test.shape[0]:,} samples")
        
        # Initialize scalers
        self.scalers['standard'] = StandardScaler()
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_test_scaled = self.scalers['standard'].transform(X_test)
        
        # Train each model
        results = {}
        
        for model_name, config in self.model_configs.items():
            print(f"\n   🔧 Training {model_name}...")
            
            model = config['model']
            requires_scaling = config['requires_scaling']
            
            # Select appropriate data (scaled vs unscaled)
            if requires_scaling:
                X_train_model = X_train_scaled
                X_test_model = X_test_scaled
            else:
                X_train_model = X_train
                X_test_model = X_test
            
            # Train model
            model.fit(X_train_model, y_train)
            
            # Make predictions
            y_pred_proba = model.predict_proba(X_test_model)[:, 1]
            y_pred = model.predict(X_test_model)
            
            # Calculate performance metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train_model, y_train, 
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='roc_auc'
            )
            
            # Feature importance (for tree-based models)
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(self.feature_names, model.feature_importances_))
                feature_importance = dict(sorted(importance_dict.items(), 
                                               key=lambda x: x[1], reverse=True))
            
            # Store model and results
            self.models[model_name] = model
            results[model_name] = {
                'auc_score': auc_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'feature_importance': feature_importance,
                'description': config['description']
            }
            
            print(f"      ✅ AUC: {auc_score:.3f} (CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f})")
        
        # Store test data for later analysis
        self.X_test = X_test
        self.y_test = y_test
        self.X_test_scaled = X_test_scaled
        self.is_trained = True
        
        print(f"\n✅ Model training completed")
        
        return results
    
    def predict_default_probability(self, borrowers: List, model_name: str = 'random_forest') -> List:
        """
        Predict default probabilities and assign risk grades
        
        Args:
            borrowers: List of Borrower objects
            model_name: Name of model to use for predictions
            
        Returns:
            List of borrowers with updated risk metrics
        """
        
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
        
        print(f"🎯 Predicting default probabilities using {model_name}...")
        
        # Prepare features
        X, _ = self.prepare_features(borrowers)
        
        # Get model and check if scaling is required
        model = self.models[model_name]
        requires_scaling = self.model_configs[model_name]['requires_scaling']
        
        # Apply scaling if required
        if requires_scaling:
            X_model = self.scalers['standard'].transform(X)
        else:
            X_model = X
        
        # Make predictions
        probabilities = model.predict_proba(X_model)[:, 1] * 100  # Convert to percentages
        
        # Update borrowers with predictions
        for i, borrower in enumerate(borrowers):
            pd_percent = probabilities[i]
            borrower.probability_default = pd_percent
            
            # Assign risk grade based on PD
            borrower.risk_grade = self._assign_risk_grade(pd_percent)
            
            # Calculate expected loss (PD * LGD * EAD)
            # Standard assumptions: LGD = 45%, EAD = full loan amount
            lgd = 0.45  # Loss Given Default
            ead = borrower.loan_amount  # Exposure at Default
            borrower.expected_loss = (pd_percent / 100) * lgd * ead
        
        print(f"✅ Updated {len(borrowers):,} borrowers with risk metrics")
        
        # Print risk distribution
        self._print_risk_distribution(borrowers)
        
        return borrowers
    
    def _assign_risk_grade(self, pd_percent: float) -> str:
        """Assign risk grade based on probability of default"""
        
        for grade, (min_pd, max_pd) in self.risk_grade_thresholds.items():
            if min_pd <= pd_percent < max_pd:
                return grade
        
        return 'C'  # Highest risk category
    
    def _print_risk_distribution(self, borrowers: List):
        """Print distribution of risk grades"""
        
        risk_grades = [b.risk_grade for b in borrowers if b.risk_grade]
        if not risk_grades:
            return
        
        from collections import Counter
        grade_counts = Counter(risk_grades)
        total = len(risk_grades)
        
        print(f"\n📊 RISK GRADE DISTRIBUTION:")
        for grade in ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C']:
            count = grade_counts.get(grade, 0)
            pct = count / total * 100 if total > 0 else 0
            print(f"   Grade {grade}: {count:,} ({pct:.1f}%)")
    
    def get_model_performance_report(self) -> str:
        """Generate comprehensive model performance report"""
        
        if not self.is_trained:
            return "❌ Models not trained yet"
        
        report = "\n📊 CREDIT RISK MODEL PERFORMANCE REPORT\n"
        report += "=" * 60 + "\n"
        
        for model_name, model in self.models.items():
            config = self.model_configs[model_name]
            requires_scaling = config['requires_scaling']
            
            # Get test predictions
            if requires_scaling:
                y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
                y_pred = model.predict(self.X_test_scaled)
            else:
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                y_pred = model.predict(self.X_test)
            
            auc = roc_auc_score(self.y_test, y_pred_proba)
            
            report += f"\n🤖 {model_name.upper().replace('_', ' ')}:\n"
            report += f"   Description: {config['description']}\n"
            report += f"   AUC Score: {auc:.3f}\n"
            report += f"   \n"
            report += f"   Classification Report:\n"
            report += classification_report(self.y_test, y_pred, 
                                          target_names=['No Default', 'Default'],
                                          digits=3)
            report += "\n" + "-" * 50 + "\n"
        
        return report
    
    def get_feature_importance_report(self, model_name: str = 'random_forest') -> str:
        """Get feature importance analysis"""
        
        if not self.is_trained:
            return "❌ Models not trained yet"
        
        if model_name not in self.models:
            return f"❌ Model {model_name} not found"
        
        model = self.models[model_name]
        
        if not hasattr(model, 'feature_importances_'):
            return f"❌ Model {model_name} does not provide feature importance"
        
        importance_dict = dict(zip(self.feature_names, model.feature_importances_))
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        report = f"\n📈 FEATURE IMPORTANCE ANALYSIS ({model_name.upper()})\n"
        report += "=" * 50 + "\n"
        
        for i, (feature, importance) in enumerate(sorted_features[:15], 1):
            report += f"{i:2d}. {feature:<25} {importance:.4f}\n"
        
        return report
    
    def save_models(self, filepath_prefix: str = 'credit_models'):
        """Save trained models to disk"""
        
        if not self.is_trained:
            print("❌ No trained models to save")
            return
        
        import os
        from datetime import datetime
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            # Save models
            for model_name, model in self.models.items():
                filename = f"{filepath_prefix}_{model_name}_{timestamp}.pkl"
                joblib.dump(model, filename)
                print(f"✅ Saved {model_name} to {filename}")
            
            # Save scalers and encoders
            metadata = {
                'scalers': self.scalers,
                'label_encoders': self.label_encoders,
                'feature_names': self.feature_names,
                'risk_grade_thresholds': self.risk_grade_thresholds
            }
            
            metadata_filename = f"{filepath_prefix}_metadata_{timestamp}.pkl"
            joblib.dump(metadata, metadata_filename)
            print(f"✅ Saved metadata to {metadata_filename}")
            
        except Exception as e:
            print(f"❌ Error saving models: {e}")
    
    def load_models(self, filepath_prefix: str, timestamp: str):
        """Load previously trained models"""
        
        try:
            # Load models
            for model_name in self.model_configs.keys():
                filename = f"{filepath_prefix}_{model_name}_{timestamp}.pkl"
                if os.path.exists(filename):
                    self.models[model_name] = joblib.load(filename)
                    print(f"✅ Loaded {model_name} from {filename}")
            
            # Load metadata
            metadata_filename = f"{filepath_prefix}_metadata_{timestamp}.pkl"
            if os.path.exists(metadata_filename):
                metadata = joblib.load(metadata_filename)
                self.scalers = metadata['scalers']
                self.label_encoders = metadata['label_encoders']
                self.feature_names = metadata['feature_names']
                self.risk_grade_thresholds = metadata['risk_grade_thresholds']
                print(f"✅ Loaded metadata from {metadata_filename}")
                
                self.is_trained = True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")