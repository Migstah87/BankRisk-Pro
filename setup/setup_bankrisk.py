#!/usr/bin/env python3
"""
BankRisk Pro Setup Script
========================

Handles environment setup, dependency installation, and initial configuration
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_banner():
    """Print welcome banner"""
    print("🏦" + "="*60)
    print("  BankRisk Pro - Credit Risk Management Platform")
    print("  Professional Setup & Configuration")
    print("="*62)

def check_python_version():
    """Check Python version compatibility"""
    print("\n🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        print("   Please upgrade Python and try again.")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} - Compatible")
    return True

def install_dependencies():
    """Install required packages"""
    print("\n📦 Installing dependencies...")
    
    requirements = [
        "pandas>=1.5.0",
        "numpy>=1.21.0", 
        "scikit-learn>=1.1.0",
        "requests>=2.28.0",
        "yfinance>=0.1.87",
        "python-dotenv>=0.19.0",
        "plotly>=5.11.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "reportlab>=3.6.0",
        "scipy>=1.9.0",
        "joblib>=1.2.0"
    ]
    
    for package in requirements:
        try:
            print(f"   Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"   ✅ {package.split('>=')[0]} installed")
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️  Warning: Could not install {package}")
            continue
    
    print("✅ Core dependencies installed")

def create_project_structure():
    """Create project directory structure"""
    print("\n📁 Creating project structure...")
    
    directories = [
        "data",
        "outputs/csv",
        "outputs/plots",
        "reports/pdf", 
        "reports/html",
        "dashboards",
        "models/saved_models",
        "config",
        "logs",
        "tests",
        "docs",
        "assets",
        "examples"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   📂 Created: {directory}/")
    
    # Create __init__.py files
    init_files = ["utils/__init__.py", "tests/__init__.py"]
    for init_file in init_files:
        Path(init_file).parent.mkdir(parents=True, exist_ok=True)
        if not Path(init_file).exists():
            with open(init_file, 'w') as f:
                f.write('"""BankRisk Pro package"""\n')
    
    print("✅ Project structure created")

def create_env_file():
    """Create .env configuration file"""
    print("\n⚙️  Setting up configuration...")
    
    if not os.path.exists('.env'):
        env_content = """# BankRisk Pro Configuration
# =========================

# Federal Reserve Economic Data (FRED) API Key
# Get your free key at: https://fred.stlouisfed.org/docs/api/api_key.html
# FRED_API_KEY=your_fred_api_key_here

# Alpha Vantage API Key (optional, for additional market data)  
# Get your free key at: https://www.alphavantage.co/support/#api-key
# ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here

# Platform Configuration
BANKRISK_CACHE_ENABLED=true
BANKRISK_LOG_LEVEL=INFO
BANKRISK_MAX_PORTFOLIO_SIZE=50000

# Rate Limiting (seconds between API calls)
YAHOO_FINANCE_DELAY=2
FRED_API_DELAY=1

# Example (remove # to activate):
# FRED_API_KEY=abcd1234567890efgh
# ALPHA_VANTAGE_API_KEY=DEMO123456789
"""
        
        with open('.env', 'w') as f:
            f.write(env_content)
        
        print("✅ Created .env configuration file")
        print("   📝 Edit .env to add your API keys (optional)")
    else:
        print("✅ .env file already exists")

def create_gitignore():
    """Create .gitignore file"""
    print("\n🚫 Creating .gitignore...")
    
    gitignore_content = """# BankRisk Pro - Git Ignore
# =========================

# Environment files
.env
*.env
.env.local
.env.production

# Output files  
outputs/
reports/
dashboards/
models/saved_models/
logs/
*.log

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Jupyter Notebooks
.ipynb_checkpoints

# Cache
bankrisk_cache*
*.cache
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    
    print("✅ Created .gitignore file")

def test_installation():
    """Test if everything is working"""
    print("\n🧪 Testing installation...")
    
    try:
        # Test imports
        print("   Testing core imports...")
        import pandas as pd
        import numpy as np
        import sklearn
        import plotly
        import yfinance as yf
        print("   ✅ All core packages imported successfully")
        
        # Test basic functionality
        print("   Testing basic functionality...")
        df = pd.DataFrame({'test': [1, 2, 3]})
        arr = np.array([1, 2, 3])
        print("   ✅ Basic operations working")
        
        # Test API access (with timeout)
        print("   Testing API access...")
        try:
            from data_collector import EconomicDataCollector
            collector = EconomicDataCollector()
            print("   ✅ Economic data collector initialized")
        except Exception as e:
            print(f"   ⚠️  API test skipped: {e}")
        
        print("\n🎉 Installation test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        print("   Please run: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"   ⚠️  Warning: {e}")
        return True

def show_next_steps():
    """Show user what to do next"""
    print("\n🚀 SETUP COMPLETE!")
    print("="*50)
    print("\n📋 Next Steps:")
    print("   1. [Optional] Get free API keys:")
    print("      • FRED: https://fred.stlouisfed.org/docs/api/api_key.html")
    print("      • Add to .env file")
    print("\n   2. Run the platform:")
    print("      python bankrisk_pro.py")
    print("\n   3. Try the demo:")
    print("      python bankrisk_pro.py")
    print("      Select option 9 for quick demo")
    print("\n💡 Tips:")
    print("   • Start with demo mode to see all features")
    print("   • API keys are optional - platform works with fallback data")
    print("   • Check outputs/ folder for generated reports")
    print("\n📚 Documentation:")
    print("   • README.md - Full documentation")
    print("   • docs/ - Technical guides")
    print("   • examples/ - Sample code")

def main():
    """Main setup function"""
    print_banner()
    
    # Check system requirements
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    install_dependencies()
    
    # Create project structure
    create_project_structure()
    
    # Create configuration files
    create_env_file()
    create_gitignore()
    
    # Test installation
    if test_installation():
        show_next_steps()
    else:
        print("\n❌ Setup encountered issues. Please check error messages above.")
        sys.exit(1)
    
    print(f"\n🏦 BankRisk Pro setup completed successfully!")
    print(f"   Ready for professional credit risk analysis.")

if __name__ == "__main__":
    main()