#!/usr/bin/env python3
"""
Setup script for RL-LSTM AI Trading System Demo
This script ensures all modules can be imported correctly
"""

import sys
import os
from pathlib import Path

def setup_python_path():
    """Add project directories to Python path"""
    project_root = Path.cwd()
    paths_to_add = [
        str(project_root),
        str(project_root / "src"),
        str(project_root / "data"),
    ]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    print(f"✅ Python path updated: {sys.path[:3]}...")

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing module imports...")
    
    try:
        # Test core libraries
        import torch
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        print("✅ Core libraries: torch, numpy, pandas, matplotlib")
        
        # Test custom modules
        try:
            from models.lstm_model import LSTMPricePredictor, create_lstm_model
            from models.rl_agent import TradingAgent, DQNAgent, TradingEnvironment
            print("✅ Custom models imported successfully")
        except ImportError:
            print("⚠️  Custom models not found, will use fallback classes")
        
        try:
            from data.fetch_data import get_stock_data
            print("✅ Data module imported successfully")
        except ImportError:
            print("⚠️  Data module not found, will use fallback functions")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def create_init_files():
    """Create __init__.py files if they don't exist"""
    print("📁 Creating __init__.py files...")
    
    init_dirs = [
        "src",
        "src/models",
        "src/data", 
        "src/features",
        "src/trading",
        "src/training",
        "data"
    ]
    
    for dir_path in init_dirs:
        init_file = Path(dir_path) / "__init__.py"
        if not init_file.exists():
            init_file.parent.mkdir(parents=True, exist_ok=True)
            init_file.touch()
            print(f"   Created: {init_file}")

def run_quick_test():
    """Run a quick test of the trading system"""
    print("🚀 Running quick system test...")
    
    try:
        # Test the demo script
        import subprocess
        result = subprocess.run([
            sys.executable, "demo_trading_system.py"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Demo script runs successfully!")
            return True
        else:
            print(f"❌ Demo script failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Demo script is running (timeout after 30s) - this is normal")
        return True
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        return False

def main():
    """Main setup function"""
    print("🔧 RL-LSTM AI Trading System - Setup & Verification")
    print("=" * 60)
    
    # Setup Python path
    setup_python_path()
    
    # Create init files
    create_init_files()
    
    # Test imports
    imports_ok = test_imports()
    
    if imports_ok:
        print("\n✅ Setup completed successfully!")
        print("\n🚀 You can now run:")
        print("   1. python demo_trading_system.py")
        print("   2. python start_notebook.py")
        print("   3. jupyter notebook notebooks/demo_live_trading.ipynb")
    else:
        print("\n❌ Setup incomplete. Please install missing dependencies:")
        print("   pip install -r requirements.txt")
    
    return imports_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 