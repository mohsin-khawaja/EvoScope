#!/usr/bin/env python3
"""
Start Jupyter Notebook with proper environment setup for the RL-LSTM Trading System
"""

import subprocess
import sys
import os
from pathlib import Path

def setup_environment():
    """Setup the Python path for the notebook"""
    # Get the current directory (project root)
    project_root = Path.cwd()
    
    # Add project root and src to Python path
    python_paths = [
        str(project_root),
        str(project_root / "src"),
        str(project_root / "data"),
    ]
    
    # Set PYTHONPATH environment variable
    current_path = os.environ.get('PYTHONPATH', '')
    new_paths = ':'.join(python_paths)
    
    if current_path:
        os.environ['PYTHONPATH'] = f"{new_paths}:{current_path}"
    else:
        os.environ['PYTHONPATH'] = new_paths
    
    print(f"✅ PYTHONPATH set to: {os.environ['PYTHONPATH']}")

def start_jupyter():
    """Start Jupyter notebook server"""
    print("🚀 Starting Jupyter Notebook...")
    print("📁 Project structure:")
    print(f"   Root: {Path.cwd()}")
    print(f"   Notebooks: {Path.cwd() / 'notebooks'}")
    
    # Change to project root
    os.chdir(Path.cwd())
    
    # Start Jupyter notebook
    try:
        subprocess.run([
            sys.executable, "-m", "jupyter", "notebook",
            "--notebook-dir=.",
            "--ip=localhost",
            "--port=8888",
            "--no-browser",
            "--allow-root"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting Jupyter: {e}")
        return False
    except KeyboardInterrupt:
        print("\n🛑 Jupyter notebook server stopped")
        return True

def main():
    """Main function"""
    print("🧠 RL-LSTM AI Trading System - Jupyter Notebook Launcher")
    print("=" * 60)
    
    # Setup environment
    setup_environment()
    
    # Check if virtual environment is activated
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
    else:
        print("⚠️  No virtual environment detected. Consider activating .venv")
    
    # Check if required packages are installed
    try:
        import torch
        import pandas
        import numpy
        import matplotlib
        print("✅ Core packages available")
    except ImportError as e:
        print(f"❌ Missing packages: {e}")
        print("Run: pip install -r requirements.txt")
        return False
    
    # Start Jupyter
    return start_jupyter()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 