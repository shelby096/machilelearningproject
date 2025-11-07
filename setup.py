"""
Setup script for the Diet Recommendation ML Project.
Run this script to set up the project environment and train models.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages."""
    print("📦 Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✅ Packages installed successfully!")

def setup_project():
    """Set up the project structure and run initial training."""
    print("🚀 Setting up Diet Recommendation ML Project...")
    
    # Install requirements
    install_requirements()
    
    # Run the pipeline
    print("🔄 Running ML pipeline...")
    subprocess.check_call([sys.executable, "run_pipeline.py"])
    
    print("✅ Project setup completed!")
    print("\n🎯 Next steps:")
    print("1. Run 'streamlit run app.py' to start the web application")
    print("2. Or run 'python -m pytest tests/' to run the test suite")
    print("3. Check the 'reports/' directory for model performance metrics")

if __name__ == "__main__":
    setup_project()
