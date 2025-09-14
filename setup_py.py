#!/usr/bin/env python3
"""
Setup script for Advanced Face Recognition Software
Creates a complete installation with all dependencies
"""

from setuptools import setup, find_packages
import sys
import subprocess
import platform

def install_system_dependencies():
    """Install system-level dependencies based on platform."""
    system = platform.system().lower()
    
    if system == "linux":
        print("Installing Linux dependencies...")
        try:
            subprocess.run(["sudo", "apt-get", "update"], check=True)
            subprocess.run([
                "sudo", "apt-get", "install", "-y",
                "build-essential", "cmake", "libopenblas-dev", 
                "liblapack-dev", "libx11-dev", "libgtk-3-dev",
                "python3-dev", "libglib2.0-dev"
            ], check=True)
        except subprocess.CalledProcessError:
            print("Warning: Could not install system dependencies automatically")
            print("Please run: sudo apt-get install build-essential cmake libopenblas-dev liblapack-dev")
    
    elif system == "darwin":  # macOS
        print("Installing macOS dependencies...")
        try:
            subprocess.run(["brew", "install", "cmake"], check=True)
        except subprocess.CalledProcessError:
            print("Warning: Could not install cmake via brew")
            print("Please install cmake manually or via brew")
    
    elif system == "windows":
        print("Windows detected. Please ensure Visual Studio Build Tools are installed.")
        print("Download from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019")

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required")
        sys.exit(1)
    print(f"Python {sys.version} detected - OK")

def install_dependencies():
    """Install Python dependencies with error handling."""
    dependencies = [
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "dlib>=19.24.0",
        "face-recognition>=1.3.0",
        "scipy>=1.10.0",
        "click>=8.0.0"
    ]
    
    for dep in dependencies:
        try:
            print(f"Installing {dep}...")
            subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error installing {dep}: {e}")
            if "dlib" in dep:
                print("dlib installation failed. This is common. Trying alternative installation...")
                try:
                    subprocess.run([sys.executable, "-m", "pip", "install", "dlib", "--no-cache-dir"], check=True)
                except subprocess.CalledProcessError:
                    print("Alternative dlib installation also failed.")
                    print("Please install dlib manually or use conda: conda install -c conda-forge dlib")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="advanced-face-recognition",
    version="1.0.0",
    author="AI Assistant",
    author_email="support@example.com",
    description="Advanced Face Recognition Software with GUI and CLI interfaces",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/face-recognition-software",
    py_modules=["face_recognition_system", "batch_processing"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Multimedia :: Video :: Capture",
        "Topic :: Security",
    ],
    python_requires=">=3.7",
    install_requires=[
        "face-recognition>=1.3.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "scipy>=1.10.0",
        "click>=8.0.0",
        "dlib>=19.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "gpu": [
            "tensorflow-gpu>=2.13.0",
            "opencv-contrib-python>=4.8.0",
        ],
        "web": [
            "flask>=2.3.0",
            "flask-cors>=4.0.0",
            "gunicorn>=21.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "face-recognition=face_recognition_system:main",
            "face-recognition-batch=batch_processing:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords="face recognition, computer vision, opencv, dlib, artificial intelligence, security, attendance",
    project_urls={
        "Bug Reports": "https://github.com/your-repo/face-recognition-software/issues",
        "Source": "https://github.com/your-repo/face-recognition-software",
        "Documentation": "https://github.com/your-repo/face-recognition-software/wiki",
    },
)

if __name__ == "__main__":
    print("=== Advanced Face Recognition Software Installation ===")
    print()
    
    # Check Python version
    check_python_version()
    
    # Install system dependencies
    install_system_dependencies()
    
    # Install Python dependencies
    print("\nInstalling Python dependencies...")
    install_dependencies()
    
    print("\n=== Installation Complete ===")
    print("You can now run the software using:")
    print("  python face_recognition_system.py --gui")
    print("  python face_recognition_system.py --webcam")
    print("  face-recognition --gui  (if installed via pip)")
    print()
    print("For help and documentation:")
    print("  python face_recognition_system.py --help")
    print("  python batch_processing.py --help")
