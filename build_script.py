#!/usr/bin/env python3
"""
Build and Package Script for Advanced Face Recognition Software
Creates distributable packages and executables
"""

import os
import sys
import subprocess
import shutil
import platform
import zipfile
import json
import datetime
from pathlib import Path
import argparse

class FaceRecognitionBuilder:
    """Build and package the face recognition software."""
    
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.build_dir = self.root_dir / "build"
        self.dist_dir = self.root_dir / "dist"
        self.system = platform.system().lower()
        
    def clean_build(self):
        """Clean previous build artifacts."""
        print("🧹 Cleaning build artifacts...")
        
        dirs_to_clean = [self.build_dir, self.dist_dir, "*.egg-info"]
        for dir_pattern in dirs_to_clean:
            if "*" in str(dir_pattern):
                # Handle glob patterns
                import glob
                for path in glob.glob(str(self.root_dir / dir_pattern)):
                    if os.path.exists(path):
                        shutil.rmtree(path)
                        print(f"  Removed: {path}")
            else:
                if dir_pattern.exists():
                    shutil.rmtree(dir_pattern)
                    print(f"  Removed: {dir_pattern}")
        
        print("✅ Build artifacts cleaned")
    
    def check_dependencies(self):
        """Check if all required dependencies are installed."""
        print("🔍 Checking dependencies...")
        
        required_packages = [
            "face_recognition",
            "opencv-python",
            "numpy",
            "Pillow",
            "dlib"
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                print(f"  ✅ {package}")
            except ImportError:
                print(f"  ❌ {package}")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
            print("Install them with: pip install " + " ".join(missing_packages))
            return False
        
        print("✅ All dependencies satisfied")
        return True
    
    def install_build_tools(self):
        """Install required build tools."""
        print("🔧 Installing build tools...")
        
        build_tools = ["pyinstaller", "setuptools", "wheel", "twine"]
        
        for tool in build_tools:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", tool], 
                             check=True, capture_output=True)
                print(f"  ✅ {tool}")
            except subprocess.CalledProcessError as e:
                print(f"  ❌ Failed to install {tool}: {e}")
                return False
        
        print("✅ Build tools installed")
        return True
    
    def create_executable(self):
        """Create standalone executable using PyInstaller."""
        print("📦 Creating standalone executable...")
        
        try:
            # PyInstaller command
            cmd = [
                "pyinstaller",
                "--onefile",
                "--windowed",
                "--name", "FaceRecognition",
                "--icon", "icon.ico" if os.path.exists("icon.ico") else None,
                "--add-data", "requirements.txt;.",
                "--hidden-import", "cv2",
                "--hidden-import", "face_recognition",
                "--hidden-import", "numpy",
                "--hidden-import", "PIL",
                "--hidden-import", "dlib",
                "face_recognition_system.py"
            ]
            
            # Remove None values
            cmd = [arg for arg in cmd if arg is not None]
            
            # Add system-specific options
            if self.system == "windows":
                cmd.extend(["--console"])  # Keep console for debugging
            
            subprocess.run(cmd, check=True)
            print("✅ Executable created successfully")
            
            # Move executable to dist folder with proper name
            exe_name = "FaceRecognition.exe" if self.system == "windows" else "FaceRecognition"
            src_exe = self.dist_dir / exe_name
            
            if src_exe.exists():
                final_name = f"FaceRecognition-{platform.machine()}-{self.system}"
                if self.system == "windows":
                    final_name += ".exe"
                
                final_path = self.dist_dir / final_name
                shutil.move(src_exe, final_path)
                print(f"✅ Executable renamed to: {final_name}")
                return final_path
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create executable: {e}")
            return None
        except FileNotFoundError:
            print("❌ PyInstaller not found. Installing...")
            if self.install_build_tools():
                return self.create_executable()
            return None
    
    def create_wheel_package(self):
        """Create Python wheel package."""
        print("🎯 Creating wheel package...")
        
        try:
            subprocess.run([sys.executable, "setup.py", "bdist_wheel"], check=True)
            print("✅ Wheel package created")
            
            # Find created wheel
            wheel_files = list(self.dist_dir.glob("*.whl"))
            if wheel_files:
                return wheel_files[0]
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create wheel: {e}")
            return None
    
    def create_source_distribution(self):
        """Create source distribution."""
        print("📄 Creating source distribution...")
        
        try:
            subprocess.run([sys.executable, "setup.py", "sdist"], check=True)
            print("✅ Source distribution created")
            
            # Find created tarball
            tar_files = list(self.dist_dir.glob("*.tar.gz"))
            if tar_files:
                return tar_files[0]
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create source distribution: {e}")
            return None
    
    def create_portable_package(self):
        """Create portable package with all files."""
        print("💼 Creating portable package...")
        
        portable_dir = self.build_dir / "portable"
        portable_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy main files
        files_to_include = [
            "face_recognition_system.py",
            "batch_processing.py",
            "requirements.txt",
            "README.md",
            "config.yaml",
            "example_usage.py",
            "test_suite.py"
        ]
        
        for file_name in files_to_include:
            src_file = self.root_dir / file_name
            if src_file.exists():
                shutil.copy2(src_file, portable_dir / file_name)
                print(f"  ✅ Copied: {file_name}")
        
        # Create examples directory
        examples_dir = portable_dir / "examples"
        examples_dir.mkdir(exist_ok=True)
        
        # Create test images directory
        test_images_dir = portable_dir / "test_images"
        test_images_dir.mkdir(exist_ok=True)
        
        # Create installation script
        install_script = self.create_install_script(portable_dir)
        
        # Create run script
        run_script = self.create_run_script(portable_dir)
        
        # Create ZIP package
        zip_name = f"FaceRecognition-Portable-{platform.machine()}-{self.system}.zip"
        zip_path = self.dist_dir / zip_name
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in portable_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(portable_dir)
                    zipf.write(file_path, arcname)
        
        print(f"✅ Portable package created: {zip_name}")
        return zip_path
    
    def create_install_script(self, target_dir):
        """Create installation script for portable package."""
        script_content = f"""#!/usr/bin/env python3
'''
Installation script for Face Recognition Software
'''

import subprocess
import sys
import os

def install_dependencies():
    print("Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {{e}}")
        return False

def main():
    print("🔍 Face Recognition Software Installation")
    print("=" * 50)
    
    if not install_dependencies():
        print("Installation failed. Please check your Python environment.")
        return False
    
    print("\\n✅ Installation completed successfully!")
    print("\\nTo run the software:")
    print("  python face_recognition_system.py --gui")
    print("  python face_recognition_system.py --webcam")
    print("\\nFor help:")
    print("  python face_recognition_system.py --help")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
"""
        
        script_name = "install.py"
        script_path = target_dir / script_name
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable on Unix systems
        if self.system != "windows":
            os.chmod(script_path, 0o755)
        
        print(f"  ✅ Created: {script_name}")
        return script_path
    
    def create_run_script(self, target_dir):
        """Create run script for easy execution."""
        if self.system == "windows":
            script_content = """@echo off
echo Starting Face Recognition Software...
python face_recognition_system.py --gui
pause
"""
            script_name = "run.bat"
        else:
            script_content = """#!/bin/bash
echo "Starting Face Recognition Software..."
python3 face_recognition_system.py --gui
"""
            script_name = "run.sh"
        
        script_path = target_dir / script_name
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable on Unix systems
        if self.system != "windows":
            os.chmod(script_path, 0o755)
        
        print(f"  ✅ Created: {script_name}")
        return script_path
    
    def create_installer(self):
        """Create system-specific installer."""
        print("📋 Creating system installer...")
        
        if self.system == "windows":
            return self.create_windows_installer()
        elif self.system == "darwin":
            return self.create_macos_installer()
        elif self.system == "linux":
            return self.create_linux_installer()
        else:
            print(f"❌ Installer not supported for {self.system}")
            return None
    
    def create_windows_installer(self):
        """Create Windows installer using Inno Setup (if available)."""
        try:
            # Check if Inno Setup is available
            inno_setup_path = None
            possible_paths = [
                r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
                r"C:\Program Files\Inno Setup 6\ISCC.exe"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    inno_setup_path = path
                    break
            
            if not inno_setup_path:
                print("❌ Inno Setup not found. Please install Inno Setup to create Windows installer.")
                return None
            
            # Create Inno Setup script
            iss_content = self.create_inno_setup_script()
            iss_file = self.build_dir / "installer.iss"
            
            with open(iss_file, 'w') as f:
                f.write(iss_content)
            
            # Compile installer
            subprocess.run([inno_setup_path, str(iss_file)], check=True)
            print("✅ Windows installer created")
            
            return self.dist_dir / "FaceRecognitionSetup.exe"
            
        except Exception as e:
            print(f"❌ Failed to create Windows installer: {e}")
            return None
    
    def create_inno_setup_script(self):
        """Create Inno Setup script content."""
        return f"""[Setup]
AppName=Face Recognition Software
AppVersion=1.0.0
AppPublisher=Advanced Face Recognition
DefaultDirName={{pf}}\\FaceRecognition
DefaultGroupName=Face Recognition
OutputDir={self.dist_dir}
OutputBaseFilename=FaceRecognitionSetup
Compression=lzma
SolidCompression=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{{cm:CreateDesktopIcon}}"; GroupDescription: "{{cm:AdditionalIcons}}"; Flags: unchecked

[Files]
Source: "face_recognition_system.py"; DestDir: "{{app}}"; Flags: ignoreversion
Source: "batch_processing.py"; DestDir: "{{app}}"; Flags: ignoreversion
Source: "requirements.txt"; DestDir: "{{app}}"; Flags: ignoreversion
Source: "README.md"; DestDir: "{{app}}"; Flags: ignoreversion

[Icons]
Name: "{{group}}\\Face Recognition"; Filename: "{{app}}\\face_recognition_system.py"
Name: "{{commondesktop}}\\Face Recognition"; Filename: "{{app}}\\face_recognition_system.py"; Tasks: desktopicon

[Run]
Filename: "{{app}}\\face_recognition_system.py"; Parameters: "--gui"; Description: "{{cm:LaunchProgram,Face Recognition}}"; Flags: nowait postinstall skipifsilent
"""
    
    def create_macos_installer(self):
        """Create macOS installer package."""
        print("🍎 Creating macOS installer package...")
        # This would require additional tools like pkgbuild
        # For now, return the portable package
        return self.create_portable_package()
    
    def create_linux_installer(self):
        """Create Linux installer package."""
        print("🐧 Creating Linux installer package...")
        # This could create .deb or .rpm packages
        # For now, return the portable package
        return self.create_portable_package()
    
    def generate_build_report(self, artifacts):
        """Generate build report."""
        report = {
            "build_date": datetime.datetime.now().isoformat(),
            "system": f"{platform.system()} {platform.release()}",
            "architecture": platform.machine(),
            "python_version": sys.version,
            "artifacts": []
        }
        
        for artifact_type, artifact_path in artifacts.items():
            if artifact_path and artifact_path.exists():
                file_size = artifact_path.stat().st_size
                report["artifacts"].append({
                    "type": artifact_type,
                    "filename": artifact_path.name,
                    "path": str(artifact_path),
                    "size_bytes": file_size,
                    "size_mb": round(file_size / (1024 * 1024), 2)
                })
        
        report_file = self.dist_dir / "build_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Build report saved to: {report_file}")
        return report
    
    def create_documentation_package(self):
        """Create documentation package."""
        print("📚 Creating documentation package...")
        
        docs_dir = self.build_dir / "docs"
        docs_dir.mkdir(exist_ok=True)
        
        # Copy documentation files
        doc_files = [
            "README.md",
            "config.yaml",
            "example_usage.py"
        ]
        
        for doc_file in doc_files:
            src = self.root_dir / doc_file
            if src.exists():
                shutil.copy2(src, docs_dir / doc_file)
        
        # Create HTML documentation (basic)
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Face Recognition Software Documentation</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1, h2, h3 { color: #333; }
        pre { background: #f5f5f5; padding: 15px; border-radius: 5px; }
        .code { background: #f0f0f0; padding: 2px 4px; border-radius: 3px; }
    </style>
</head>
<body>
    <h1>Face Recognition Software Documentation</h1>
    <h2>Quick Start</h2>
    <p>To start the GUI interface:</p>
    <pre>python face_recognition_system.py --gui</pre>
    
    <h2>Command Line Usage</h2>
    <p>Process webcam feed:</p>
    <pre>python face_recognition_system.py --webcam</pre>
    
    <p>Process image:</p>
    <pre>python face_recognition_system.py --image path/to/image.jpg</pre>
    
    <p>Register new face:</p>
    <pre>python face_recognition_system.py --register "Name" path/to/photo.jpg</pre>
    
    <h2>Batch Processing</h2>
    <p>Process multiple images:</p>
    <pre>python batch_processing.py --images /path/to/images/ --output /path/to/output/</pre>
    
    <h2>Configuration</h2>
    <p>Edit <span class="code">config.yaml</span> to customize settings.</p>
    
    <h2>Troubleshooting</h2>
    <ul>
        <li>Ensure all dependencies are installed: <span class="code">pip install -r requirements.txt</span></li>
        <li>Check camera permissions for webcam usage</li>
        <li>For GPU acceleration, install appropriate CUDA libraries</li>
    </ul>
</body>
</html>
"""
        
        html_file = docs_dir / "documentation.html"
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        # Create docs ZIP
        docs_zip = self.dist_dir / "FaceRecognition-Documentation.zip"
        with zipfile.ZipFile(docs_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in docs_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(docs_dir)
                    zipf.write(file_path, arcname)
        
        print(f"✅ Documentation package created: {docs_zip.name}")
        return docs_zip
    
    def run_tests(self):
        """Run test suite before building."""
        print("🧪 Running test suite...")
        
        try:
            result = subprocess.run([sys.executable, "test_suite.py"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ All tests passed")
                return True
            else:
                print("❌ Some tests failed:")
                print(result.stdout)
                print(result.stderr)
                return False
                
        except FileNotFoundError:
            print("⚠️ Test suite not found, skipping tests")
            return True
    
    def validate_build_environment(self):
        """Validate build environment."""
        print("🔧 Validating build environment...")
        
        issues = []
        
        # Check Python version
        if sys.version_info < (3, 7):
            issues.append("Python 3.7 or higher required")
        
        # Check disk space (basic check)
        if self.dist_dir.exists():
            stat = shutil.disk_usage(self.dist_dir)
            free_gb = stat.free / (1024**3)
            if free_gb < 1:
                issues.append("Less than 1GB free disk space")
        
        # Check if main files exist
        required_files = ["face_recognition_system.py", "requirements.txt"]
        for file_name in required_files:
            if not (self.root_dir / file_name).exists():
                issues.append(f"Missing required file: {file_name}")
        
        if issues:
            print("❌ Build environment issues:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        
        print("✅ Build environment validated")
        return True
    
    def build_all(self, include_tests=True, include_installer=False):
        """Build all package types."""
        print("🚀 Starting complete build process...")
        print("=" * 50)
        
        # Validate environment
        if not self.validate_build_environment():
            return False
        
        # Clean previous builds
        self.clean_build()
        
        # Check dependencies
        if not self.check_dependencies():
            return False
        
        # Run tests
        if include_tests and not self.run_tests():
            print("❌ Build aborted due to test failures")
            return False
        
        # Create dist directory
        self.dist_dir.mkdir(exist_ok=True)
        
        # Build artifacts
        artifacts = {}
        
        print("\n🏗️ Building artifacts...")
        
        # Create executable
        artifacts["executable"] = self.create_executable()
        
        # Create wheel package
        artifacts["wheel"] = self.create_wheel_package()
        
        # Create source distribution
        artifacts["source"] = self.create_source_distribution()
        
        # Create portable package
        artifacts["portable"] = self.create_portable_package()
        
        # Create documentation
        artifacts["documentation"] = self.create_documentation_package()
        
        # Create installer (optional)
        if include_installer:
            artifacts["installer"] = self.create_installer()
        
        # Generate build report
        report = self.generate_build_report(artifacts)
        
        # Summary
        print("\n" + "=" * 50)
        print("📦 BUILD SUMMARY")
        print("=" * 50)
        
        successful_builds = [k for k, v in artifacts.items() if v is not None]
        failed_builds = [k for k, v in artifacts.items() if v is None]
        
        print(f"✅ Successful builds: {len(successful_builds)}")
        for build_type in successful_builds:
            artifact_path = artifacts[build_type]
            if artifact_path and artifact_path.exists():
                size_mb = artifact_path.stat().st_size / (1024 * 1024)
                print(f"  - {build_type}: {artifact_path.name} ({size_mb:.1f} MB)")
        
        if failed_builds:
            print(f"❌ Failed builds: {len(failed_builds)}")
            for build_type in failed_builds:
                print(f"  - {build_type}")
        
        print(f"\n📊 Total artifacts: {len(successful_builds)}")
        print(f"📁 Output directory: {self.dist_dir}")
        
        return len(successful_builds) > 0


def main():
    """Main build script entry point."""
    parser = argparse.ArgumentParser(description="Build Face Recognition Software")
    parser.add_argument("--clean-only", action="store_true", help="Only clean build artifacts")
    parser.add_argument("--no-tests", action="store_true", help="Skip running tests")
    parser.add_argument("--installer", action="store_true", help="Create system installer")
    parser.add_argument("--type", choices=["executable", "wheel", "source", "portable", "all"], 
                       default="all", help="Build type")
    
    args = parser.parse_args()
    
    builder = FaceRecognitionBuilder()
    
    if args.clean_only:
        builder.clean_build()
        return
    
    if args.type == "all":
        success = builder.build_all(
            include_tests=not args.no_tests,
            include_installer=args.installer
        )
    else:
        # Build specific type
        builder.clean_build()
        if not builder.check_dependencies():
            return
        
        builder.dist_dir.mkdir(exist_ok=True)
        
        if args.type == "executable":
            artifact = builder.create_executable()
        elif args.type == "wheel":
            artifact = builder.create_wheel_package()
        elif args.type == "source":
            artifact = builder.create_source_distribution()
        elif args.type == "portable":
            artifact = builder.create_portable_package()
        
        success = artifact is not None
        
        if success:
            print(f"✅ {args.type} build completed: {artifact.name}")
        else:
            print(f"❌ {args.type} build failed")
    
    if success:
        print("\n🎉 Build process completed successfully!")
        print("You can find the built packages in the 'dist' directory.")
    else:
        print("\n❌ Build process failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()