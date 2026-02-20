"""
Setup for notebooks-for-test: Ensures the NanoRust library is compiled
and installed in the ACTIVE virtual environment.

Usage (first cell of every notebook):
    %run _setup.py

Requirements:
    1. Create venv: python -m venv .venv
    2. Install kernel: python -m ipykernel install --user --name nanorust
    3. Select 'nanorust' kernel in Jupyter
"""
import os, sys, subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def setup_all():
    """Build and install nano_rust_py into the active venv."""

    # Verify we're in a venv
    if sys.prefix == sys.base_prefix:
        print("⚠️  WARNING: Not running inside a virtual environment!")
        print("   Create one: python -m venv .venv")
        print("   Activate:   .venv\\Scripts\\activate  (Windows)")
        print("   Install:    pip install maturin numpy torch torchvision jupyter ipykernel")
        print("   Register:   python -m ipykernel install --user --name nanorust")

    # Add scripts to path for nano_rust_utils
    scripts_dir = os.path.join(PROJECT_ROOT, 'scripts')
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    # Set CARGO_TARGET_DIR outside OneDrive to avoid file locking
    cargo_target = os.path.join(os.path.expanduser('~'), '.nanorust_target')
    os.environ['CARGO_TARGET_DIR'] = cargo_target

    # Build with maturin
    print(f"🔨 Building nano_rust_py (target: {cargo_target})...")
    result = subprocess.run(
        [sys.executable, '-m', 'maturin', 'develop', '--release'],
        cwd=PROJECT_ROOT,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"❌ Build failed:\n{result.stderr}")
        raise RuntimeError("maturin develop failed")

    print("✅ nano_rust_py built and installed")

    # Verify import
    import nano_rust_py
    print(f"✅ nano_rust_py imported from: {nano_rust_py.__file__}")
    return True
