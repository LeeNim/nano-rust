"""
Shared setup logic for all NANO-RUST notebooks.
Import this at the start of any notebook:

    from _setup import setup_all, PROJECT_ROOT
    setup_all()
"""
import subprocess, sys, os, platform, time, shutil
from pathlib import Path

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
VENV_DIR = os.path.join(PROJECT_ROOT, '.venv')
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'scripts')
CARGO_BIN = os.path.join(os.path.expanduser('~'), '.cargo', 'bin')

# QUAN TRỌNG: OneDrive lock files trong target/ khi sync.
# Di chuyển CARGO_TARGET_DIR ra ngoài OneDrive để tránh os error 32.
CARGO_TARGET_DIR = os.path.join(os.path.expanduser('~'), '.nanorust_target')

IS_WINDOWS = platform.system() == 'Windows'

if IS_WINDOWS:
    VENV_PYTHON = os.path.join(VENV_DIR, 'Scripts', 'python.exe')
    VENV_PIP = os.path.join(VENV_DIR, 'Scripts', 'pip.exe')
    VENV_MATURIN = os.path.join(VENV_DIR, 'Scripts', 'maturin.exe')
    VENV_SITE = os.path.join(VENV_DIR, 'Lib', 'site-packages')
else:
    VENV_PYTHON = os.path.join(VENV_DIR, 'bin', 'python')
    VENV_PIP = os.path.join(VENV_DIR, 'bin', 'pip')
    VENV_MATURIN = os.path.join(VENV_DIR, 'bin', 'maturin')
    py_ver = f'python{sys.version_info.major}.{sys.version_info.minor}'
    VENV_SITE = os.path.join(VENV_DIR, 'lib', py_ver, 'site-packages')


def setup_venv():
    """Create virtualenv and install dependencies."""
    if not os.path.exists(VENV_DIR):
        print(f'📦 Tạo virtualenv tại {VENV_DIR}...')
        subprocess.check_call([sys.executable, '-m', 'venv', VENV_DIR])
        print('✅ Virtualenv created')
    else:
        print(f'✅ Virtualenv: {VENV_DIR}')

    os.environ['VIRTUAL_ENV'] = VENV_DIR

    for pkg in ['numpy', 'torch', 'maturin']:
        r = subprocess.run([VENV_PYTHON, '-c', f'import {pkg}'], capture_output=True)
        if r.returncode == 0:
            print(f'✅ {pkg}')
        else:
            print(f'📦 Installing {pkg}...')
            subprocess.check_call([VENV_PIP, 'install', pkg, '-q'])
            print(f'✅ {pkg} installed')

    if VENV_SITE not in sys.path:
        sys.path.insert(0, VENV_SITE)
    if SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, SCRIPTS_DIR)


def setup_rust():
    """Ensure Rust toolchain is available."""
    if CARGO_BIN not in os.environ.get('PATH', ''):
        os.environ['PATH'] = CARGO_BIN + os.pathsep + os.environ.get('PATH', '')

    if shutil.which('rustc') and shutil.which('cargo'):
        ver = subprocess.check_output(['rustc', '--version'], text=True).strip()
        print(f'✅ {ver}')
        return

    print('⚠️ Rust chưa cài! Đang tự động cài...')
    if IS_WINDOWS:
        import urllib.request
        rustup_exe = os.path.join(os.environ.get('TEMP', '.'), 'rustup-init.exe')
        urllib.request.urlretrieve('https://win.rustup.rs/x86_64', rustup_exe)
        subprocess.check_call([rustup_exe, '-y', '--default-toolchain', 'stable'])
    else:
        subprocess.check_call(
            'curl --proto =https --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y',
            shell=True
        )
    os.environ['PATH'] = CARGO_BIN + os.pathsep + os.environ.get('PATH', '')
    ver = subprocess.check_output(['rustc', '--version'], text=True).strip()
    print(f'✅ Rust cài xong: {ver}')


def build_binding():
    """Build nano-rust-py via maturin develop."""
    cargo_toml = os.path.join(PROJECT_ROOT, 'py_binding', 'Cargo.toml')
    if not os.path.exists(cargo_toml):
        raise FileNotFoundError(f'Không tìm thấy {cargo_toml}!')

    os.environ['VIRTUAL_ENV'] = VENV_DIR
    # Đặt CARGO_TARGET_DIR ngoài OneDrive để tránh file locking (os error 32)
    os.environ['CARGO_TARGET_DIR'] = CARGO_TARGET_DIR
    os.makedirs(CARGO_TARGET_DIR, exist_ok=True)
    print(f'🔧 CARGO_TARGET_DIR = {CARGO_TARGET_DIR}')

    print('🔨 Building nano-rust-py (maturin develop --release)...')
    start = time.time()

    proc = subprocess.Popen(
        [VENV_MATURIN, 'develop', '--manifest-path', cargo_toml, '--release'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=PROJECT_ROOT, env=os.environ.copy()
    )
    stdout_bytes, stderr_bytes = proc.communicate()
    elapsed = time.time() - start

    stdout_str = stdout_bytes.decode('utf-8', errors='replace') if stdout_bytes else ''
    stderr_str = stderr_bytes.decode('utf-8', errors='replace') if stderr_bytes else ''

    if proc.returncode == 0:
        print(f'✅ Build OK ({elapsed:.1f}s)')
        for line in (stdout_str + stderr_str).split('\n'):
            if any(kw in line for kw in ['Compiling nano', 'Finished', 'Installing', 'Built']):
                print(f'   {line.strip()}')
    else:
        print(f'❌ Build failed ({elapsed:.1f}s):')
        print(stderr_str)
        raise RuntimeError('maturin develop failed — xem lỗi ở trên')


def setup_all():
    """One-click: venv + deps + Rust + build. Call this at notebook start."""
    print('=' * 50)
    print('   NANO-RUST Setup')
    print('=' * 50)
    setup_venv()
    print()
    setup_rust()
    print()
    build_binding()
    print()
    print('🎉 Setup complete! Ready to use.')
    print('=' * 50)
