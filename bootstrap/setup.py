# Setup script to create a virtual environment and install dependencies for the Rough-Pricing project.

import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"

def run(cmd, cwd=ROOT):
    print(">", " ".join(map(str, cmd)))
    subprocess.check_call(list(map(str, cmd)), cwd=cwd)

def venv_python():
    if platform.system().lower().startswith("win"):
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"

def main():
    # 1) Create venv if missing
    if not VENV.exists():
        run([sys.executable, "-m", "venv", str(VENV)])

    py = venv_python()

    # 2) Upgrade tooling
    run([py, "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"])

    # 3) Install project + dev extras (from pyproject.toml)
    run([py, "-m", "pip", "install", "-e", ".[dev]"])

    # 4) Quick sanity check
    print("\nSanity check:")
    run([py, "-c", "import roughvol; print('OK: roughvol import works')"])

    print("\nDone.")
    print("Activate the venv next:")
    if platform.system().lower().startswith("win"):
        print(r"  .venv\Scripts\activate")
    else:
        print("  source .venv/bin/activate")

if __name__ == "__main__":
    main()
