# scripts.py

import subprocess

def test():
    subprocess.run(["pytest", "tests/"])

def coverage():
    subprocess.run(["pytest", "--cov=langdspy", "--cov-report=html", "tests/"])