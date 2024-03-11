# scripts.py
import subprocess
import sys

def test(test_path=None):
    command = ["pytest", "--tb=long"]  + sys.argv[1:]
    subprocess.run(command)

def coverage():
    subprocess.run(["pytest", "--cov=langdspy", "--cov-report=html", "tests/"])