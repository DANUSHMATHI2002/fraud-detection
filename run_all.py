"""One command: generate data -> train -> rebuild dashboard -> open it in your browser.   python run_all.py"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
py = sys.executable
subprocess.run([py, "src/generate_data.py"], cwd=ROOT, check=True)
subprocess.run([py, "train.py"], cwd=ROOT / "src", check=True)
sys.path.insert(0, str(ROOT / "src"))
from build_dashboard import build  # noqa: E402

build(open_browser=True)
print("\nDone. Output files: outputs/ (metrics.json, charts), tableau/ (CSVs), dashboard/index.html")
