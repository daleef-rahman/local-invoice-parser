"""
Experiment 4: MiniCPM-V invoice parsing via llama.cpp server.
Self-sufficient: starts llama-server automatically if not already running.

Run:
    python exp4_vlm_minicpm.py --image invoice.jpg
    python exp4_vlm_minicpm.py --image invoice.jpg --llama-url http://localhost:9090
"""
import sys
import json
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from pipeline.vlm import Config, run_pipeline, print_results, save_output

parser = argparse.ArgumentParser(description="Invoice parser: MiniCPM-V via llama.cpp")
parser.add_argument("--image", required=True, help="Path to invoice image")
parser.add_argument("--llama-url", default="http://localhost:8081", help="llama.cpp server URL")
args = parser.parse_args()

cfg = Config(vlm_backend="minicpmv", vlm_backend_kwargs={"base_url": args.llama_url})
receipt, timings = run_pipeline(args.image, cfg)

print_results(receipt, timings)

output_path = str(
    Path(__file__).resolve().parent.parent.parent
    / "reports"
    / f"exp4_vlm_minicpm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
)
save_output(output_path, args.image, receipt, timings)

payload = {"image": args.image, "timings": timings, "receipt": receipt.model_dump()}
print(json.dumps(payload))
