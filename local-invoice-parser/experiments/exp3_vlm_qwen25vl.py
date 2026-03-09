"""
Experiment 3: Qwen2.5-VL invoice parsing via llama.cpp server.
Self-sufficient: starts llama-server automatically if not already running.

Run:
    python exp3_vlm_qwen25vl.py --image invoice.jpg
    python exp3_vlm_qwen25vl.py --image invoice.jpg --llama-url http://localhost:9090/v1
"""
import sys
import json
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from pipeline.vlm import Config, run_pipeline, print_results, save_output

parser = argparse.ArgumentParser(description="Invoice parser: Qwen2.5-VL via llama.cpp")
parser.add_argument("--image", required=True, help="Path to invoice image")
parser.add_argument("--llama-url", default="http://localhost:8080/v1", help="llama.cpp server URL")
args = parser.parse_args()

cfg = Config(vlm_backend="llamacpp", vlm_backend_kwargs={"base_url": args.llama_url})
receipt, timings = run_pipeline(args.image, cfg)

print_results(receipt, timings)

output_path = str(
    Path(__file__).resolve().parent.parent.parent
    / "reports"
    / f"exp3_vlm_qwen25vl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
)
save_output(output_path, args.image, receipt, timings)

payload = {"image": args.image, "timings": timings, "receipt": receipt.model_dump()}
print(json.dumps(payload))
