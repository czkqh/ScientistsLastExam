from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path

TASK_ID = "WavePropagation/SeismicMomentTensorInference"
ROOT = Path(__file__).resolve().parents[4]


def main():
    p = argparse.ArgumentParser(); p.add_argument("--candidate", required=True); p.add_argument("--metrics-out", required=True); p.add_argument("--timeout", type=float, default=180.0); args = p.parse_args()
    result = {"combined_score": -1e18, "valid": 0.0}
    try:
        run = subprocess.run([sys.executable, "-m", "sle", "eval", "--allow-uncertified", "--task", TASK_ID, "--candidate", str(Path(args.candidate).resolve()), "--timeout", str(args.timeout)], cwd=str(ROOT), capture_output=True, text=True, timeout=args.timeout + 60, env={**os.environ, "PYTHONPATH": str(ROOT)})
        if run.returncode: raise RuntimeError(run.stderr[-500:])
        result.update(json.loads(run.stdout))
    except Exception as exc: result["error_message"] = str(exc)
    Path(args.metrics_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result.get(k) for k in ("combined_score", "valid")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
