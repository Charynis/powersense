"""
PowerSense Pro -- push locally trained model artifacts to a running
instance (e.g. Render) via POST /models/upload.

Companion to train_powersense_pytorch.py: run that first to populate
models/trained/, then run this to get the result live on the deployed app
without a redeploy (main.py's /models/upload hot-reloads every registered
device's models as soon as a file lands).

Usage:
    python upload_models.py --url https://your-app.onrender.com
    python upload_models.py --url https://your-app.onrender.com --device fan01

Reads UPLOAD_API_KEY from .env if the deployed instance has one set (see
main.py -- it's optional there, so this is too).
"""

import argparse
import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()

MODEL_DIR = os.environ.get(
    "MODEL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "trained")
)
UPLOAD_API_KEY = os.environ.get("UPLOAD_API_KEY", "")

SHARED_FILES = {"anomaly_scaler.pkl", "forecast_scaler.pkl"}


def _device_id_for(filename: str) -> str:
    """Best-effort device_id from e.g. anomaly_fan01.pt -> fan01. Purely
    informational on the server side, so an empty guess is harmless."""
    stem = os.path.splitext(filename)[0]
    for prefix in ("anomaly_threshold_", "anomaly_", "forecast_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return ""


def upload_file(base_url: str, path: str) -> None:
    filename = os.path.basename(path)
    headers = {"X-Upload-Key": UPLOAD_API_KEY} if UPLOAD_API_KEY else {}
    with open(path, "rb") as fh:
        resp = httpx.post(
            f"{base_url.rstrip('/')}/models/upload",
            files={"file": (filename, fh)},
            data={"device_id": _device_id_for(filename)},
            headers=headers,
            timeout=60.0,
        )
    resp.raise_for_status()
    print(f"Uploaded {filename} -> {resp.json()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload trained models to a running PowerSense Pro instance")
    parser.add_argument("--url", required=True, help="Base URL of the deployed app, e.g. https://your-app.onrender.com")
    parser.add_argument(
        "--device", action="append", dest="devices",
        help="Only upload files for this device_id (repeatable). Default: every .pt/.pkl in models/trained/.",
    )
    args = parser.parse_args()

    if not os.path.isdir(MODEL_DIR):
        print(f"No model directory at {MODEL_DIR} -- run train_powersense_pytorch.py first.")
        sys.exit(1)

    files = sorted(f for f in os.listdir(MODEL_DIR) if f.endswith((".pt", ".pkl")))
    if args.devices:
        files = [
            f for f in files
            if f in SHARED_FILES or _device_id_for(f) in args.devices
        ]

    if not files:
        print(f"No matching .pt/.pkl files found in {MODEL_DIR}.")
        sys.exit(1)

    for filename in files:
        upload_file(args.url, os.path.join(MODEL_DIR, filename))

    print(f"Done -- uploaded {len(files)} file(s) to {args.url}.")


if __name__ == "__main__":
    main()
