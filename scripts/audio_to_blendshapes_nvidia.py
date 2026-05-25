#!/usr/bin/env python3
"""NVIDIA NIM audio-to-blendshape client stub.

This script is intentionally minimal and safe:
- reads API key only from NVIDIA_API_KEY env var
- never prints key
- does not hardcode secrets
- provides a structured place for Codex 5.3 to integrate actual endpoint calls
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Call NVIDIA NIM for blendshape inference")
    p.add_argument("--audio", required=True, help="Path to wav/mp3 audio file")
    p.add_argument("--out", required=True, help="Output JSON path for blendshape clip")
    p.add_argument(
        "--endpoint",
        default="https://integrate.api.nvidia.com/v1",
        help="NVIDIA API base endpoint",
    )
    p.add_argument(
        "--model",
        default="a2f-3d",
        help="Target model id/name to invoke",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print request metadata without calling the API",
    )
    return p.parse_args()


def build_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }


def main() -> int:
    args = parse_args()

    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key:
        print("error: NVIDIA_API_KEY is not set", file=sys.stderr)
        return 2

    audio_path = Path(args.audio)
    if not audio_path.exists() or not audio_path.is_file():
        print(f"error: audio file not found: {audio_path}", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Placeholder request shape. Update with exact A2F NIM contract as needed.
    # The runtime contract for output remains schemas/avatar_clip.schema.json.
    request_meta = {
        "endpoint": args.endpoint,
        "model": args.model,
        "audio": str(audio_path),
        "out": str(out_path),
    }

    if args.dry_run:
        print(json.dumps({"dry_run": True, **request_meta}, indent=2))
        return 0

    # NOTE: This is a scaffold call. Replace URL/payload parsing to match the
    # exact model route used in your NVIDIA account.
    url = f"{args.endpoint.rstrip('/')}/health"
    try:
        resp = requests.get(url, headers=build_headers(api_key), timeout=20)
        data = {
            "status": resp.status_code,
            "ok": resp.ok,
            "endpoint_tested": url,
            "message": "Replace scaffold call with exact A2F inference endpoint.",
        }
    except requests.RequestException as exc:
        data = {
            "status": "request_error",
            "ok": False,
            "endpoint_tested": url,
            "error": str(exc),
        }

    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
