#!/usr/bin/env python3
"""Download the Qwen3.5-0.8B Q8_0 GGUF model from HuggingFace Hub."""
import sys
from pathlib import Path

URL = "https://huggingface.co/bartowski/Qwen_Qwen3.5-0.8B-GGUF/resolve/main/Qwen_Qwen3.5-0.8B-Q8_0.gguf"
DEST = Path(__file__).resolve().parent / "model.gguf"


def main():
    try:
        import httpx
    except ImportError:
        print("httpx not installed. Run: pip install httpx", file=sys.stderr)
        sys.exit(1)

    if DEST.exists():
        size_mb = DEST.stat().st_size / 1e6
        print(f"{DEST.name} already exists ({size_mb:.0f} MB). Overwrite? [y/N] ", end="", flush=True)
        if input().strip().lower() != "y":
            print("Aborted.")
            return
        DEST.unlink()

    print(f"Downloading {URL}")
    with httpx.stream("GET", URL, follow_redirects=True, timeout=httpx.Timeout(connect=30, read=None, write=30, pool=30)) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        done = 0
        mb = 1024 * 1024
        with open(DEST, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=mb):
                f.write(chunk)
                done += len(chunk)
                pct = (done / total * 100) if total else 0
                print(f"\r  {done / mb:.0f} / {total / mb:.0f} MB ({pct:.0f}%)", end="", flush=True)
        print()

    print(f"Done: {DEST} ({DEST.stat().st_size / mb:.0f} MB)")


if __name__ == "__main__":
    main()
