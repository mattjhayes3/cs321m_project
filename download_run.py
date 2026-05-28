#!/usr/bin/env python3
"""
download_run.py — Download active-learning run data from Modal volume.

Usage:
    python download_run.py 2026-05-27T11-56-00_add_option
    python download_run.py 2026-05-27T11-56-00_add_option 2026-05-27T12-40-29_add_option
    python download_run.py --list   # List all available runs
"""

import argparse
import json
import os
import sys


def list_runs():
    """List all available runs on the Modal volume."""
    import modal
    vol = modal.Volume.from_name("benchmark-eval-results")
    
    print("Available runs on Modal volume:")
    for entry in sorted(vol.listdir("/active_loop_runs/"), key=lambda e: e.path):
        if entry.type.name == "DIRECTORY":
            run_id = entry.path.split("/")[-1]
            # Check what round files exist
            round_files = []
            try:
                for f in vol.listdir(entry.path):
                    if f.path.endswith(".json") and "round_" in f.path:
                        round_files.append(f.path.split("/")[-1])
            except Exception:
                pass
            n_rounds = len(round_files)
            has_summary = any(f.path.endswith("summary.json") 
                             for f in vol.listdir(entry.path))
            status = "✅ complete" if has_summary else f"🔄 in progress ({n_rounds} rounds)"
            print(f"  {run_id}  {status}")


def download_run(run_id: str, output_base: str = "active_loop_runs"):
    """Download a single run from Modal volume to local filesystem."""
    import modal
    vol = modal.Volume.from_name("benchmark-eval-results")
    
    remote_base = f"/active_loop_runs/{run_id}"
    local_base = os.path.join(output_base, run_id)
    os.makedirs(local_base, exist_ok=True)
    
    def download_dir(remote_dir, local_dir):
        """Recursively download a directory."""
        os.makedirs(local_dir, exist_ok=True)
        n_files = 0
        try:
            for entry in vol.listdir(remote_dir):
                name = entry.path.split("/")[-1]
                local_path = os.path.join(local_dir, name)
                
                if entry.type.name == "DIRECTORY":
                    n_files += download_dir(entry.path, local_path)
                else:
                    # Download file
                    data = b""
                    for chunk in vol.read_file(entry.path):
                        data += chunk
                    with open(local_path, "wb") as f:
                        f.write(data)
                    n_files += 1
                    size_kb = len(data) / 1024
                    print(f"    ↓ {name} ({size_kb:.1f} KB)")
        except Exception as e:
            print(f"  Error reading {remote_dir}: {e}")
        return n_files
    
    print(f"\nDownloading run: {run_id}")
    print(f"  Remote: {remote_base}")
    print(f"  Local:  {local_base}")
    
    n = download_dir(remote_base, local_base)
    print(f"\n  Downloaded {n} files to {local_base}")
    return local_base


def main():
    parser = argparse.ArgumentParser(description="Download run data from Modal")
    parser.add_argument("run_ids", nargs="*", help="Run IDs to download")
    parser.add_argument("--list", action="store_true", help="List available runs")
    parser.add_argument("--output-dir", default="active_loop_runs",
                        help="Local output directory")
    args = parser.parse_args()
    
    if args.list:
        list_runs()
        return
    
    if not args.run_ids:
        print("Error: specify run IDs to download, or use --list")
        sys.exit(1)
    
    for run_id in args.run_ids:
        download_run(run_id, args.output_dir)


if __name__ == "__main__":
    main()
