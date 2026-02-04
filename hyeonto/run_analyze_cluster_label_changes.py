#!/usr/bin/env python3
"""
Host-side wrapper script to run analyze_cluster_label_changes.py inside Docker container.

Usage (from host):
    python scripts/run_analyze_cluster_label_changes.py --csv-1-1 <path> --csv-3-1 <path> --out-dir <path> --tag <tag>

Example:
    python scripts/run_analyze_cluster_label_changes.py \
        --csv-1-1 hyeonto/report_1-1/sentence_k3_normalized/sentence_clusters.csv \
        --csv-3-1 hyeonto/report_3-1_reweighted/sentence_k3_reweighted/sentence_clusters_weighted.csv \
        --out-dir cluster_label_analysis \
        --tag sentence
"""
import subprocess
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Analyze cluster label changes inside Docker")
    parser.add_argument("--csv-1-1", required=True, help="Path to 1:1 weighted CSV")
    parser.add_argument("--csv-3-1", required=True, help="Path to 3:1 weighted CSV")
    parser.add_argument("--out-dir", required=True, help="Output directory for analysis")
    parser.add_argument("--tag", required=True, help="Data type tag (sentence/phrase)")

    args = parser.parse_args()

    print(f"[Host] Analyzing cluster label changes for {args.tag}")
    print(f"[Host] 1:1 CSV: {args.csv_1_1}")
    print(f"[Host] 3:1 CSV: {args.csv_3_1}")
    print(f"[Host] Output dir: {args.out_dir}")

    # Build docker compose exec command
    cmd = [
        "docker", "compose", "exec", "csp", "python",
        "scripts/analyze_cluster_label_changes.py",
        "--csv-1-1", args.csv_1_1,
        "--csv-3-1", args.csv_3_1,
        "--out-dir", args.out_dir,
        "--tag", args.tag
    ]

    print(f"\n[Host] Executing: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, check=False)

    if result.returncode == 0:
        print(f"\n[Host] ✓ Label change analysis completed successfully")
        print(f"[Host] Report saved to: {args.out_dir}/{args.tag}_label_change_analysis.md")
    else:
        print(f"\n[Host] ✗ Label change analysis failed with exit code {result.returncode}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()
