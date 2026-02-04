#!/usr/bin/env python3
"""
Host-side wrapper script to run cluster_with_weights.py inside Docker container.

Usage (from host):
    python scripts/run_cluster_with_weights.py --csv <path> --embedding-cache <path> --out-dir <path> --k <k> --canon-weight <weight> --tag <tag>

Example:
    python scripts/run_cluster_with_weights.py \
        --csv hyeonto/report_1-1/sentence_k3_normalized/sentence_clusters.csv \
        --embedding-cache hyeonto/results/sentence_embeddings_cache.npy \
        --out-dir hyeonto/report_3-1_reweighted/sentence_k3_reweighted \
        --k 3 --canon-weight 3.0 --tag sentence
"""
import subprocess
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run weighted clustering inside Docker")
    parser.add_argument("--csv", required=True, help="Path to input CSV file")
    parser.add_argument("--embedding-cache", required=True, help="Path to embedding cache (.npy)")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--k", type=int, required=True, help="Number of clusters")
    parser.add_argument("--canon-weight", type=float, default=3.0, help="Weight for Canon books")
    parser.add_argument("--tag", required=True, help="Data type tag (sentence/phrase)")

    args = parser.parse_args()

    print(f"[Host] Running weighted clustering for {args.tag}")
    print(f"[Host] Input CSV: {args.csv}")
    print(f"[Host] Embedding cache: {args.embedding-cache}")
    print(f"[Host] Output dir: {args.out_dir}")
    print(f"[Host] K={args.k}, Canon weight={args.canon_weight}")

    # Build docker compose exec command
    cmd = [
        "docker", "compose", "exec", "csp", "python",
        "scripts/cluster_with_weights.py",
        "--csv", args.csv,
        "--embedding-cache", args.embedding_cache,
        "--out-dir", args.out_dir,
        "--k", str(args.k),
        "--canon-weight", str(args.canon_weight),
        "--tag", args.tag
    ]

    print(f"\n[Host] Executing: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, check=False)

    if result.returncode == 0:
        print(f"\n[Host] ✓ Weighted clustering completed successfully")
        print(f"[Host] Output saved to: {args.out_dir}/")
    else:
        print(f"\n[Host] ✗ Weighted clustering failed with exit code {result.returncode}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()
